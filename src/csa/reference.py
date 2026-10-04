"""Pure-PyTorch reference for Compressed Sparse Attention (DeepSeek-V4 §2.3).

Equations 9–19, the grouped output projection, and the §2.3.3 details
(RMSNorm, partial RoPE, sliding-window KV, attention sink). Written for
readability rather than speed. See docs/derivation.md for the paper-to-code map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class CSAConfig:
    """Hyperparameters for one CSA layer.

    m       : compression block size (paper)
    k       : top-k blocks selected by the lightning indexer
    n_h     : core-attention query heads
    n_h_i   : indexer query heads
    d_c     : shared low-rank query latent dim
    c       : core-attention head dim
    c_i     : indexer head dim
    """

    m: int
    k: int
    n_h: int
    n_h_i: int
    d_c: int
    c: int
    c_i: int
    causal: bool = True
    scale: Literal["sqrt_c", "none"] = "sqrt_c"
    # Grouped output projection (§2.3.1). n_h must be divisible by n_groups.
    # d_g is the per-group intermediate width; None picks group_width // 2.
    n_groups: int = 1
    d_g: int | None = None
    # §2.3.3. rope_dim 0 disables RoPE. V4 uses the last 64 channels.
    rope_dim: int = 0
    rope_base: float = 10_000.0
    # Recent uncompressed KV entries concatenated into core attention. 0 disables.
    n_win: int = 0
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.n_groups < 1 or self.n_h % self.n_groups != 0:
            raise ValueError(f"n_groups={self.n_groups} must be a positive divisor of n_h={self.n_h}")
        group_width = self.c * (self.n_h // self.n_groups)
        if group_width < 2:
            raise ValueError(f"group width c*n_h/n_groups={group_width} is too small for a grouped projection")
        if self.d_g is None:
            object.__setattr__(self, "d_g", group_width // 2)
        if not isinstance(self.d_g, int) or not (1 <= self.d_g < group_width):
            raise ValueError(f"d_g={self.d_g} must satisfy 1 <= d_g < group width {group_width}")
        if self.rope_dim < 0 or self.rope_dim % 2 != 0:
            raise ValueError(f"rope_dim={self.rope_dim} must be a non-negative even integer")
        if self.rope_dim > self.c or self.rope_dim > self.c_i:
            raise ValueError(f"rope_dim={self.rope_dim} must be <= c={self.c} and <= c_i={self.c_i}")
        if self.n_win < 0:
            raise ValueError(f"n_win={self.n_win} must be >= 0")
        if self.rope_base <= 0:
            raise ValueError(f"rope_base={self.rope_base} must be positive")
        if self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps={self.rms_norm_eps} must be positive")


@dataclass(frozen=True)
class CSAParams:
    # KV compression projections (eqs. 9–10).
    w_a_kv: torch.Tensor  # [d, c]
    w_b_kv: torch.Tensor  # [d, c]
    w_a_z: torch.Tensor  # [d, c]
    w_b_z: torch.Tensor  # [d, c]
    b_a: torch.Tensor  # [m, c]
    b_b: torch.Tensor  # [m, c]

    # Indexer keys compression (same overlapped compressor, but with c_i dim).
    w_a_k_i: torch.Tensor  # [d, c_i]
    w_b_k_i: torch.Tensor  # [d, c_i]
    w_a_z_i: torch.Tensor  # [d, c_i]
    w_b_z_i: torch.Tensor  # [d, c_i]
    b_a_i: torch.Tensor  # [m, c_i]
    b_b_i: torch.Tensor  # [m, c_i]

    # Indexer low-rank query path (eqs. 13–16).
    w_dq: torch.Tensor  # [d, d_c]
    w_iuq: torch.Tensor  # [d_c, c_i * n_h_i]
    w_w: torch.Tensor  # [d, n_h_i]

    # Core attention query up-projection (eq. 18).
    w_uq: torch.Tensor  # [d_c, c * n_h]

    # RMSNorm scales, applied per head-dim before core attention (§2.3.3).
    w_q_norm: torch.Tensor  # [c]
    w_kv_norm: torch.Tensor  # [c]

    # Per-head attention-sink logits (eq. 27).
    sink: torch.Tensor  # [n_h]

    # Grouped output projection. group_width = c * n_h / n_groups.
    w_oa: torch.Tensor  # [n_groups, group_width, d_g]
    w_ob: torch.Tensor  # [n_groups * d_g, d]

    # Sliding-window uncompressed KV (§2.3.3). Unused when n_win == 0.
    w_kv_win: torch.Tensor  # [d, c]
    w_kv_win_norm: torch.Tensor  # [c]


def _check_2d(name: str, x: torch.Tensor) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must be rank-2 [n, d], got shape={tuple(x.shape)}")


def _check_shape(name: str, x: torch.Tensor, shape: tuple[int | None, ...]) -> None:
    if x.ndim != len(shape):
        raise ValueError(f"{name} must have ndim={len(shape)}, got {x.ndim} (shape={tuple(x.shape)})")
    for i, (got, exp) in enumerate(zip(x.shape, shape, strict=True)):
        if exp is not None and got != exp:
            raise ValueError(f"{name} shape mismatch at dim {i}: got {got}, expected {exp}; shape={tuple(x.shape)}")


def _compress_overlapped(
    *,
    c_a: torch.Tensor,
    c_b: torch.Tensor,
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    b_a: torch.Tensor,
    b_b: torch.Tensor,
    m: int,
) -> torch.Tensor:
    """Overlapped KV compression (eqs. 11-12). Returns `[n // m, c]`.

    The b-stream of the first output block is padded with -inf logits and zero
    values, per the paper. Requires `n` divisible by `m`.
    """
    _check_2d("c_a", c_a)
    _check_2d("c_b", c_b)
    _check_2d("z_a", z_a)
    _check_2d("z_b", z_b)
    _check_2d("b_a", b_a)
    _check_2d("b_b", b_b)
    if c_a.shape != c_b.shape or c_a.shape != z_a.shape or z_a.shape != z_b.shape:
        raise ValueError(
            "c_a, c_b, z_a, z_b must all have identical shape [n, c]; "
            f"got c_a={tuple(c_a.shape)} c_b={tuple(c_b.shape)} z_a={tuple(z_a.shape)} z_b={tuple(z_b.shape)}"
        )
    n, c = c_a.shape
    if b_a.shape != (m, c) or b_b.shape != (m, c):
        raise ValueError(f"b_a and b_b must be [m, c]=[{m}, {c}], got b_a={tuple(b_a.shape)} b_b={tuple(b_b.shape)}")
    if n % m != 0:
        raise ValueError(f"sequence length n={n} must be divisible by m={m} for reference implementation")

    n_blk = n // m

    # [n_blk, m, c]
    c_a_blk = c_a.view(n_blk, m, c)
    c_b_blk = c_b.view(n_blk, m, c)
    z_a_blk = z_a.view(n_blk, m, c)
    z_b_blk = z_b.view(n_blk, m, c)

    # Previous-block padding for i=0: logits=-inf, values=0.
    c_b_prev, z_b_prev = _shift_b_stream(c_b_blk, z_b_blk)
    return _compress_from_streams(c_a=c_a_blk, c_b_prev=c_b_prev, z_a=z_a_blk, z_b_prev=z_b_prev, b_a=b_a, b_b=b_b)


def _shift_b_stream(c_b_blk: torch.Tensor, z_b_blk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block i reads block i-1's b-stream. Block 0 is zero values and -inf logits."""
    n_blk = c_b_blk.shape[0]
    neg_inf = torch.finfo(z_b_blk.dtype).min if z_b_blk.dtype.is_floating_point else -1e9
    c_b_prev = torch.empty_like(c_b_blk)
    z_b_prev = torch.empty_like(z_b_blk)
    c_b_prev[0] = 0
    z_b_prev[0] = neg_inf
    if n_blk > 1:
        c_b_prev[1:] = c_b_blk[:-1]
        z_b_prev[1:] = z_b_blk[:-1]
    return c_b_prev, z_b_prev


def _compress_from_streams(
    *,
    c_a: torch.Tensor,
    c_b_prev: torch.Tensor,
    z_a: torch.Tensor,
    z_b_prev: torch.Tensor,
    b_a: torch.Tensor,
    b_b: torch.Tensor,
) -> torch.Tensor:
    """One overlapped compression step (eqs. 11–12).

    Token axis is -2, so this accepts `[m, c]` or `[batch, m, c]` (and the
    vectorized `[n_blk, m, c]` layout). Biases are `[m, c]` and broadcast.
    Returns the compressed vector with the token axis removed.
    """
    # logits: [..., 2m, c]. Softmax over the 2m row axis (eq. 11).
    logits = torch.cat([z_a + b_a, z_b_prev + b_b], dim=-2)
    s = torch.softmax(logits, dim=-2)
    m = c_a.shape[-2]
    s_a, s_b = s[..., :m, :], s[..., m:, :]
    # Hadamard-weighted sum over the token axis (eq. 12).
    return (s_a * c_a).sum(dim=-2) + (s_b * c_b_prev).sum(dim=-2)


def _b_stream_padding(like: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero values and -inf logits with shape `[..., m, c]`, matching block 0."""
    neg_inf = torch.finfo(like.dtype).min if like.dtype.is_floating_point else -1e9
    c_pad = torch.zeros(*like.shape[:-1], m, like.shape[-1], device=like.device, dtype=like.dtype)
    z_pad = torch.empty_like(c_pad)
    z_pad.fill_(neg_inf)
    return c_pad, z_pad


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, *, eps: float) -> torch.Tensor:
    """RMSNorm over the last dimension, with a learnable per-channel scale.

    Matches the usual DeepSeek form: x * rsqrt(mean(x^2) + eps) * weight,
    computed in fp32. `weight` has shape `[x.shape[-1]]`.
    """
    _check_shape("weight", weight, (x.shape[-1],))
    xf = x.float()
    scale = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * scale * weight.float()).to(dtype=x.dtype)


def _rope_cos_sin(
    positions: torch.Tensor,
    rope_dim: int,
    base: float,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved-RoPE cos/sin for `positions`, expanded to `rope_dim` channels.

    Frequencies follow the usual θ_i = base^(-2i/dim) over pairs. The returned
    tensors have shape `positions.shape + (rope_dim,)` and are ready for
    `_apply_rope` (no extra repeat). `rope_dim == 0` returns empty trailing dims.
    """
    if rope_dim == 0:
        empty = torch.empty(*positions.shape, 0, device=positions.device, dtype=dtype)
        return empty, empty
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim={rope_dim} must be even")
    pair = torch.arange(0, rope_dim, 2, device=positions.device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (pair / rope_dim))
    freqs = positions.to(dtype=torch.float32).unsqueeze(-1) * inv_freq
    cos = freqs.cos().repeat_interleave(2, dim=-1)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Interleaved half-rotation: (x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply interleaved RoPE to the trailing `cos.shape[-1]` channels of `x`.

    Leading channels are left alone. `cos` and `sin` broadcast over any head
    dimension sitting between the position axis and the channel axis.
    """
    rope_dim = cos.shape[-1]
    if rope_dim == 0:
        return x
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = (rope.float() * cos.float()) + (_rotate_half(rope).float() * sin.float())
    return torch.cat([nope, rotated.to(dtype=x.dtype)], dim=-1)


def _core_attn_mqa(
    *,
    q: torch.Tensor,  # [n_h, c]
    kv: torch.Tensor,  # [k, c]
    sink: torch.Tensor,  # [n_h]
    scale: Literal["sqrt_c", "none"] = "sqrt_c",
) -> torch.Tensor:
    """Multi-query core attention (eq. 19) with an attention sink (eq. 27).

    Returns `[n_h, c]`. The sink logit is an extra softmax column that is
    dropped before the value mix, so the attention weights sum to less than 1.
    """
    _check_shape("q", q, (None, None))
    _check_shape("kv", kv, (None, None))
    _check_shape("sink", sink, (q.shape[0],))
    if q.shape[1] != kv.shape[1]:
        raise ValueError(f"q and kv must share last dim c; got q={tuple(q.shape)} kv={tuple(kv.shape)}")
    if kv.shape[0] == 0:
        return torch.zeros_like(q)
    c = q.shape[1]
    logits = q @ kv.T  # [n_h, k]
    if scale == "sqrt_c":
        logits = logits / (c**0.5)
    # eq. 27 — Exp(z'_h) joins the denominator. Sink is not scaled by 1/sqrt(c).
    combined = torch.cat([logits, sink.to(dtype=logits.dtype).unsqueeze(-1)], dim=-1)
    attn = torch.softmax(combined, dim=-1)[..., :-1]
    return attn @ kv  # [n_h, c]


def _grouped_output(o: torch.Tensor, w_oa: torch.Tensor, w_ob: torch.Tensor) -> torch.Tensor:
    """Grouped output projection (§2.3.1). `o` is `[n, n_h, c]` and the result is `[n, d]`.

    Heads are split into `g` groups. Each group is projected to `d_g`, then the
    concatenated intermediate is projected to the model hidden size.
    """
    n = o.shape[0]
    _check_shape("w_oa", w_oa, (None, None, None))
    g, group_width, d_g = w_oa.shape
    if o.reshape(n, -1).shape[-1] != g * group_width:
        raise ValueError(f"o flattened width {o.shape[1] * o.shape[2]} != n_groups * group_width {g * group_width}")
    _check_shape("w_ob", w_ob, (g * d_g, None))
    grouped = o.reshape(n, g, group_width)
    mid = torch.einsum("ngi,gio->ngo", grouped, w_oa)
    return mid.reshape(n, g * d_g) @ w_ob


@torch.no_grad()
def csa_reference(
    *,
    h: torch.Tensor,  # [n, d]
    cfg: CSAConfig,
    p: CSAParams,
) -> dict[str, torch.Tensor]:
    """CSA forward for a single sequence.

    Returns a dict with `c_comp [n_blk, c]` and `k_i_comp [n_blk, c_i]` (the
    compression outputs, before RMSNorm and RoPE), `i_scores [n, n_blk]`
    (masked with -inf for non-visible blocks; RoPE is included when
    `cfg.rope_dim > 0`), `topk_idx [n, k]` (-1 padding for early tokens with
    fewer than k visible blocks), `o [n, n_h, c]` (core-attention outputs
    after the inverse-RoPE countermeasure), and `y [n, d]` (grouped output
    projection of `o`).
    """
    _check_2d("h", h)
    n, d = h.shape
    m = cfg.m
    if n % m != 0:
        raise ValueError(f"n={n} must be divisible by m={m} for reference implementation")
    n_blk = n // m

    # eqs. 9–10
    _check_shape("p.w_a_kv", p.w_a_kv, (d, cfg.c))
    _check_shape("p.w_b_kv", p.w_b_kv, (d, cfg.c))
    _check_shape("p.w_a_z", p.w_a_z, (d, cfg.c))
    _check_shape("p.w_b_z", p.w_b_z, (d, cfg.c))
    c_a = h @ p.w_a_kv
    c_b = h @ p.w_b_kv
    z_a = h @ p.w_a_z
    z_b = h @ p.w_b_z

    c_comp = _compress_overlapped(c_a=c_a, c_b=c_b, z_a=z_a, z_b=z_b, b_a=p.b_a, b_b=p.b_b, m=m)

    # Indexer keys compression (paper: "same compression operation used for C_comp").
    _check_shape("p.w_a_k_i", p.w_a_k_i, (d, cfg.c_i))
    _check_shape("p.w_b_k_i", p.w_b_k_i, (d, cfg.c_i))
    _check_shape("p.w_a_z_i", p.w_a_z_i, (d, cfg.c_i))
    _check_shape("p.w_b_z_i", p.w_b_z_i, (d, cfg.c_i))
    k_i_a = h @ p.w_a_k_i
    k_i_b = h @ p.w_b_k_i
    z_i_a = h @ p.w_a_z_i
    z_i_b = h @ p.w_b_z_i
    k_i_comp = _compress_overlapped(
        c_a=k_i_a, c_b=k_i_b, z_a=z_i_a, z_b=z_i_b, b_a=p.b_a_i, b_b=p.b_b_i, m=m
    )  # [n_blk, c_i]

    # eqs. 13–16: produce indexer queries + weights, then scores against preceding blocks.
    _check_shape("p.w_dq", p.w_dq, (d, cfg.d_c))
    _check_shape("p.w_iuq", p.w_iuq, (cfg.d_c, cfg.c_i * cfg.n_h_i))
    _check_shape("p.w_w", p.w_w, (d, cfg.n_h_i))

    c_q = h @ p.w_dq  # [n, d_c] (eq. 13)
    q_i = (c_q @ p.w_iuq).view(n, cfg.n_h_i, cfg.c_i)  # [n, n_h_i, c_i] (eq. 14)
    w_i = h @ p.w_w  # [n, n_h_i] (eq. 15)

    # Partial RoPE (§2.3.3) on indexer queries and compressed indexer keys.
    # Compressed entry i is placed at absolute position i * m, the first token
    # of that block. The paper specifies the rotation but not this index; i*m
    # is the position used by the V4 attention code.
    token_pos = torch.arange(n, device=h.device)
    blk_pos = torch.arange(n_blk, device=h.device) * m
    cos_t, sin_t = _rope_cos_sin(token_pos, cfg.rope_dim, cfg.rope_base, dtype=h.dtype)
    cos_b, sin_b = _rope_cos_sin(blk_pos, cfg.rope_dim, cfg.rope_base, dtype=h.dtype)
    q_i = _apply_rope(q_i, cos_t[:, None, :], sin_t[:, None, :])
    k_i_rope = _apply_rope(k_i_comp, cos_b, sin_b)

    # Scores I_{t,s} for s < floor(t/m) (eq. 16).
    # dot: [n, n_h_i, n_blk]
    dot = torch.einsum("tnc,sc->tns", q_i, k_i_rope)  # q·K
    dot = torch.relu(dot)
    i_scores = torch.einsum("tn,tns->ts", w_i, dot)  # sum_h w * relu(dot)

    # Mask non-visible blocks.
    if cfg.causal:
        blk_of_t = torch.arange(n, device=h.device, dtype=torch.int64) // m  # floor(t/m)
        s_idx = torch.arange(n_blk, device=h.device, dtype=torch.int64).view(1, n_blk)
        visible = s_idx < blk_of_t.view(n, 1)
        i_scores = i_scores.masked_fill(~visible, float("-inf"))

    # eq. 17: top-k selection over I_{t,:}
    k = cfg.k
    if k > n_blk:
        raise ValueError(f"cfg.k={k} must be <= n_blk={n_blk}")
    topk_val, topk_idx = torch.topk(i_scores, k=k, dim=1)  # [n, k]
    # When a token has fewer than k visible blocks (early tokens), topk includes -inf; mark those as invalid.
    valid = torch.isfinite(topk_val)
    topk_idx = torch.where(valid, topk_idx, torch.full_like(topk_idx, -1))

    # eq. 18: core attention queries from shared c_q
    _check_shape("p.w_uq", p.w_uq, (cfg.d_c, cfg.c * cfg.n_h))
    _check_shape("p.w_q_norm", p.w_q_norm, (cfg.c,))
    _check_shape("p.w_kv_norm", p.w_kv_norm, (cfg.c,))
    _check_shape("p.sink", p.sink, (cfg.n_h,))
    q = (c_q @ p.w_uq).view(n, cfg.n_h, cfg.c)  # [n, n_h, c]

    # §2.3.3: RMSNorm each query head and the compressed KV head, then RoPE
    # on the trailing rope_dim channels, just before core attention.
    q = _rms_norm(q, p.w_q_norm, eps=cfg.rms_norm_eps)
    c_comp_attn = _rms_norm(c_comp, p.w_kv_norm, eps=cfg.rms_norm_eps)
    q = _apply_rope(q, cos_t[:, None, :], sin_t[:, None, :])
    c_comp_attn = _apply_rope(c_comp_attn, cos_b, sin_b)

    # Sliding-window branch (§2.3.3): n_win uncompressed KV entries for the
    # most recent tokens, including the current one. Separate projection from
    # the compressor. Empty when n_win == 0.
    _check_shape("p.w_kv_win", p.w_kv_win, (d, cfg.c))
    _check_shape("p.w_kv_win_norm", p.w_kv_win_norm, (cfg.c,))
    kv_win = _rms_norm(h @ p.w_kv_win, p.w_kv_win_norm, eps=cfg.rms_norm_eps)
    kv_win = _apply_rope(kv_win, cos_t, sin_t)

    # eq. 19: MQA over the sliding-window KV and the selected compressed blocks.
    o = torch.empty((n, cfg.n_h, cfg.c), device=h.device, dtype=h.dtype)
    for t in range(n):
        parts: list[torch.Tensor] = []
        if cfg.n_win > 0:
            start = max(0, t - cfg.n_win + 1)
            parts.append(kv_win[start : t + 1])
        idx = topk_idx[t]
        idx = idx[idx >= 0]
        if idx.numel() > 0:
            parts.append(c_comp_attn.index_select(0, idx))
        if not parts:
            o[t].zero_()
            continue
        kv_t = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        o[t] = _core_attn_mqa(q=q[t], kv=kv_t, sink=p.sink, scale=cfg.scale)

    # K and V are the same rotated vectors, so o carries absolute positions.
    # RoPE at -t (cos, -sin at the query position) puts the mix back in terms
    # of the query-to-KV distance (§2.3.3).
    o = _apply_rope(o, cos_t[:, None, :], -sin_t[:, None, :])

    group_width = cfg.c * (cfg.n_h // cfg.n_groups)
    _check_shape("p.w_oa", p.w_oa, (cfg.n_groups, group_width, cfg.d_g))
    _check_shape("p.w_ob", p.w_ob, (cfg.n_groups * cfg.d_g, d))
    y = _grouped_output(o, p.w_oa, p.w_ob)

    return {
        "c_comp": c_comp,
        "k_i_comp": k_i_comp,
        "i_scores": i_scores,
        "topk_idx": topk_idx,
        "o": o,
        "y": y,
    }


def random_params(
    *,
    cfg: CSAConfig,
    d: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    std: float = 0.02,
) -> CSAParams:
    """CSAParams with N(0, std^2) projections and biases. For tests and demos.

    RMSNorm scales start at 1. The sink logit starts at 0, which still adds
    exp(0) = 1 to the core-attention denominator.
    """

    def randn(*shape: int) -> torch.Tensor:
        t = torch.empty(*shape, device=device, dtype=dtype)
        t.normal_(mean=0.0, std=std, generator=generator)
        return t

    def ones(*shape: int) -> torch.Tensor:
        return torch.ones(*shape, device=device, dtype=dtype)

    def zeros(*shape: int) -> torch.Tensor:
        return torch.zeros(*shape, device=device, dtype=dtype)

    group_width = cfg.c * (cfg.n_h // cfg.n_groups)
    d_g = cfg.d_g
    assert d_g is not None  # set in CSAConfig.__post_init__
    return CSAParams(
        w_a_kv=randn(d, cfg.c),
        w_b_kv=randn(d, cfg.c),
        w_a_z=randn(d, cfg.c),
        w_b_z=randn(d, cfg.c),
        b_a=randn(cfg.m, cfg.c),
        b_b=randn(cfg.m, cfg.c),
        w_a_k_i=randn(d, cfg.c_i),
        w_b_k_i=randn(d, cfg.c_i),
        w_a_z_i=randn(d, cfg.c_i),
        w_b_z_i=randn(d, cfg.c_i),
        b_a_i=randn(cfg.m, cfg.c_i),
        b_b_i=randn(cfg.m, cfg.c_i),
        w_dq=randn(d, cfg.d_c),
        w_iuq=randn(cfg.d_c, cfg.c_i * cfg.n_h_i),
        w_w=randn(d, cfg.n_h_i),
        w_uq=randn(cfg.d_c, cfg.c * cfg.n_h),
        w_q_norm=ones(cfg.c),
        w_kv_norm=ones(cfg.c),
        sink=zeros(cfg.n_h),
        w_oa=randn(cfg.n_groups, group_width, d_g),
        w_ob=randn(cfg.n_groups * d_g, d),
        w_kv_win=randn(d, cfg.c),
        w_kv_win_norm=ones(cfg.c),
    )
