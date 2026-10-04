"""Batched CSA module with an incremental KV cache.

The cache stores compressed blocks and the sliding-window KV. A block is
appended only after the token that completes it has already attended, so every
cached block satisfies ``s < floor(t / m)``. Stepping a sequence in any chunk
sizes matches ``csa_reference`` on the concatenated tokens, provided the full
length is divisible by ``m``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import nn

from csa.reference import (
    CSAConfig,
    CSAParams,
    _apply_rope,
    _b_stream_padding,
    _compress_from_streams,
    _core_attn_mqa,
    _grouped_output,
    _rms_norm,
    _rope_cos_sin,
)


def _param_shapes(cfg: CSAConfig, d: int) -> dict[str, tuple[int, ...]]:
    group_width = cfg.c * (cfg.n_h // cfg.n_groups)
    d_g = cfg.d_g
    assert d_g is not None
    return {
        "w_a_kv": (d, cfg.c),
        "w_b_kv": (d, cfg.c),
        "w_a_z": (d, cfg.c),
        "w_b_z": (d, cfg.c),
        "b_a": (cfg.m, cfg.c),
        "b_b": (cfg.m, cfg.c),
        "w_a_k_i": (d, cfg.c_i),
        "w_b_k_i": (d, cfg.c_i),
        "w_a_z_i": (d, cfg.c_i),
        "w_b_z_i": (d, cfg.c_i),
        "b_a_i": (cfg.m, cfg.c_i),
        "b_b_i": (cfg.m, cfg.c_i),
        "w_dq": (d, cfg.d_c),
        "w_iuq": (cfg.d_c, cfg.c_i * cfg.n_h_i),
        "w_w": (d, cfg.n_h_i),
        "w_uq": (cfg.d_c, cfg.c * cfg.n_h),
        "w_q_norm": (cfg.c,),
        "w_kv_norm": (cfg.c,),
        "sink": (cfg.n_h,),
        "w_oa": (cfg.n_groups, group_width, d_g),
        "w_ob": (cfg.n_groups * d_g, d),
        "w_kv_win": (d, cfg.c),
        "w_kv_win_norm": (cfg.c,),
    }


@dataclass
class CSACache:
    """Incremental state for one CSA layer. Every row of the batch shares ``n_seen``.

    ``pending_*`` holds the current unfinished block (length ``< m``).
    ``overlap_*`` is the previous block's b-stream; ``None`` means block 0,
    which is padded with zero values and ``-inf`` logits. ``c_comp_attn`` and
    ``k_i_rope`` are the compressed entries after the same norm / RoPE the
    full forward applies before attention. ``kv_win`` is the sliding window
    after RMSNorm and RoPE, oldest token first, length at most ``n_win``.
    """

    n_seen: int
    pending_c_a: torch.Tensor
    pending_c_b: torch.Tensor
    pending_z_a: torch.Tensor
    pending_z_b: torch.Tensor
    pending_k_a: torch.Tensor
    pending_k_b: torch.Tensor
    pending_kz_a: torch.Tensor
    pending_kz_b: torch.Tensor
    overlap_c_b: torch.Tensor | None
    overlap_z_b: torch.Tensor | None
    overlap_k_b: torch.Tensor | None
    overlap_kz_b: torch.Tensor | None
    c_comp_attn: torch.Tensor
    k_i_rope: torch.Tensor
    kv_win: torch.Tensor

    @staticmethod
    def empty(batch: int, cfg: CSAConfig, *, device: torch.device, dtype: torch.dtype) -> CSACache:
        def zeros(*shape: int) -> torch.Tensor:
            return torch.zeros(*shape, device=device, dtype=dtype)

        return CSACache(
            n_seen=0,
            pending_c_a=zeros(batch, 0, cfg.c),
            pending_c_b=zeros(batch, 0, cfg.c),
            pending_z_a=zeros(batch, 0, cfg.c),
            pending_z_b=zeros(batch, 0, cfg.c),
            pending_k_a=zeros(batch, 0, cfg.c_i),
            pending_k_b=zeros(batch, 0, cfg.c_i),
            pending_kz_a=zeros(batch, 0, cfg.c_i),
            pending_kz_b=zeros(batch, 0, cfg.c_i),
            overlap_c_b=None,
            overlap_z_b=None,
            overlap_k_b=None,
            overlap_kz_b=None,
            c_comp_attn=zeros(batch, 0, cfg.c),
            k_i_rope=zeros(batch, 0, cfg.c_i),
            kv_win=zeros(batch, 0, cfg.c),
        )

    @property
    def batch(self) -> int:
        return self.pending_c_a.shape[0]


def _append_time(buf: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
    """Append a `[batch, dim]` token onto a `[batch, time, dim]` buffer."""
    return torch.cat([buf, token.unsqueeze(1)], dim=1)


def _attend_row(
    *,
    q: torch.Tensor,
    q_i: torch.Tensor,
    w_i: torch.Tensor,
    kv_win: torch.Tensor,
    c_comp: torch.Tensor,
    k_i: torch.Tensor,
    sink: torch.Tensor,
    cfg: CSAConfig,
) -> torch.Tensor:
    """Core attention for one batch row at one position. Returns `[n_h, c]`."""
    parts: list[torch.Tensor] = []
    if cfg.n_win > 0 and kv_win.shape[0] > 0:
        parts.append(kv_win)
    n_blk = c_comp.shape[0]
    if n_blk > 0:
        # eq. 16, over blocks already emitted (all of them are visible).
        dot = torch.relu(torch.einsum("hc,sc->hs", q_i, k_i))
        scores = torch.einsum("h,hs->s", w_i, dot)
        take = min(cfg.k, n_blk)
        idx = torch.topk(scores, k=take).indices
        parts.append(c_comp.index_select(0, idx))
    if not parts:
        return torch.zeros_like(q)
    kv = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    return _core_attn_mqa(q=q, kv=kv, sink=sink, scale=cfg.scale)


def _emit_block(cache: CSACache, cfg: CSAConfig, p: CSAParams) -> None:
    """Compress the finished pending block and append it to the cache."""
    m = cfg.m
    if cache.overlap_c_b is None or cache.overlap_z_b is None:
        c_b_prev, z_b_prev = _b_stream_padding(cache.pending_c_a[:, 0], m)
    else:
        c_b_prev, z_b_prev = cache.overlap_c_b, cache.overlap_z_b
    if cache.overlap_k_b is None or cache.overlap_kz_b is None:
        k_b_prev, kz_b_prev = _b_stream_padding(cache.pending_k_a[:, 0], m)
    else:
        k_b_prev, kz_b_prev = cache.overlap_k_b, cache.overlap_kz_b

    comp = _compress_from_streams(
        c_a=cache.pending_c_a,
        c_b_prev=c_b_prev,
        z_a=cache.pending_z_a,
        z_b_prev=z_b_prev,
        b_a=p.b_a,
        b_b=p.b_b,
    )
    k_comp = _compress_from_streams(
        c_a=cache.pending_k_a,
        c_b_prev=k_b_prev,
        z_a=cache.pending_kz_a,
        z_b_prev=kz_b_prev,
        b_a=p.b_a_i,
        b_b=p.b_b_i,
    )
    comp = _rms_norm(comp, p.w_kv_norm, eps=cfg.rms_norm_eps)

    block_index = cache.c_comp_attn.shape[1]
    pos = torch.full((cache.batch,), block_index * m, device=comp.device, dtype=torch.long)
    cos_b, sin_b = _rope_cos_sin(pos, cfg.rope_dim, cfg.rope_base, dtype=comp.dtype)
    comp = _apply_rope(comp, cos_b, sin_b)
    k_comp = _apply_rope(k_comp, cos_b, sin_b)

    cache.c_comp_attn = torch.cat([cache.c_comp_attn, comp.unsqueeze(1)], dim=1)
    cache.k_i_rope = torch.cat([cache.k_i_rope, k_comp.unsqueeze(1)], dim=1)
    cache.overlap_c_b = cache.pending_c_b
    cache.overlap_z_b = cache.pending_z_b
    cache.overlap_k_b = cache.pending_k_b
    cache.overlap_kz_b = cache.pending_kz_b

    empty_c = cache.pending_c_a[:, :0]
    empty_i = cache.pending_k_a[:, :0]
    cache.pending_c_a = empty_c
    cache.pending_c_b = empty_c
    cache.pending_z_a = empty_c
    cache.pending_z_b = empty_c
    cache.pending_k_a = empty_i
    cache.pending_k_b = empty_i
    cache.pending_kz_a = empty_i
    cache.pending_kz_b = empty_i


def csa_forward_cached(
    *,
    h: torch.Tensor,
    cfg: CSAConfig,
    p: CSAParams,
    cache: CSACache | None = None,
) -> tuple[torch.Tensor, CSACache]:
    """Run CSA over ``h`` of shape ``[batch, seq, d]``, updating ``cache``.

    Returns ``y`` of shape ``[batch, seq, d]`` and the cache to pass into the
    next call. Rows of a batch advance together.
    """
    if h.ndim != 3:
        raise ValueError(f"h must be rank-3 [batch, seq, d], got shape={tuple(h.shape)}")
    if not cfg.causal:
        raise ValueError("the incremental cache implements the causal CSA path")
    batch, seq, d = h.shape
    if cache is None:
        cache = CSACache.empty(batch, cfg, device=h.device, dtype=h.dtype)
    elif cache.batch != batch:
        raise ValueError(f"cache batch {cache.batch} does not match h batch {batch}")
    if seq == 0:
        return h.new_empty(batch, 0, d), cache

    y = torch.empty(batch, seq, d, device=h.device, dtype=h.dtype)
    for j in range(seq):
        t = cache.n_seen
        h_t = h[:, j]
        pos = torch.full((batch,), t, device=h.device, dtype=torch.long)
        cos_t, sin_t = _rope_cos_sin(pos, cfg.rope_dim, cfg.rope_base, dtype=h.dtype)

        c_q = h_t @ p.w_dq
        q_i = _apply_rope((c_q @ p.w_iuq).view(batch, cfg.n_h_i, cfg.c_i), cos_t[:, None, :], sin_t[:, None, :])
        w_i = h_t @ p.w_w
        q = _rms_norm((c_q @ p.w_uq).view(batch, cfg.n_h, cfg.c), p.w_q_norm, eps=cfg.rms_norm_eps)
        q = _apply_rope(q, cos_t[:, None, :], sin_t[:, None, :])

        if cfg.n_win > 0:
            kv_t = _rms_norm(h_t @ p.w_kv_win, p.w_kv_win_norm, eps=cfg.rms_norm_eps)
            kv_t = _apply_rope(kv_t, cos_t, sin_t)
            cache.kv_win = _append_time(cache.kv_win, kv_t)[:, -cfg.n_win :]

        o = torch.empty(batch, cfg.n_h, cfg.c, device=h.device, dtype=h.dtype)
        for b in range(batch):
            o[b] = _attend_row(
                q=q[b],
                q_i=q_i[b],
                w_i=w_i[b],
                kv_win=cache.kv_win[b],
                c_comp=cache.c_comp_attn[b],
                k_i=cache.k_i_rope[b],
                sink=p.sink,
                cfg=cfg,
            )
        o = _apply_rope(o, cos_t[:, None, :], -sin_t[:, None, :])
        y[:, j] = _grouped_output(o, p.w_oa, p.w_ob)

        cache.pending_c_a = _append_time(cache.pending_c_a, h_t @ p.w_a_kv)
        cache.pending_c_b = _append_time(cache.pending_c_b, h_t @ p.w_b_kv)
        cache.pending_z_a = _append_time(cache.pending_z_a, h_t @ p.w_a_z)
        cache.pending_z_b = _append_time(cache.pending_z_b, h_t @ p.w_b_z)
        cache.pending_k_a = _append_time(cache.pending_k_a, h_t @ p.w_a_k_i)
        cache.pending_k_b = _append_time(cache.pending_k_b, h_t @ p.w_b_k_i)
        cache.pending_kz_a = _append_time(cache.pending_kz_a, h_t @ p.w_a_z_i)
        cache.pending_kz_b = _append_time(cache.pending_kz_b, h_t @ p.w_b_z_i)
        cache.n_seen = t + 1
        if cache.pending_c_a.shape[1] == cfg.m:
            _emit_block(cache, cfg, p)

    return y, cache


class CSA(nn.Module):
    """One CSA layer. ``forward`` takes ``[batch, seq, d]`` and an optional cache."""

    def __init__(self, cfg: CSAConfig, d: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.d = d
        for name, shape in _param_shapes(cfg, d).items():
            self.register_parameter(name, nn.Parameter(torch.empty(*shape)))

    def as_params(self) -> CSAParams:
        return CSAParams(**{f.name: getattr(self, f.name) for f in fields(CSAParams)})

    def load_params(self, params: CSAParams) -> None:
        for f in fields(CSAParams):
            dest = getattr(self, f.name)
            src = getattr(params, f.name)
            if tuple(dest.shape) != tuple(src.shape):
                raise ValueError(f"{f.name} shape {tuple(src.shape)} != parameter {tuple(dest.shape)}")
            dest.data.copy_(src)

    @classmethod
    def from_params(cls, cfg: CSAConfig, params: CSAParams, d: int) -> CSA:
        module = cls(cfg, d)
        module.load_params(params)
        return module

    def forward(self, h: torch.Tensor, cache: CSACache | None = None) -> tuple[torch.Tensor, CSACache]:
        return csa_forward_cached(h=h, cfg=self.cfg, p=self.as_params(), cache=cache)
