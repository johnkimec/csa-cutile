# CSA: paper to code

A walkthrough of DeepSeek-V4 §2.3.1 (Compressed Sparse Attention), with the
relevant lines in `src/csa/reference.py` cited next to each equation.

## Notation

| Symbol               | Meaning                                            | Shape       |
| -------------------- | -------------------------------------------------- | ----------- |
| `H` (`h`)            | input hidden states for one sequence               | `[n, d]`    |
| `W_a^{KV}, W_b^{KV}` | KV projections for the two streams                 | `[d, c]`    |
| `W_a^Z, W_b^Z`       | compression-weight projections                     | `[d, c]`    |
| `B^a, B^b`           | learnable positional biases                        | `[m, c]`    |
| `C^a, C^b`           | per-token KV vectors before compression            | `[n, c]`    |
| `Z^a, Z^b`           | per-token compression-weight vectors               | `[n, c]`    |
| `C^Comp`             | compressed KV entries                              | `[n/m, c]`  |
| `K_I^Comp`           | compressed indexer keys                            | `[n/m, c_i]`|
| `c_t^Q`              | shared low-rank query latent                       | `[d_c]`     |
| `q_t^I`              | indexer queries                                    | `[n_h_i, c_i]` |
| `w_t^I`              | per-head indexer weights                           | `[n_h_i]`   |
| `I_{t, s}`           | indexer score, query token `t` over block `s`      | scalar      |
| `q_t`                | core-attention queries                             | `[n_h, c]`  |
| `o_{t, i}`           | per-head core-attention output                     | `[c]`       |

## Eqs. 9–10. Two streams of per-token KV vectors and weights

`C^a = H W_a^{KV}`, `C^b = H W_b^{KV}`, and likewise for `Z^a`, `Z^b`.

The two streams exist so that block `i`'s compressed entry can pull from both
block `i` (`a` stream) and block `i-1` (`b` stream). This is the "overlapped
compression" mentioned right after eq. 12.

In code: the four `h @ p.w_*` matmuls in `csa_reference()` immediately before
the call to `_compress_overlapped`.

## Eqs. 11–12. Overlapped compression of m tokens into one entry

For each output block `i`, build a `[2m, c]` logits matrix by stacking the `m`
rows of `Z^a` from block `i` (plus bias `B^a`) on top of the `m` rows of `Z^b`
from block `i-1` (plus bias `B^b`). Row-wise softmax over the `2m` axis gives
weights `S^a, S^b`; the compressed entry is the Hadamard-weighted sum.

Boundary case `i = 0`: the previous block does not exist. The paper specifies
padding `Z^b` with `-inf` and `C^b` with zeros, which makes the b-half of the
softmax collapse and the b-sum vanish.

In code: `_compress_overlapped()`.

## Lightning indexer: compressed indexer keys

Same overlapped compressor as above, but with head dim `c_i` and a separate
set of projection / bias parameters. Produces `K_I^Comp` of shape `[n/m, c_i]`.

In code: the second `_compress_overlapped` call in `csa_reference()`.

## Eqs. 13–14. Shared low-rank query latent

`c_t^Q = h_t W^{DQ}`, then `q_t^I = c_t^Q W^{IUQ}` reshaped to `[n_h_i, c_i]`.

The latent `c_t^Q` is shared between the indexer query path (eq. 14) and the
core-attention query path (eq. 18). That sharing is what makes the indexer
cheap.

In code: `c_q = h @ p.w_dq` and `q_i = (c_q @ p.w_iuq).view(...)`.

## Eqs. 15–16. Indexer score per (query token, compressed block)

`w_t^I = h_t W^w`, then

```
I_{t, s} = sum_h w^I_{t, h} * relu(q^I_{t, h} . K^I,Comp_s)
```

Three things to notice: (i) the per-head dot product passes through ReLU
before weighting; (ii) the head weights `w_t^I` come from a separate
projection of `h_t`, not from `c_t^Q`; (iii) the score is a scalar per
`(t, s)`.

Causality (paper text): only blocks `s` with `s < floor(t / m)` are visible,
since block `floor(t / m)` overlaps positions ≥ `t`. In code we mask
non-visible blocks with `-inf` before top-k.

In code: `dot = einsum("tnc,sc->tns", q_i, k_i_comp); dot = relu(dot);
i_scores = einsum("tn,tns->ts", w_i, dot)`.

## Eq. 17. Top-k selection

`torch.topk(i_scores, k=cfg.k, dim=1)`. Tokens early in the sequence have
fewer than `k` visible blocks; their selected indices are padded with the
sentinel `-1` and filtered out before attention.

## Eqs. 18–19. Multi-Query Attention over selected blocks

`q_t = c_t^Q W^{UQ}` reshaped to `[n_h, c]`, then standard scaled-dot-product
attention with the selected compressed entries serving as both keys and
values for every head (the MQA part).

In code: the `for t in range(n)` loop calling `_core_attn_mqa(q=q[t], kv=...)`
in `csa_reference()`. Scaling defaults to `1 / sqrt(c)`. The KV set for that
call is the sliding-window entries (below) followed by the selected compressed
blocks.

## Grouped output projection

`c * n_h` is wide, so the head outputs are not projected to `d` in one matmul.
Split the `n_h` heads into `g = n_groups` groups. Project each group from
`c * n_h / g` down to `d_g`, concatenate, and project `g * d_g` to `d`.

In code: `_grouped_output()`, applied to `o` after the inverse RoPE below.
The returned tensor is `y` with shape `[n, d]`. `o` itself stays `[n, n_h, c]`.

## §2.3.3. RMSNorm before core attention

Each query head and the single compressed KV head are RMSNorm'd over the head
dimension, with a learnable per-channel scale, in fp32:

```
y = x * rsqrt(mean(x^2) + eps) * weight
```

This happens after the eq. 18 projection and after eq. 12, and before RoPE and
core attention. The indexer keys are not normalized: the paper places this
norm immediately before core attention.

In code: `_rms_norm()` on `q` with `p.w_q_norm` and on `c_comp` with
`p.w_kv_norm`. The dict still returns the pre-norm `c_comp`.

## §2.3.3. Partial RoPE

Interleaved RoPE on the trailing `rope_dim` channels (V4 uses 64; `rope_dim = 0`
turns it off). Queries sit at their token position `t`. A compressed entry `i`
sits at position `i * m`, the first token of that block — the paper requires
the rotation and the V4 code supplies this index.

Because the same vector is the key and the value, the attention output inherits
the keys' absolute positions. The countermeasure is RoPE at `-t` on that
output, implemented as `(cos_t, -sin_t)` since cosine is even and sine is odd.
After that, a key from position `s` contributes as a function of `s - t`.

The same rotation is applied to indexer queries and compressed indexer keys
before eq. 16, so the index scores see the same relative positions. Returned
`k_i_comp` is still the pre-RoPE compression output; `i_scores` includes RoPE
when `rope_dim > 0`.

In code: `_rope_cos_sin()`, `_apply_rope()`.

## §2.3.3. Sliding-window branch

Compressed attention only sees blocks `s < floor(t / m)`, so a token cannot
read the other tokens in its own block. For each query, core attention also
receives the `n_win` most recent uncompressed KV vectors, including the
current token (`j` from `t - n_win + 1` through `t`). They come from a separate
projection `h W^{win}`, then the same RMSNorm and RoPE as the compressed path.
`n_win = 0` leaves the branch empty.

In code: `kv_win` inside `csa_reference()`, concatenated in front of the
selected compressed entries.

## Eq. 27. Attention sink

Each head has a learnable logit `z'_h`. Softmax runs over the attention logits
plus that extra column, and the sink column is dropped before the value mix:

```
s_{h,j} = exp(z_{h,j}) / (sum_k exp(z_{h,k}) + exp(z'_h))
```

The retained weights sum to less than 1. The sink logit is not multiplied by
`1/sqrt(c)`. A query with no visible keys writes zeros, which is the same
result as putting all mass on the sink.

In code: the extra column in `_core_attn_mqa()`.

## Batched module and incremental cache

`CSA` in `src/csa/module.py` is the same math on `[batch, seq, d]`. `CSACache` keeps the unfinished block, the previous block's b-stream (zero / `-inf` until the first block exists), the compressed KV and indexer keys after RMSNorm and RoPE, and the sliding window.

A block is written to the cache only after the token that fills it has already attended. Every cached block then satisfies `s < floor(t / m)`, which is the same visibility rule as the single-sequence forward. Because of that, any split of a sequence into chunks — including one token at a time — produces the same `y` as `csa_reference` on the whole sequence. The full length still has to be a multiple of `m` for the last block to close. Rows of a batch advance together.

## Details this reference keeps from the paper text

Two choices in the V4 module differ from the wording used here, and this file
follows the paper:

- A block is visible only when `s < floor(t / m)`. The V4 module uses
  `(t + 1) // m`, which also lets the last token of a finished block see that
  block.
- Eq. 16 is an unscaled sum of ReLU dots. The V4 indexer additionally
  multiplies the dots by `1/sqrt(c_i)` and the head weights by `1/sqrt(n_h_i)`.
