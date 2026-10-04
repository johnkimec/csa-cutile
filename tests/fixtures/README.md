# Fixtures

Bitwise tensors captured from DeepSeek-V4 inference are deferred. That stack is
not runnable on the Apple Silicon machine this repo is developed on. Correctness
for now is the structural tests in `tests/test_reference.py`.

The capture protocol below is notes for a later NVIDIA-GPU pass. Do not block
reference work on it.

## Capture protocol

1. Clone the inference impl from
   <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference>.
2. Run a single CSA layer at small scale (e.g. `n=128, d_model=64, m=4, k=4`)
   on a fixed-seed random hidden-state tensor.
3. Save a `csa_<config-tag>.pt` containing:
   - `h`        — input hidden states `[n, d]`
   - `params`   — projection / bias tensors as a flat dict matching `CSAParams`
   - `c_comp`   — compressed KV `[n_blk, c]`
   - `k_i_comp` — compressed indexer keys `[n_blk, c_i]`
   - `i_scores` — index scores `[n, n_blk]`
   - `topk_idx` — selected blocks `[n, k]`
   - `o`        — core attention outputs `[n, n_h, c]`
4. Add a test that loads the fixture and asserts `csa_reference(...)` matches
   each tensor to <1e-6 in FP32.
