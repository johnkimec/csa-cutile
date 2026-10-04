"""The batched module matches csa_reference, including chunked decode."""

from __future__ import annotations

import pytest
import torch

from csa import CSA, CSAConfig, csa_reference, random_params


def _case(*, n: int, batch: int, seed: int, n_win: int = 0, rope_dim: int = 0):
    cfg = CSAConfig(
        m=4,
        k=2,
        n_h=4,
        n_h_i=2,
        d_c=16,
        c=8,
        c_i=4,
        n_win=n_win,
        rope_dim=rope_dim,
    )
    g = torch.Generator().manual_seed(seed)
    d = 24
    h = torch.empty(batch, n, d).normal_(generator=g)
    params = random_params(cfg=cfg, d=d, generator=g)
    module = CSA.from_params(cfg, params, d)
    return cfg, h, params, module


def _reference_batch(h: torch.Tensor, cfg: CSAConfig, params) -> torch.Tensor:
    rows = [csa_reference(h=h[b], cfg=cfg, p=params)["y"] for b in range(h.shape[0])]
    return torch.stack(rows, dim=0)


def _chunked(module: CSA, h: torch.Tensor, chunks: list[int]) -> torch.Tensor:
    cache = None
    pieces = []
    offset = 0
    for size in chunks:
        y, cache = module(h[:, offset : offset + size], cache)
        pieces.append(y)
        offset += size
    assert offset == h.shape[1]
    assert cache is not None
    assert cache.n_seen == h.shape[1]
    assert cache.pending_c_a.shape[1] == 0
    assert cache.c_comp_attn.shape[1] == h.shape[1] // module.cfg.m
    return torch.cat(pieces, dim=1)


def test_one_shot_matches_reference():
    cfg, h, params, module = _case(n=16, batch=2, seed=0)
    y, cache = module(h)
    assert torch.allclose(y, _reference_batch(h, cfg, params), atol=1e-5)
    assert cache.n_seen == 16
    assert cache.c_comp_attn.shape[1] == 4


def test_token_decode_matches_reference():
    cfg, h, params, module = _case(n=16, batch=1, seed=1)
    y = _chunked(module, h, [1] * h.shape[1])
    assert torch.allclose(y, _reference_batch(h, cfg, params), atol=1e-5)


def test_uneven_chunks_match_reference():
    """A partial block stays buffered across the chunk boundary."""
    cfg, h, params, module = _case(n=16, batch=2, seed=2, n_win=3, rope_dim=4)
    y = _chunked(module, h, [6, 1, 5, 4])
    assert torch.allclose(y, _reference_batch(h, cfg, params), atol=1e-5)


def test_batch_rows_do_not_mix():
    cfg, h, params, module = _case(n=12, batch=2, seed=3, n_win=2, rope_dim=4)
    y, _ = module(h)
    for b in range(2):
        alone, _ = module(h[b : b + 1])
        assert torch.allclose(y[b], alone[0], atol=1e-5)
        assert torch.allclose(alone[0], csa_reference(h=h[b], cfg=cfg, p=params)["y"], atol=1e-5)


def test_cache_batch_mismatch_raises():
    _, h, _, module = _case(n=8, batch=1, seed=4)
    _, cache = module(h[:, :4])
    with pytest.raises(ValueError, match="batch"):
        module(torch.cat([h, h], dim=0), cache)


def test_forward_is_differentiable():
    _, h, _, module = _case(n=8, batch=1, seed=5, n_win=2)
    y, _ = module(h)
    y.sum().backward()
    assert module.w_uq.grad is not None
    assert module.w_a_kv.grad is not None
    assert torch.isfinite(module.w_uq.grad).all()
