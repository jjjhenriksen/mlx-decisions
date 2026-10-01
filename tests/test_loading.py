"""Failure during checkpoint loading must not leak process-wide MLX settings."""

import pytest

mx = pytest.importorskip("mlx.core")


def test_load_failure_restores_allocator_cache_limit(monkeypatch, tmp_path):
    import mlx_decisions.engine as module

    def fail_load(*args, **kwargs):
        raise ValueError("invalid checkpoint")

    monkeypatch.setattr(module, "load", fail_load)
    original = mx.set_cache_limit(17 * 1024**2)
    try:
        with pytest.raises(ValueError, match="invalid checkpoint"):
            module.Engine(str(tmp_path))
        # set_cache_limit returns the old value: the loader must have restored it.
        assert mx.set_cache_limit(17 * 1024**2) == 17 * 1024**2
    finally:
        mx.set_cache_limit(original)
