import pytest

torch = pytest.importorskip("torch")

from spark.kv_patcher import KVCachePatcher


def test_kv_cache_roundtrip():
    patcher = KVCachePatcher()
    key = torch.randn(1, 1, 8)
    value = torch.randn(1, 1, 8)
    patcher.update(1, key, value)
    keys, values = patcher.gather([1])
    assert torch.allclose(keys, key)
    assert torch.allclose(values, value)
    patcher.mark(1)
    patcher.sweep()
    assert 1 not in patcher.cache
