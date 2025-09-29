import pytest

torch = pytest.importorskip("torch")

from spark.hash_router import route_tokens


def test_route_tokens_shape():
    hidden = torch.randn(2, 4, 16)
    indices, compressed = route_tokens(hidden, sparsity=0.1, num_neurons=32)
    assert indices.shape[:2] == (2, 4)
    assert compressed.shape[:2] == (2, 4)
