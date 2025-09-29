import pytest

torch = pytest.importorskip("torch")

from spark.hash_router import route_tokens, route_tokens_with_metadata


def test_route_tokens_shape():
    hidden = torch.randn(2, 4, 16)
    indices, compressed = route_tokens(hidden, sparsity=0.1, num_neurons=32)
    assert indices.shape[:2] == (2, 4)
    assert compressed.shape[:2] == (2, 4)


def test_route_tokens_with_metadata():
    hidden = torch.randn(2, 4, 16)
    indices, weights, compressed = route_tokens_with_metadata(
        hidden, sparsity=0.25, num_neurons=8
    )
    assert indices.shape == (2, 4, 2)
    assert weights.shape == (2, 4, 2)
    assert compressed.shape == (2, 4, 2, 16)
