import pytest

torch = pytest.importorskip("torch")

from spark.attention import AttentionBasisSynthesizer, AttentionPatch, AtomConfig


def test_attention_patch_shapes():
    synthesiser = AttentionBasisSynthesizer(
        [AtomConfig("sin", 1.0), AtomConfig("cos", 2.0)],
        max_length=128,
    )
    patch = AttentionPatch(
        atom_indices=torch.tensor([0, 1]),
        gains=torch.tensor([1.0, 0.5]),
        shifts=torch.tensor([0, 1]),
        window=torch.ones(2),
    )
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    out = synthesiser.apply_attention(patch, q, k, v)
    assert out.shape == (1, 4, 8)
