import pytest

torch = pytest.importorskip("torch")

from spark.demopack import CodebookSpec, DemopackCodebook, DemopackDecoder, build_random_instructions


def test_demopack_decode_matches_linear():
    spec = CodebookSpec(num_codewords=16, embedding_dim=8)
    codebook = DemopackCodebook(spec)
    instructions = build_random_instructions(4, (4, 8), spec.num_codewords)
    layer = DemopackDecoder(codebook, out_features=16, in_features=8, instructions=instructions)
    x = torch.randn(2, 8)
    out = layer(x)
    assert out.shape == (2, 16)
