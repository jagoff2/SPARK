import pytest

torch = pytest.importorskip("torch")

from spark.data import DatasetStatistics, InstructionSeedSet, materialize_instructions
from spark.demopack import CodebookSpec, DemopackCodebook, DemopackDecoder


def test_demopack_decode_matches_linear():
    spec = CodebookSpec(num_codewords=16, embedding_dim=8)
    codebook = DemopackCodebook(spec)
    stats = DatasetStatistics.synthetic(vocab_size=spec.num_codewords, sequence_length=8)
    seed_set = InstructionSeedSet.from_statistics(stats, num_layers=4)
    instructions = materialize_instructions(
        seed_set=seed_set,
        num_layers=4,
        tile_shape=(4, 8),
        codebook_size=spec.num_codewords,
    )
    layer = DemopackDecoder(codebook, out_features=16, in_features=8, instructions=instructions)
    x = torch.randn(2, 8)
    out = layer(x)
    assert out.shape == (2, 16)
