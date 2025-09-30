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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_decoder_offload_respects_request_device():
    spec = CodebookSpec(num_codewords=64, embedding_dim=16)
    codebook = DemopackCodebook(spec)
    stats = DatasetStatistics.synthetic(vocab_size=spec.num_codewords, sequence_length=16)
    seed_set = InstructionSeedSet.from_statistics(stats, num_layers=4)
    instructions = materialize_instructions(
        seed_set=seed_set,
        num_layers=4,
        tile_shape=(4, 16),
        codebook_size=spec.num_codewords,
    )
    layer = DemopackDecoder(codebook, out_features=32, in_features=16, instructions=instructions)
    layer = layer.to(device=torch.device("cuda"), dtype=torch.float16)
    layer.codebook.to(device=torch.device("cpu"), dtype=torch.float16)
    for module in layer.instructions:
        module.to(device=torch.device("cpu"), dtype=torch.float16)
    inputs = torch.randn(4, 16, device="cuda", dtype=torch.float16)
    output = layer(inputs)
    assert output.device.type == "cuda"
    assert layer.codebook.codewords.device.type == "cpu"
