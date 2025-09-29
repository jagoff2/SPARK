import pytest

torch = pytest.importorskip("torch")

from spark.data import (
    METADATA_DIM,
    DatasetStatistics,
    InstructionSeedSet,
    ProceduralDataset,
    ProceduralDatasetConfig,
    deserialize_metadata_tensor,
    load_instruction_seed_set,
    materialize_instructions,
    save_instruction_seed_set,
    serialize_metadata_tensor,
)
from spark.demopack import deserialize_instruction_set, serialize_instruction_set


def test_procedural_dataset_emits_expected_shapes():
    config = ProceduralDatasetConfig(
        vocab_size=32,
        sequence_length=12,
        batch_size=4,
        num_batches=2,
        seed=123,
    )
    dataset = ProceduralDataset(config)
    batch = next(dataset.iter_batches(num_batches=1))
    tokens, metadata, targets = batch
    assert tokens.shape == (4, 12)
    assert metadata.shape == (4, METADATA_DIM)
    assert targets.shape == (4, 12)
    dataset_again = ProceduralDataset(config)
    tokens_b, metadata_b, targets_b = next(dataset_again.iter_batches(num_batches=1))
    assert torch.equal(tokens, tokens_b)
    assert torch.allclose(metadata, metadata_b)
    assert torch.equal(targets, targets_b)


def test_instruction_seed_set_materialization_roundtrip(tmp_path):
    config = ProceduralDatasetConfig(vocab_size=48, sequence_length=6, batch_size=2, seed=7)
    dataset = ProceduralDataset(config)
    stats = dataset.statistics
    seed_set = InstructionSeedSet.from_statistics(stats, num_layers=3)
    payload = seed_set.to_dict()
    restored = InstructionSeedSet.from_dict(payload)
    path = tmp_path / "seeds.json"
    save_instruction_seed_set(restored, path)
    restored = load_instruction_seed_set(path)
    instructions = materialize_instructions(
        seed_set=restored,
        num_layers=3,
        tile_shape=(2, config.sequence_length),
        codebook_size=config.vocab_size,
    )
    instructions_again = materialize_instructions(
        seed_set=restored,
        num_layers=3,
        tile_shape=(2, config.sequence_length),
        codebook_size=config.vocab_size,
    )
    for inst_a, inst_b in zip(instructions, instructions_again):
        assert torch.equal(inst_a.codeword_indices, inst_b.codeword_indices)
        assert inst_a.scale == pytest.approx(inst_b.scale)


def test_instruction_and_metadata_serialization():
    config = ProceduralDatasetConfig(vocab_size=16, sequence_length=8, batch_size=3, seed=99)
    dataset = ProceduralDataset(config)
    tokens, metadata, _ = next(dataset.iter_batches(num_batches=1))
    seed_set = InstructionSeedSet.from_statistics(dataset.statistics, num_layers=2)
    instructions = materialize_instructions(
        seed_set=seed_set,
        num_layers=2,
        tile_shape=(2, config.sequence_length),
        codebook_size=config.vocab_size,
    )
    serialized_instructions = serialize_instruction_set(instructions)
    restored_instructions = deserialize_instruction_set(serialized_instructions)
    for inst_original, inst_restored in zip(instructions, restored_instructions):
        assert torch.equal(inst_original.codeword_indices, inst_restored.codeword_indices)
        assert inst_original.scale == pytest.approx(inst_restored.scale)
        if inst_original.rotation is None:
            assert inst_restored.rotation is None
    serialized_metadata = serialize_metadata_tensor(metadata)
    restored_metadata = deserialize_metadata_tensor(serialized_metadata)
    assert restored_metadata.shape == metadata.shape
    assert torch.allclose(restored_metadata, metadata)
    stats_payload = dataset.statistics.to_dict()
    restored_stats = DatasetStatistics.from_dict(stats_payload)
    assert torch.allclose(restored_stats.to_metadata_tensor(), dataset.statistics.to_metadata_tensor())
