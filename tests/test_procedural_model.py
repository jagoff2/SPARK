import pytest

torch = pytest.importorskip("torch")

from spark.demopack import CodebookSpec
from spark.layer_generator import GeneratorConfig
from spark.opcode_vm import Instruction, Opcode
from spark.procedural_model import ProceduralLanguageModel, ProceduralModelConfig


def test_procedural_model_forward_and_reason():
    config = ProceduralModelConfig(
        input_dim=32,
        hidden_dim=64,
        vocab_size=128,
        codebook_spec=CodebookSpec(num_codewords=64, embedding_dim=32),
        generator_config=GeneratorConfig(embed_dim=16, hidden_dim=32, rank=4),
    )
    model = ProceduralLanguageModel(config)
    inputs = torch.randn(2, 32)
    logits = model(inputs)
    assert logits.shape == (2, 128)
    trace = [Instruction(Opcode.PLAN, "demo"), Instruction(Opcode.CHECK, "demo")]
    log = model.reason(trace)
    assert isinstance(log, list)
