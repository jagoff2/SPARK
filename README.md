# SPARK Procedural Runtime

This repository implements a proof-of-concept runtime for the SPARK proposal:
**Sparse Procedural Attention with Runtime Kernels**.  The focus is on shipping
small procedural generators that can be expanded into large computations on
commodity GPUs such as an NVIDIA RTX 5060 Ti (compute capability 12.0, CUDA
12.8).  All components are written in Python and PyTorch with Windows 10 support
in mind.

## Features

The implementation covers the core ideas outlined in the proposal:

* **Demopack Weights** – `spark.demopack` provides vector-quantised codebooks
  and fused decode+GEMM layers that synthesise weights on-demand.
* **Layer Generator** – `spark.layer_generator` implements a hypernetwork that
  generates LoRA-style low-rank deltas conditioned on metadata.
* **Procedural Attention** – `spark.attention` exposes a basis synthesiser that
  builds attention kernels from sinusoid and Legendre atoms.
* **Opcode VM** – `spark.opcode_vm` defines a bytecode interpreter for
  instruction-tape reasoning with a strict step budget.
* **KV Patcher** – `spark.kv_patcher` caches and reuses KV tensors across
  multiple decoding passes.
* **Sparse Hash Router** – `spark.hash_router` selects a sparse subset of MLP
  neurons or attention heads per token.
* **Integration** – `spark.procedural_model` wires the components together into
  a toy language model skeleton.

## Installation

The project uses a lightweight dependency set.  On Windows 10 with CUDA 12.8
installed, create an environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Note**: PyTorch wheels for CUDA 12.8 will be released closer to the official
> CUDA SDK.  If unavailable, install the latest nightly wheel matching your CUDA
> version from https://download.pytorch.org/.

## Usage

Run the unit tests to ensure everything works:

```powershell
pytest
```

A consolidated CLI is available via ``python -m spark``.  It exposes utilities
for automated evaluation and a lightweight chat demonstration:

```powershell
# Run a single evaluation sweep and emit JSON metrics
python -m spark eval --batch-size 8 --runs 3 --output results.json

# Launch the interactive chat demo (type /exit to quit)
python -m spark chat --show-trace
```

A quick interactive demo can be executed with Python:

```python
from spark.demopack import CodebookSpec
from spark.layer_generator import GeneratorConfig
from spark.procedural_model import ProceduralLanguageModel, ProceduralModelConfig
from spark.opcode_vm import Instruction, Opcode

config = ProceduralModelConfig(
    input_dim=32,
    hidden_dim=64,
    vocab_size=32000,
    codebook_spec=CodebookSpec(num_codewords=128, embedding_dim=32),
    generator_config=GeneratorConfig(embed_dim=32, hidden_dim=64, rank=4),
)
model = ProceduralLanguageModel(config)

# Forward pass with procedural weights.
inputs = torch.randn(2, 32)
logits = model(inputs)

# Execute a reasoning tape.
trace = [Instruction(Opcode.PLAN, "factor problem"), Instruction(Opcode.CHECK, "result")]
print(model.reason(trace))
```

## Testing

The repository includes a small suite of `pytest` unit tests covering the code
paths of each subsystem.  These tests execute on CPU but are written to run on
CUDA GPUs when available.

## Roadmap

Future work will focus on replacing the PyTorch reference implementations with
custom CUDA 12.8 kernels, integrating Triton-based fused decode kernels, and
adding dataset synthesis pipelines that emit procedural seeds instead of static
examples.

## Evaluation

The `spark.evaluation` module contains utilities for benchmarking the procedural
model against dense baselines.  It exposes helpers to build a comparable dense
network, measure perplexity, estimate throughput, and approximate the parameter
footprint so results can be compared under identical conditions.  Example usage:

```python
from spark.demopack import CodebookSpec
from spark.evaluation import (
    EvaluationBatch,
    build_dense_baseline,
    compare_models,
)
from spark.layer_generator import GeneratorConfig
from spark.procedural_model import ProceduralLanguageModel, ProceduralModelConfig
import torch

config = ProceduralModelConfig(
    input_dim=32,
    hidden_dim=64,
    vocab_size=512,
    codebook_spec=CodebookSpec(num_codewords=64, embedding_dim=32),
    generator_config=GeneratorConfig(embed_dim=16, hidden_dim=32, rank=4),
)
procedural = ProceduralLanguageModel(config)
dense = build_dense_baseline(config)

inputs = torch.randn(8, 32)
targets = torch.randint(0, config.vocab_size, (8,))
batch = EvaluationBatch(inputs=inputs, targets=targets)
report = compare_models(procedural, dense, batch)
print(report)
```

Additional helpers demonstrate qualitative advantages such as opcode reasoning
traces and KV-cache reuse timings once quantitative metrics are established.
