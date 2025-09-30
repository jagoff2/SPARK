import json
from pathlib import Path

from spark.training.pipeline import (
    build_frontier_training_plan,
    materialize_plan,
)


def test_frontier_training_plan_structure(tmp_path: Path) -> None:
    plan = build_frontier_training_plan(tmp_path, seed=123, device="cuda")
    payload = plan.to_dict()

    # Four curriculum phases covering the end-to-end schedule.
    assert len(payload["phases"]) == 4

    # Resume chaining is encoded by default for phases beyond the first.
    assert payload["phases"][1]["resume_from"].endswith(
        "01_foundation_bootstrap/last.pt"
    )

    # Chat command must reference the trained checkpoint.
    chat_command = payload["chat"]["command"]
    assert "--checkpoint" in chat_command

    assets = materialize_plan(plan, output_dir=tmp_path, emit_script=True)

    # All configuration files are emitted and contain serialised TrainingConfig payloads.
    config_paths = [Path(path) for path in assets["config_paths"]]
    assert len(config_paths) == 4
    for path in config_paths:
        data = json.loads(path.read_text())
        assert data["epochs"] >= 1
        assert data["checkpoint_dir"]

    script_path = Path(assets["script_path"])
    script_text = script_path.read_text()
    assert "spark train" in script_text
    assert "spark eval" in script_text
    assert "spark chat" in script_text
