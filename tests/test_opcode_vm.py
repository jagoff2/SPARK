import pytest

torch = pytest.importorskip("torch")

from spark.opcode_vm import Instruction, Opcode, OpcodeVM


def test_opcode_vm_budget():
    vm = OpcodeVM(budget=2)
    program = [Instruction(Opcode.PLAN, "start"), Instruction(Opcode.RETR, "doc"), Instruction(Opcode.CHECK, "done")]
    state = vm.execute(program)
    assert "Budget exceeded" in state.log[-1]
