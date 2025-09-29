"""Instruction tape virtual machine for bytecode reasoning."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional


class VMError(RuntimeError):
    """Raised when the opcode VM encounters an unrecoverable error."""


class Opcode(str, Enum):
    PLAN = "PLAN"
    RETR = "RETR"
    ASSUME = "ASSUME"
    EXEC = "EXEC"
    LEMMA = "LEMMA"
    CHECK = "CHECK"
    BRANCH = "BRANCH"
    MERGE = "MERGE"
    CITE = "CITE"


@dataclass
class Instruction:
    opcode: Opcode
    argument: Optional[str] = None


@dataclass
class ExecutionState:
    stack: List[str]
    log: List[str]
    budget: int

    def push(self, value: str) -> None:
        self.stack.append(value)

    def pop(self) -> str:
        if not self.stack:
            raise VMError("Stack underflow")
        return self.stack.pop()


class OpcodeVM:
    """Minimal virtual machine executing reasoning bytecode."""

    def __init__(self, budget: int = 32) -> None:
        self.handlers: Dict[Opcode, Callable[[ExecutionState, Optional[str]], None]] = {}
        self.budget = budget
        self.register_default_handlers()

    def register_default_handlers(self) -> None:
        self.handlers = {
            Opcode.PLAN: self._handle_plan,
            Opcode.RETR: self._handle_retr,
            Opcode.ASSUME: self._handle_assume,
            Opcode.EXEC: self._handle_exec,
            Opcode.LEMMA: self._handle_lemma,
            Opcode.CHECK: self._handle_check,
            Opcode.BRANCH: self._handle_branch,
            Opcode.MERGE: self._handle_merge,
            Opcode.CITE: self._handle_cite,
        }

    def execute(self, instructions: Iterable[Instruction]) -> ExecutionState:
        state = ExecutionState(stack=[], log=[], budget=self.budget)
        for step, inst in enumerate(instructions):
            if step >= state.budget:
                state.log.append("Budget exceeded")
                break
            handler = self.handlers.get(inst.opcode)
            if handler is None:
                raise VMError(f"Unhandled opcode {inst.opcode}")
            handler(state, inst.argument)
        return state

    def _handle_plan(self, state: ExecutionState, argument: Optional[str]) -> None:
        """Register a plan and make it available for subsequent opcodes.

        The ``PLAN`` opcode typically sets up a hypothesis that later
        instructions (like ``CHECK``) will validate.  The previous
        implementation only logged the plan without storing it which meant
        that any immediate ``CHECK`` would operate on an empty stack and raise
        a ``VMError``.  By pushing the argument onto the stack we retain the
        planned hypothesis for later use while keeping the logging behaviour
        unchanged.
        """

        if argument is not None:
            state.push(argument)
        state.log.append(f"PLAN: {argument or 'no-op'}")

    def _handle_retr(self, state: ExecutionState, argument: Optional[str]) -> None:
        state.log.append(f"RETRIEVE: {argument}")
        state.push(argument or "")

    def _handle_assume(self, state: ExecutionState, argument: Optional[str]) -> None:
        state.log.append(f"ASSUME: {argument}")
        state.push(argument or "")

    def _handle_exec(self, state: ExecutionState, argument: Optional[str]) -> None:
        command = state.pop()
        state.log.append(f"EXEC: {command}")

    def _handle_lemma(self, state: ExecutionState, argument: Optional[str]) -> None:
        state.log.append(f"LEMMA: {argument}")
        state.push(argument or "")

    def _handle_check(self, state: ExecutionState, argument: Optional[str]) -> None:
        if state.stack:
            hypothesis = state.pop()
        elif argument is not None:
            hypothesis = argument
        else:
            raise VMError("No hypothesis available for CHECK")
        state.log.append(f"CHECK: {hypothesis}")

    def _handle_branch(self, state: ExecutionState, argument: Optional[str]) -> None:
        state.log.append("BRANCH")
        state.push("branch::" + (argument or ""))

    def _handle_merge(self, state: ExecutionState, argument: Optional[str]) -> None:
        merged = ", ".join(state.stack)
        state.stack.clear()
        state.push(merged)
        state.log.append(f"MERGE: {merged}")

    def _handle_cite(self, state: ExecutionState, argument: Optional[str]) -> None:
        state.log.append(f"CITE: {argument}")


__all__ = ["Opcode", "Instruction", "OpcodeVM", "VMError", "ExecutionState"]
