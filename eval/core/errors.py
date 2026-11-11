"""Error types shared by every task evaluator."""

from __future__ import annotations


class EvaluationError(RuntimeError):
    """A fatal setup, communication, or state-machine error."""


class TransitionTimeout(EvaluationError):
    """A controller transition did not finish before its deadline."""


class SafetyStop(EvaluationError):
    """Deploy reported that its safety flag is false."""


class TrialFailure(RuntimeError):
    """A recoverable failure attributed to the current trial."""

    def __init__(self, result: str, detail: str) -> None:
        super().__init__(detail)
        self.result = result
        self.detail = detail
