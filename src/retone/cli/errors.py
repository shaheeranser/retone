"""CLI-level errors that carry the exit code the command should return."""

from __future__ import annotations

from retone.cli.exit_codes import ExitCode


class CliError(RuntimeError):
    """A failure the top-level handler turns into a message and a stable exit code."""

    def __init__(self, message: str, code: ExitCode) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
