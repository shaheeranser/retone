"""Enforces the module dependency direction from specs/architecture.md section 4."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "retone"

# Each module may import only these project modules; third-party limits are checked below.
ALLOWED_PROJECT_IMPORTS = {
    "executor": set(),
    "inference": {"executor"},
    "eval": {"executor"},
    "data": {"executor"},
    "train": {"data", "executor"},
    "cli": {"inference", "executor"},
}
ML_FRAMEWORKS = {"torch", "transformers", "unsloth", "trl", "peft", "datasets"}
TRAIN_ONLY = {"data", "train"}


def imported_roots(module: str) -> tuple[set[str], set[str]]:
    """Return (project modules, third-party roots) imported anywhere in a package."""
    project: set[str] = set()
    external: set[str] = set()
    for path in (SRC / module).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                parts = name.split(".")
                if parts[0] == "retone" and len(parts) > 1:
                    project.add(parts[1])
                elif parts[0] != "retone" and parts[0] not in sys.stdlib_module_names:
                    external.add(parts[0])
    project.discard(module)
    return project, external


@pytest.mark.parametrize("module", sorted(ALLOWED_PROJECT_IMPORTS))
def test_project_imports_follow_dependency_direction(module: str) -> None:
    project, _ = imported_roots(module)
    assert project <= ALLOWED_PROJECT_IMPORTS[module]


@pytest.mark.parametrize("module", sorted(set(ALLOWED_PROJECT_IMPORTS) - TRAIN_ONLY))
def test_only_data_and_train_use_ml_frameworks(module: str) -> None:
    _, external = imported_roots(module)
    assert not external & ML_FRAMEWORKS
