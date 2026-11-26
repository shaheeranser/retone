"""Quantization command construction and orchestration, without running llama.cpp."""

from __future__ import annotations

from pathlib import Path

import pytest

from retone.train.quantize import (
    MMPROJ_F16,
    MODEL_F16,
    QuantizeConfig,
    convert_command,
    mmproj_command,
    quantization_names,
    quantize_command,
    run,
)


def _config(tmp_path: Path) -> QuantizeConfig:
    return QuantizeConfig(
        model="someone/retone-merged",
        out_dir=tmp_path / "out",
        llama_cpp=tmp_path / "llama.cpp",
        python="/usr/bin/python3",
        quantize_tool=tmp_path / "llama-quantize",
    )


def test_convert_command_uses_f16(tmp_path: Path) -> None:
    config = _config(tmp_path)
    command = convert_command(config, config.out_dir / MODEL_F16)
    assert command == [
        "/usr/bin/python3",
        str(tmp_path / "llama.cpp" / "convert_hf_to_gguf.py"),
        "someone/retone-merged",
        "--outfile",
        str(config.out_dir / MODEL_F16),
        "--outtype",
        "f16",
    ]


def test_mmproj_command_only_converts_the_projector(tmp_path: Path) -> None:
    config = _config(tmp_path)
    command = mmproj_command(config, config.out_dir / MMPROJ_F16)
    assert command[-1] == "--mmproj"
    assert command[:-1] == convert_command(config, config.out_dir / MMPROJ_F16)


def test_quantize_command_names_the_tool_source_and_kind(tmp_path: Path) -> None:
    command = quantize_command(
        tmp_path / "model-f16.gguf",
        tmp_path / "model-Q4_K_M.gguf",
        "Q4_K_M",
        tool=tmp_path / "llama-quantize",
    )
    assert command == [
        str(tmp_path / "llama-quantize"),
        str(tmp_path / "model-f16.gguf"),
        str(tmp_path / "model-Q4_K_M.gguf"),
        "Q4_K_M",
    ]


def test_default_quantize_binary_lives_in_the_checkout(tmp_path: Path) -> None:
    config = QuantizeConfig(model="m", out_dir=tmp_path, llama_cpp=tmp_path / "llama.cpp")
    assert config.quantize_binary() == tmp_path / "llama.cpp" / "build" / "bin" / "llama-quantize"


def test_run_converts_then_quantizes_in_order(tmp_path: Path) -> None:
    config = _config(tmp_path)
    commands: list[list[str]] = []

    summary = run(config, runner=lambda command: commands.append(list(command)))

    assert (config.out_dir).is_dir()
    assert len(commands) == 4
    assert commands[0] == convert_command(config, config.out_dir / MODEL_F16)
    assert commands[1] == mmproj_command(config, config.out_dir / MMPROJ_F16)
    assert commands[2] == quantize_command(
        config.out_dir / MODEL_F16,
        config.out_dir / "model-Q4_K_M.gguf",
        "Q4_K_M",
        tool=config.quantize_binary(),
    )
    assert commands[3] == quantize_command(
        config.out_dir / MODEL_F16,
        config.out_dir / "model-Q8_0.gguf",
        "Q8_0",
        tool=config.quantize_binary(),
    )
    assert summary.quants == {
        "Q4_K_M": str(config.out_dir / "model-Q4_K_M.gguf"),
        "Q8_0": str(config.out_dir / "model-Q8_0.gguf"),
    }
    assert summary.mmproj == str(config.out_dir / MMPROJ_F16)


def test_quantization_names_follow_the_serve_script_convention() -> None:
    assert quantization_names(("Q4_K_M", "Q8_0")) == {
        "Q4_K_M": "model-Q4_K_M.gguf",
        "Q8_0": "model-Q8_0.gguf",
    }


def test_run_rejects_an_empty_quantization_list(tmp_path: Path) -> None:
    config = QuantizeConfig(model="m", out_dir=tmp_path, llama_cpp=tmp_path, quants=())
    with pytest.raises(ValueError):
        run(config, runner=lambda command: None)
