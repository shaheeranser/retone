"""Convert a merged model to GGUF and quantize it, keeping the projector at f16.

The input is the merged f16 Hugging Face tree that `retone.train.finetune` pushes
to `<output-repo>-merged`; merging the LoRA is that entry point's job, so this
module is only the llama.cpp conversion and quantization step. It shells out to a
llama.cpp checkout and never imports the training stack.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from retone.train.env import load_env

DEFAULT_QUANTS = ("Q4_K_M", "Q8_0")
MODEL_F16 = "model-f16.gguf"
MMPROJ_F16 = "mmproj-f16.gguf"

Runner = Callable[[Sequence[str]], None]


@dataclass(frozen=True)
class QuantizeConfig:
    model: str
    out_dir: Path
    llama_cpp: Path
    python: str = sys.executable
    quantize_tool: Path | None = None
    quants: tuple[str, ...] = DEFAULT_QUANTS

    def quantize_binary(self) -> Path:
        return self.quantize_tool or self.llama_cpp / "build" / "bin" / "llama-quantize"


@dataclass(frozen=True)
class QuantizeSummary:
    model_f16: str
    mmproj: str
    quants: dict[str, str]


def convert_command(config: QuantizeConfig, outfile: Path) -> list[str]:
    """Convert the merged HF model to a single f16 GGUF language model."""
    return [
        config.python,
        str(config.llama_cpp / "convert_hf_to_gguf.py"),
        config.model,
        "--outfile",
        str(outfile),
        "--outtype",
        "f16",
    ]


def mmproj_command(config: QuantizeConfig, outfile: Path) -> list[str]:
    """Convert only the vision projector to f16 GGUF with the same converter.

    The `--mmproj` flag exists in recent llama.cpp; the projector stays f16 because
    quantizing it degrades the model's view of the image more than it saves space.
    """
    return [*convert_command(config, outfile), "--mmproj"]


def quantize_command(source: Path, destination: Path, kind: str, *, tool: Path) -> list[str]:
    """llama-quantize `source` into `destination` at the named quantization type."""
    return [str(tool), str(source), str(destination), kind]


def quantization_names(quants: Sequence[str]) -> dict[str, str]:
    return {kind: f"model-{kind}.gguf" for kind in quants}


def run(config: QuantizeConfig, *, runner: Runner | None = None) -> QuantizeSummary:
    """Convert to f16, convert the projector, then quantize each requested type."""
    if not config.quants:
        raise ValueError("at least one quantization type is required")
    execute = runner if runner is not None else _subprocess_runner
    config.out_dir.mkdir(parents=True, exist_ok=True)

    model_f16 = config.out_dir / MODEL_F16
    mmproj = config.out_dir / MMPROJ_F16
    execute(convert_command(config, model_f16))
    execute(mmproj_command(config, mmproj))

    names = quantization_names(config.quants)
    for kind, name in names.items():
        execute(
            quantize_command(model_f16, config.out_dir / name, kind, tool=config.quantize_binary())
        )
    return QuantizeSummary(
        model_f16=str(model_f16),
        mmproj=str(mmproj),
        quants={kind: str(config.out_dir / name) for kind, name in names.items()},
    )


def _subprocess_runner(command: Sequence[str]) -> None:
    subprocess.run(command, check=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert and quantize the merged model to GGUF")
    parser.add_argument("--model", required=True, help="merged model: HF id or local directory")
    parser.add_argument("--out", type=Path, required=True, help="directory for the GGUF files")
    parser.add_argument(
        "--llama-cpp",
        type=Path,
        required=True,
        help="llama.cpp checkout containing convert_hf_to_gguf.py",
    )
    parser.add_argument(
        "--quantize-tool",
        type=Path,
        default=None,
        help="path to llama-quantize (default: <llama-cpp>/build/bin/llama-quantize)",
    )
    parser.add_argument(
        "--quant",
        action="append",
        default=None,
        choices=DEFAULT_QUANTS,
        help="quantization type; repeatable (default: Q4_K_M and Q8_0)",
    )
    parser.add_argument("--python", default=sys.executable, help="interpreter for the converter")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # The converter child process reads HF_TOKEN from the environment, so load a
    # local .env first; a private merged repository needs that token to download.
    load_env()
    args = _parser().parse_args(argv)
    config = QuantizeConfig(
        model=args.model,
        out_dir=args.out,
        llama_cpp=args.llama_cpp,
        python=args.python,
        quantize_tool=args.quantize_tool,
        quants=tuple(args.quant) if args.quant else DEFAULT_QUANTS,
    )
    try:
        summary = run(config)
    except (OSError, subprocess.CalledProcessError) as err:
        print(f"quantize failed: {err}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
