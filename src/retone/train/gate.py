"""Pilot gate: does a trained adapter beat doing nothing on held-out rows?"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from retone.eval import parameter_mae
from retone.executor import Params
from retone.train import qlora


def verdict(model_mae: float, identity_mae: float, margin: float) -> bool:
    """True when the model's mean error is at most `margin` times the identity baseline."""
    return model_mae <= margin * identity_mae


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True, help="checkpoint directory")
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--margin", type=float, default=0.85)
    args = parser.parse_args(argv)

    token = os.environ.get("HF_TOKEN", "")
    settings = qlora.TrainSettings(
        model_name=str(args.adapter), dataset_repo=args.dataset_repo, output_repo=""
    )
    _train, test = qlora.load_examples_from_hub(settings, token)
    rows = [r for r in test if any(r.params.as_dict().values())][: args.rows]
    loaded = qlora.load_model(str(args.adapter), lora=False)
    predicted = qlora.predict(loaded, rows, len(rows))
    expected = [r.params for r in rows]
    model_mae = _mean(parameter_mae(predicted, expected))
    identity_mae = _mean(parameter_mae([Params()] * len(rows), expected))
    passed = verdict(model_mae, identity_mae, args.margin)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "model_mae": model_mae,
                "identity_mae": identity_mae,
                "pass": passed,
            }
        )
    )
    return 0 if passed else 1


def _mean(values: dict[str, float]) -> float:
    return sum(values.values()) / len(values)


if __name__ == "__main__":
    sys.exit(main())
