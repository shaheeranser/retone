# ruff: noqa: E501  (inline HTML strings)
#!/usr/bin/env python3
"""Original vs quantized-model photos in one simple HTML page (camera pairs and phone photos).

Needs a served model for the phone photos:
    MODEL_DIR=~/.retone/model scripts/serve.sh --port 8080 &
    python scripts/side_by_side.py --phone <dir> --out <dir>
Camera photos reuse the predictions already written by scripts/quant_compare.py.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from retone.eval.human import load_human_pairs
from retone.executor import Params, apply, load_image
from retone.inference import LlamaServerClient, thumbnail, validate_model_output

_STYLE = """body{font-family:system-ui,sans-serif;margin:2rem;background:#111;color:#eee}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #444;padding:.5rem;vertical-align:top}
img{max-width:320px;height:auto;display:block}figure{margin:0}
figcaption{font-size:.85rem;color:#aaa;margin-top:.3rem}h2{margin-top:2.5rem}"""


def _small(source: Path | NDArray[np.integer], name: str, out: Path) -> str:
    image = Image.open(source) if isinstance(source, Path) else Image.fromarray(source)
    image = image.convert("RGB")
    image.thumbnail((720, 720))
    image.save(out / "img" / name, quality=82)
    return f"img/{html.escape(name)}"


def _fmt(params: Params) -> str:
    changes = [f"{k} {v:+d}" for k, v in params.as_dict().items() if v]
    return ", ".join(changes) or "no change"


def _figure(src: str, caption: str) -> str:
    return f'<td><figure><img src="{src}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure></td>'


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phone", type=Path, required=True)
    parser.add_argument(
        "--camera-pairs", type=Path, default=Path("artifacts/data/human/pairs.jsonl")
    )
    parser.add_argument("--quant-dir", type=Path, default=Path("artifacts/eval/quant/2026-11-28"))
    parser.add_argument("--label", default="Q4_K_M")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    (args.out / "img").mkdir(parents=True, exist_ok=True)

    client = LlamaServerClient(base_url=args.server_url, timeout=600.0)
    rows: list[str] = []
    for photo in sorted(
        p for p in args.phone.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ):
        image = load_image(photo)
        params = validate_model_output(client.choose_params(thumbnail(image))).params
        stem = re.sub(r"\W+", "_", photo.stem)
        original = _small(photo, f"phone-{stem}-orig.jpg", args.out)
        result = _small(apply(image, params), f"phone-{stem}-model.jpg", args.out)
        rows.append(
            f"<tr><td>{html.escape(photo.name)}</td>{_figure(original, 'Original')}"
            f"{_figure(result, f'{args.label}: ' + _fmt(params))}</tr>"
        )
        print("phone", photo.name, _fmt(params), flush=True)

    summary = json.loads((args.quant_dir / f"{args.label}.json").read_text())
    camera: list[str] = []
    for pair in load_human_pairs(args.camera_pairs):
        pid = pair.pair_id
        params = Params(**summary["params"][pid])
        camera.append(
            f"<tr><td>{pid}<br><small>ΔE {summary['per_pair_delta_e'][pid]:.2f}</small></td>"
            + _figure(_small(pair.before, f"cam-{pid}-orig.jpg", args.out), "Original")
            + _figure(_small(pair.after, f"cam-{pid}-human.jpg", args.out), "Your edit")
            + _figure(
                _small(
                    args.quant_dir / f"render-{args.label}" / f"{pid}.jpg",
                    f"cam-{pid}-model.jpg",
                    args.out,
                ),
                f"{args.label}: {_fmt(params)}",
            )
            + "</tr>"
        )

    page = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>retone side by side</title>'
        f"<style>{_STYLE}</style></head><body><h1>retone side by side</h1>"
        f"<p>Original against the quantized model ({args.label}). Camera photos also show your edit and "
        f"ΔE to it; phone photos have no reference.</p>"
        f"<h2>Phone photos</h2><table><thead><tr><th>Photo</th><th>Original</th><th>Model</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        f"<h2>Camera photos</h2><table><thead><tr><th>Pair</th><th>Original</th><th>Your edit</th><th>Model</th></tr></thead>"
        f"<tbody>{''.join(camera)}</tbody></table></body></html>"
    )
    (args.out / "index.html").write_text(page)
    print("wrote", args.out / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
