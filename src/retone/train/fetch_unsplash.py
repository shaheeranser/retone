"""Fetch a small Unsplash subset and write the source manifest the builder consumes."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path

USER_AGENT = "retone-dataset/0.1 (hackathon project)"


def _entries(urls_path: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for line in urls_path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        missing = [key for key in ("photo_id", "url") if not raw.get(key)]
        if missing:
            raise ValueError(f"entry missing {missing}: {line}")
        entries.append(
            {
                "photo_id": str(raw["photo_id"]),
                "url": str(raw["url"]),
                "photographer": str(raw.get("photographer", "unknown")),
                "license": str(raw.get("license", "Unsplash License")),
            }
        )
    return entries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Unsplash photos and write a manifest")
    parser.add_argument("--urls", type=Path, required=True, help="JSONL with photo_id and url")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"
    written = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for entry in _entries(args.urls):
            if args.limit is not None and written >= args.limit:
                break
            target = args.out / f"{entry['photo_id']}.jpg"
            request = urllib.request.Request(entry["url"], headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    target.write_bytes(response.read())
            except OSError as err:
                print(f"skip {entry['photo_id']}: {err}", file=sys.stderr)
                continue
            manifest.write(
                json.dumps(
                    {
                        "photo_id": entry["photo_id"],
                        "file": target.name,
                        "photographer": entry["photographer"],
                        "url": entry["url"],
                        "license": entry["license"],
                    }
                )
                + "\n"
            )
            written += 1
    print(f"wrote {written} photos and {manifest_path}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
