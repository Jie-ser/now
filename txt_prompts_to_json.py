"""Convert an image-name/prompt TXT file into run_bon_batch.py prompt JSON.

Accepted line formats:
    test0051.png + Prompt text
    test_real0183: Prompt text

Image extensions are removed from the JSON key, so the result matches the
stem-based lookup used by ``run_bon_batch.py``. Blank lines are ignored.
"""

import argparse
import json
import re
from pathlib import Path


LINE = re.compile(
    r"^\s*(?P<name>(?:test|test_real)\d{4})(?:\.(?:png|jpe?g))?\s*"
    # Some source TXT files use an em dash (and may have been saved with a
    # legacy code page), so accept any run of non-word separators as well.
    r"(?:\+|:|[^\w\s]+)\s*(?P<prompt>.+?)\s*$",
    re.IGNORECASE,
)


def parse_prompts(path: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        match = LINE.match(line)
        if not match:
            raise ValueError(f"{path}:{line_number}: unsupported prompt line: {line!r}")
        name = match.group("name")
        if name in prompts:
            raise ValueError(f"{path}:{line_number}: duplicate image key {name!r}")
        prompts[name] = match.group("prompt")
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Prompt TXT file")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file")
    args = parser.parse_args()

    prompts = parse_prompts(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prompts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(prompts)} prompts to {args.output}")


if __name__ == "__main__":
    main()
