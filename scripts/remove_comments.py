from __future__ import annotations

import sys
from pathlib import Path


def remove_comments(source: str) -> str:
    lines = source.splitlines(keepends=True)
    result = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        result.append(line)
    return "".join(result)


def process_file(path: Path) -> None:
    original = path.read_text(encoding="utf-8")
    cleaned = remove_comments(original)
    path.write_text(cleaned, encoding="utf-8")
    print(f"processed: {path}")


if __name__ == "__main__":
    targets = sys.argv[1:]
    if not targets:
        print("usage: python scripts/remove_comments.py <file_or_dir> ...")
        sys.exit(1)

    for target in targets:
        p = Path(target)
        if p.is_file():
            process_file(p)
        elif p.is_dir():
            for py_file in sorted(p.rglob("*.py")):
                process_file(py_file)
        else:
            print(f"skipped (not found): {target}")
