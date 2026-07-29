from __future__ import annotations

import compileall
from pathlib import Path


ROOTS = (Path("src"), Path("tests"))
MAX_LINE_LENGTH = 96


def main() -> None:
    if not all(compileall.compile_dir(root, quiet=1) for root in ROOTS):
        raise SystemExit("Python compilation failed")

    violations: list[str] = []
    for root in ROOTS:
        for path in root.rglob("*.py"):
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if len(line) > MAX_LINE_LENGTH:
                    violations.append(
                        f"{path}:{line_number}: {len(line)} characters "
                        f"(maximum {MAX_LINE_LENGTH})"
                    )
    if violations:
        raise SystemExit("\n".join(violations))
    print("Compilation and line-length checks passed")


if __name__ == "__main__":
    main()
