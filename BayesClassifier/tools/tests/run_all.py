#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = CURRENT_DIR.parent
ROOT_DIR = TOOLS_DIR.parent


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT_DIR, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    python = str((TOOLS_DIR / "venv" / "bin" / "python"))
    tests = [
        [python, str(ROOT_DIR / "tools/tests/test_default_schema.py")],
        [python, str(ROOT_DIR / "tools/tests/test_preprocess_and_analysis.py")],
        [python, str(ROOT_DIR / "tools/tests/test_combine_modes.py")],
        [python, str(ROOT_DIR / "tools/tests/run_wine_pdfs.py")],
    ]
    for cmd in tests:
        run(cmd)
    print("\n✅ All test scripts completed successfully.")


if __name__ == "__main__":
    main()
