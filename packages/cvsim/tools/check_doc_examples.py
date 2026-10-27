"""Run the self-contained Python examples of README.md and docs/ (quick start, index).

    uv run python tools/check_doc_examples.py

Code blocks are executed in order, one namespace per file, so later blocks may use earlier names.
Blocks marked with the comment ``# not executed`` on their first line are skipped.
"""

from __future__ import annotations

import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["README.md", "docs/index.md", "docs/quickstart.md"]
BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def main() -> int:
    failed = 0
    for name in FILES:
        text = (ROOT / name).read_text()
        ns: dict = {"__name__": "__doc_example__"}
        for k, code in enumerate(BLOCK.findall(text)):
            if code.lstrip().startswith("# not executed"):
                continue
            t0 = time.perf_counter()
            try:
                exec(compile(code, f"{name}[{k}]", "exec"), ns)
                print(f"ok   {time.perf_counter() - t0:6.1f} s  {name} block {k}")
            except Exception:
                failed += 1
                print(f"FAIL {name} block {k}")
                traceback.print_exc()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
