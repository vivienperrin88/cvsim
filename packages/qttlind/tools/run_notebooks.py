"""Execute the tutorial notebooks in place (outputs saved), and report failures and timings.

    uv run python tools/run_notebooks.py                 # all notebooks, quick mode
    uv run python tools/run_notebooks.py 06 08           # selected notebooks (by prefix)
    QTTLIND_PAPER_SIZE=1 uv run python tools/run_notebooks.py --timeout 36000
    uv run python tools/run_notebooks.py --check         # execute, but do not write outputs

Each notebook's ``checks`` dictionary (``name -> (error, tolerance)``), when it defines one, is
verified after execution: the script fails if any error exceeds its tolerance.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

CHECK_CELL = """
try:
    _c = checks
except NameError:
    _c = {}
_bad = {k: v for k, v in _c.items() if not (v[0] <= v[1])}
print("QTTLIND_CHECKS", len(_c), "failed:", _bad)
assert not _bad, f"checks failed: {_bad}"
"""


def run(path: Path, timeout: int, write: bool) -> tuple[bool, float, str]:
    nb = nbformat.read(path, as_version=4)
    probe = nbformat.v4.new_code_cell(CHECK_CELL)
    nb.cells.append(probe)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}})
    t0 = time.perf_counter()
    try:
        client.execute()
        ok, msg = True, ""
    except Exception as exc:
        ok, msg = False, str(exc).splitlines()[-1] if str(exc) else type(exc).__name__
    dt = time.perf_counter() - t0
    summary = ""
    for out in probe.get("outputs", []):
        if out.get("name") == "stdout" and "QTTLIND_CHECKS" in out.get("text", ""):
            summary = out["text"].strip()
    nb.cells.pop()  # never save the probe
    if write and ok:
        nbformat.write(nb, path)
    return ok, dt, summary or msg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefixes", nargs="*", help="notebook name prefixes (default: all)")
    ap.add_argument("--timeout", type=int, default=3600, help="per-cell timeout in seconds")
    ap.add_argument("--check", action="store_true", help="execute without writing outputs")
    args = ap.parse_args()
    paths = sorted(NB_DIR.glob("[0-9][0-9]_*.ipynb"))
    if args.prefixes:
        paths = [p for p in paths if any(p.name.startswith(x) for x in args.prefixes)]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    failed = 0
    for p in paths:
        ok, dt, info = run(p, args.timeout, write=not args.check)
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {dt:8.1f} s  {p.name}  {info}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
