# Release progress (temporary — delete before tagging v2.0.0)

Done: library S00–S38 implemented; 781 tests pass; ruff + mypy clean; docs written (guide, design,
API, tutorials); README/CHANGELOG/LICENSE/CITATION/CI; tools/run_notebooks.py,
tools/check_doc_examples.py; notebooks 00–07 executed with outputs (quick mode).

Remaining:
1. `uv run python tools/run_notebooks.py 08 09 10 11 12 13`
2. `QTTLIND_DOCS_OFFLINE=1 uv run sphinx-build -W -b html docs docs/_build/html` (fix warnings)
3. `uv run python tools/check_doc_examples.py`
4. `uv run pytest` again (docstring edits since last full run), `uv build`, test the wheel
5. tag v2.0.0, remove this file
