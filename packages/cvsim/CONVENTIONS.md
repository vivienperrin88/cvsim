# Conventions

These choices are fixed once for the whole code base. They are **identical to qttlind's**
(`packages/qttlind/CONVENTIONS.md`, section *Continuous variables*) so that the two Gaussian
engines can be compared number for number (decision D2). Rules marked **[tested]** are implemented
twice, independently, in `cvsim.core.conventions` and in the oracle `cvsim.reference`, and
cross-checked by `tests/test_conventions.py`; when qttlind is installed, the contract tests also
compare them with qttlind.

*The rules are written in session S02.*
