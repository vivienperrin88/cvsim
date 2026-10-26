# API reference

The top-level namespace re-exports the user API (`ql.System`, `ql.LindbladSolver`, …);
everything else lives in the subpackages below, in dependency order (a package imports only from
those above it in this list, plus `errors`).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   qttlind.errors
   qttlind.core
   qttlind.bosons
   qttlind.closed
   qttlind.lindblad
   qttlind.api
   qttlind.gaussian
   qttlind.cv
   qttlind.circuits
   qttlind.reference
```

## Top-level namespace

```{eval-rst}
.. automodule:: qttlind
   :no-members:

.. autofunction:: qttlind.require
```
