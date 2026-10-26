"""Operator algebra: write Hamiltonians as ``g * a.dag()**2 * b + eps * (a + a.dag())``.

An :class:`Operator` is a finite sum ``sum_terms coeff(t) x prod_modes factors``. On each mode the
factors are a product of *atoms*:

* ladder words ``a^dag^p a^q`` kept in **normal order**. The product of two words follows from
  ``[a, a^dag] = 1`` (Wick):
  ``(a†^p a^q)(a†^r a^s) = sum_k C(q,k) C(r,k) k! a†^(p+r-k) a^(q+s-k)``;
* opaque factors (``parity()``, the energies ``H0()`` and named operators of a
  :class:`CustomBasis`), which do not commute with anything and are kept in order.

**Physics choice.** On a truncated Fock space ``[a, a^dag] = 1`` fails on the last level. Normal
ordering builds ``a a^dag`` as ``a^dag a + 1`` (exact in the infinite space), not as the product of
truncated matrices: the two differ only at ``n = N - 1``.

Coefficients are a constant times a product of functions of time (``coeff(f) * op``); terms with
the same atoms *and the same functions* (by identity) are merged. Operators are immutable and
hashable. Lower layers never see this module: :meth:`Operator.to_local_terms` lowers an operator to
the :class:`~qttlind.lindblad.kraus.LocalTerm` lists of the Kraus integrator, with cached MPOs.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..bosons import operators as bops
from ..core.mpo import MPO
from ..core.tt import TensorTrain
from ..errors import InputError

if TYPE_CHECKING:
    from ..lindblad.kraus import LocalTerm
    from .system import System

__all__ = ["CustomBasis", "Mode", "Operator", "clear_mpo_cache", "coeff"]

#: Discarded weight when building and multiplying single-mode MPOs.
MPO_TOL = 1e-24

Number = int | float | complex | np.number


# ---------------------------------------------------------------------------------- atoms
@dataclass(frozen=True)
class Word:
    """``a†^p a^q`` (normal ordered)."""

    p: int
    q: int

    def dagger(self) -> Word:
        return Word(self.q, self.p)

    def __str__(self) -> str:
        def pw(sym: str, k: int) -> str:
            return "" if k == 0 else sym if k == 1 else f"{sym}^{k}"

        return pw("a†", self.p) + pw("a", self.q) or "1"


@dataclass(frozen=True)
class Opaque:
    """A non-polynomial factor: ``kind`` in {parity, H0, op}; ``name``/``offset`` for basis operators."""

    kind: str
    name: str = ""
    offset: int | None = None
    dag: bool = False

    def dagger(self) -> Opaque:
        if self.kind in ("parity", "H0"):  # Hermitian
            return self
        return Opaque(self.kind, self.name, self.offset, not self.dag)

    def __str__(self) -> str:
        if self.kind != "op":
            return self.kind
        off = "" if self.offset is None else f"[{self.offset:+d}]"
        return f"{self.name}{off}{'†' if self.dag else ''}"


Atom = Word | Opaque
Factors = tuple[tuple[str, tuple[Atom, ...]], ...]  # sorted by mode name
Key = tuple[Factors, tuple[Callable[[float], complex], ...]]


def _wick(x: Word, y: Word) -> list[tuple[float, Word]]:
    """``x * y`` in normal order."""
    out = []
    for k in range(min(x.q, y.p) + 1):
        c = math.comb(x.q, k) * math.comb(y.p, k) * math.factorial(k)
        out.append((float(c), Word(x.p + y.p - k, x.q + y.q - k)))
    return out


def _normalise(atoms: tuple[Atom, ...]) -> list[tuple[float, tuple[Atom, ...]]]:
    """Merge adjacent words by Wick's rule; drop identity words. A sum of atom sequences."""
    results: list[tuple[float, tuple[Atom, ...]]] = [(1.0, ())]
    for atom in atoms:
        nxt: list[tuple[float, tuple[Atom, ...]]] = []
        for c, seq in results:
            if isinstance(atom, Word) and seq and isinstance(seq[-1], Word):
                for c2, w in _wick(seq[-1], atom):
                    nxt.append((c * c2, seq[:-1] + ((w,) if (w.p, w.q) != (0, 0) else ())))
            elif isinstance(atom, Word) and (atom.p, atom.q) == (0, 0):
                nxt.append((c, seq))
            else:
                nxt.append((c, (*seq, atom)))
        results = nxt
    return results


# ---------------------------------------------------------------------------------- coefficients
_CONJ: dict[int, Callable[[float], complex]] = {}


def _conj_fn(f: Callable[[float], complex]) -> Callable[[float], complex]:
    """The complex conjugate of ``f``, the same object every time (so conjugate terms still merge)."""
    g = _CONJ.get(id(f))
    if g is None:

        def g(t: float, f=f) -> complex:
            return complex(np.conj(f(t)))

        g._qttlind_conj_of = f  # type: ignore[attr-defined]
        _CONJ[id(f)] = g
        _CONJ[id(g)] = f
    return g


def _funcs_key(funcs: Iterable[Callable[[float], complex]]) -> tuple[Callable[[float], complex], ...]:
    return tuple(sorted(funcs, key=id))


# ---------------------------------------------------------------------------------- operator
class Operator:
    """An immutable sum of products of single-mode factors with (possibly time-dependent) coefficients."""

    __slots__ = ("_terms",)
    __array_ufunc__ = None  # numpy scalars defer to Operator.__rmul__ / __radd__
    _terms: dict[Key, complex]

    def __init__(self, terms: Mapping[Key, complex] | None = None):
        clean: dict[Key, complex] = {}
        for (factors, funcs), c in (terms or {}).items():
            c = complex(c)
            if c != 0:
                key = (factors, _funcs_key(funcs))
                clean[key] = clean.get(key, 0) + c
        object.__setattr__(self, "_terms", {k: v for k, v in clean.items() if v != 0})

    # ------------------------------------------------------------- construction helpers
    @classmethod
    def scalar(cls, value: Number) -> Operator:
        return cls({((), ()): complex(value)})

    @classmethod
    def _single(cls, mode: str, atom: Atom, c: complex = 1.0) -> Operator:
        return cls({(((mode, (atom,)),), ()): c})

    @staticmethod
    def _coerce(x: Any) -> Operator | None:
        if isinstance(x, Operator):
            return x
        if isinstance(x, (int, float, complex, np.number)):
            return Operator.scalar(x)
        return None

    # ------------------------------------------------------------- algebra
    def __add__(self, other: Any) -> Operator:
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        terms = dict(self._terms)
        for k, v in o._terms.items():
            terms[k] = terms.get(k, 0) + v
        return Operator(terms)

    __radd__ = __add__

    def __neg__(self) -> Operator:
        return Operator({k: -v for k, v in self._terms.items()})

    def __sub__(self, other: Any) -> Operator:
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        return self + (-o)

    def __rsub__(self, other: Any) -> Operator:
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        return o + (-self)

    def __mul__(self, other: Any) -> Operator:
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        terms: dict[Key, complex] = {}
        for (fx, gx), cx in self._terms.items():
            for (fy, gy), cy in o._terms.items():
                funcs = _funcs_key(gx + gy)
                modes = sorted({m for m, _ in fx} | {m for m, _ in fy})
                dx, dy = dict(fx), dict(fy)
                partial: list[tuple[complex, list[tuple[str, tuple[Atom, ...]]]]] = [(cx * cy, [])]
                for m in modes:
                    seqs = _normalise(dx.get(m, ()) + dy.get(m, ()))
                    partial = [(c * cs, [*fac, (m, seq)] if seq else fac) for c, fac in partial for cs, seq in seqs]
                for c, fac in partial:
                    key = (tuple(fac), funcs)
                    terms[key] = terms.get(key, 0) + c
        return Operator(terms)

    def __rmul__(self, other: Any) -> Operator:
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        return o * self

    def __truediv__(self, other: Any) -> Operator:
        if isinstance(other, (int, float, complex, np.number)):
            return self * (1.0 / other)
        return NotImplemented

    def __pow__(self, k: int) -> Operator:
        if not isinstance(k, (int, np.integer)) or k < 0:
            raise InputError(f"operator powers must be non-negative integers, got {k!r}")
        out = Operator.scalar(1.0)
        for _ in range(int(k)):
            out = out * self
        return out

    def dag(self) -> Operator:
        """Hermitian conjugate."""
        terms = {}
        for (factors, funcs), c in self._terms.items():
            fac = tuple((m, tuple(a.dagger() for a in reversed(seq))) for m, seq in factors)
            terms[(fac, tuple(_conj_fn(f) for f in funcs))] = np.conj(c)
        # a reversed sequence of adjacent words is no longer normal ordered: renormalise via products
        out = Operator()
        for (fac, funcs), c in terms.items():
            prod = Operator({((), funcs): c})
            for m, seq in fac:
                for a in seq:
                    prod = prod * Operator._single(m, a)
            out = out + prod
        return out

    dagger = dag

    # ------------------------------------------------------------- inspection
    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, float, complex, np.number)):
            other = Operator.scalar(other)
        if not isinstance(other, Operator):
            return NotImplemented
        return self._terms == other._terms

    def __hash__(self) -> int:
        return hash(frozenset(self._terms.items()))

    def __bool__(self) -> bool:
        return bool(self._terms)

    @property
    def terms(self) -> dict[Key, complex]:
        return dict(self._terms)

    @property
    def is_time_dependent(self) -> bool:
        return any(funcs for (_, funcs) in self._terms)

    @property
    def modes(self) -> set[str]:
        return {m for (fac, _) in self._terms for m, _ in fac}

    def __repr__(self) -> str:
        if not self._terms:
            return "0"
        parts = []
        for (fac, funcs), c in self._terms.items():
            body = "·".join(f"{''.join(str(a) for a in seq)}({m})" for m, seq in fac) or "1"
            f = "".join(f"·{getattr(g, '__name__', 'f')}(t)" for g in funcs)
            parts.append(f"{_fmt(c)}{f}·{body}")
        return " + ".join(parts)

    # ------------------------------------------------------------- lowering
    def coefficient(self, key: Key) -> complex | Callable[[float], complex]:
        c = self._terms[key]
        funcs = key[1]
        if not funcs:
            return c

        def value(t: float, c=c, funcs=funcs) -> complex:
            out = c
            for f in funcs:
                out *= complex(f(t))
            return out

        return value

    def to_local_terms(self, system: System, *, tol: float = MPO_TOL) -> list[LocalTerm]:
        """Lower to :class:`~qttlind.lindblad.kraus.LocalTerm` (single-mode MPOs keyed by mode index)."""
        from ..lindblad.kraus import LocalTerm

        out = []
        for key in self._terms:
            factors = key[0]
            ops: dict[int, MPO] = {}
            for m, seq in factors:
                idx = system.index(m)
                ops[idx] = _sequence_mpo(system.mode_info(m), seq, tol)
            out.append(LocalTerm(self.coefficient(key), ops))
        return out

    def mode_mpos(self, system: System, *, tol: float = MPO_TOL) -> list[tuple[complex | Callable, dict[int, MPO]]]:
        return [(t.coeff, dict(t.ops)) for t in self.to_local_terms(system, tol=tol)]

    def to_dense(self, system: System, t: float = 0.0) -> np.ndarray:
        """Dense matrix on the full space (modes in system order, Kronecker product); small systems only."""
        dims = [2**b for b in system.bits]
        D = int(np.prod(dims))
        if D > 4096:
            raise InputError(f"to_dense is for small systems: the space has {D} > 4096 states")
        out = np.zeros((D, D), dtype=np.complex128)
        for term in self.to_local_terms(system):
            mats = [term.ops[k].to_dense() if k in term.ops else np.eye(dims[k]) for k in range(len(dims))]
            M = mats[0]
            for x in mats[1:]:
                M = np.kron(M, x)
            out += term.value(t) * M
        return out


def _fmt(c: complex) -> str:
    if c.imag == 0:
        return f"{c.real:.4g}"
    if c.real == 0:
        return f"{c.imag:.4g}j"
    return f"({c.real:.4g}{c.imag:+.4g}j)"


def coeff(f: Callable[[float], complex]) -> Operator:
    """A time-dependent scalar: ``coeff(f) * op`` has coefficient ``f(t)``."""
    if not callable(f):
        raise InputError("coeff expects a callable f(t); multiply by numbers directly")
    return Operator({((), (f,)): 1.0})


# ---------------------------------------------------------------------------------- custom bases
class CustomBasis:
    """A mode described in its own eigenbasis (a transmon, say) instead of Fock states.

    ``energies[k]`` is the energy of level ``k``; ``ops`` are named dense operators in that basis.
    Levels beyond ``len(energies)`` (up to ``2**bits``) are padded with zeros.
    """

    def __init__(self, energies: np.ndarray, ops: Mapping[str, np.ndarray] | None = None):
        self.energies = np.asarray(energies, dtype=float)
        if self.energies.ndim != 1 or self.energies.size == 0:
            raise InputError("energies must be a non-empty 1D array")
        n = self.energies.size
        self.ops: dict[str, np.ndarray] = {}
        for name, m in (ops or {}).items():
            m = np.asarray(m, dtype=np.complex128)
            if m.shape != (n, n):
                raise InputError(f"operator {name!r} has shape {m.shape}, expected ({n}, {n})")
            self.ops[name] = m

    @property
    def dim(self) -> int:
        return self.energies.size

    def offsets(self, name: str, atol: float = 0.0) -> list[int]:
        """Diagonal offsets ``d = k - l`` on which ``ops[name][k, l]`` has non-zero entries."""
        m = self._op(name)
        n = m.shape[0]
        return [d for d in range(-(n - 1), n) if np.abs(np.diagonal(m, -d)).max() > atol]

    def _op(self, name: str) -> np.ndarray:
        if name not in self.ops:
            raise InputError(f"unknown basis operator {name!r}; known: {sorted(self.ops)}")
        return self.ops[name]

    def matrix(self, atom: Opaque, N: int) -> np.ndarray:
        out = np.zeros((N, N), dtype=np.complex128)
        if atom.kind == "H0":
            out[np.arange(self.dim), np.arange(self.dim)] = self.energies
            return out
        m = self._op(atom.name)
        if atom.offset is not None:
            d = atom.offset
            m = np.tril(np.triu(m, -d), -d)  # keep the diagonal k - l = d
        out[: self.dim, : self.dim] = m
        return out.conj().T if atom.dag else out


@dataclass(frozen=True)
class ModeInfo:
    name: str
    bits: int
    basis: CustomBasis | None


# ---------------------------------------------------------------------------------- mode
class Mode(Operator):
    """The annihilation operator of a mode, plus constructors for the mode's other operators."""

    __slots__ = ("info",)
    info: ModeInfo

    def __init__(self, info: ModeInfo):
        super().__init__({(((info.name, (Word(0, 1),)),), ()): 1.0})
        object.__setattr__(self, "info", info)

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def bits(self) -> int:
        return self.info.bits

    @property
    def dim(self) -> int:
        return 2**self.info.bits

    def a(self) -> Operator:
        return Operator(self._terms)

    def n(self) -> Operator:
        """Number operator ``a^dag a`` (the level index for a custom basis)."""
        return Operator._single(self.name, Word(1, 1))

    level = n

    def parity(self) -> Operator:
        """``(-1)^n``."""
        return Operator._single(self.name, Opaque("parity"))

    def H0(self) -> Operator:
        """``sum_k E_k |k><k|`` of a custom basis."""
        self._need_basis()
        return Operator._single(self.name, Opaque("H0"))

    def op(self, name: str, offset: int | None = None) -> Operator:
        """A named operator of the custom basis (``offset``: only its diagonal ``k - l = offset``)."""
        basis = self._need_basis()
        basis._op(name)
        return Operator._single(self.name, Opaque("op", name, None if offset is None else int(offset)))

    def op_offsets(self, name: str) -> list[int]:
        return self._need_basis().offsets(name)

    def _need_basis(self) -> CustomBasis:
        if self.info.basis is None:
            raise InputError(f"mode {self.name!r} has no custom basis")
        return self.info.basis

    def __repr__(self) -> str:
        return f"Mode({self.name!r}, bits={self.bits})"

    __hash__ = Operator.__hash__


# ---------------------------------------------------------------------------------- MPO cache
@functools.lru_cache(maxsize=512)
def _word_mpo(bits: int, p: int, q: int, tol: float) -> MPO:
    if p == q:
        return bops.normal_power(p, bits)
    if q == 0:
        return bops.annihilation_power(p, bits).dagger()
    if p == 0:
        return bops.annihilation_power(q, bits)
    return (bops.annihilation_power(p, bits).dagger() @ bops.annihilation_power(q, bits)).compress(tol)[0]


@functools.lru_cache(maxsize=256)
def _parity_mpo(bits: int) -> MPO:
    return bops.parity(bits)


_BASIS_CACHE: dict[tuple[int, Opaque, int, float], MPO] = {}


def _opaque_mpo(info: ModeInfo, atom: Opaque, tol: float) -> MPO:
    if atom.kind == "parity":
        return _parity_mpo(info.bits)
    assert info.basis is not None
    key = (id(info.basis), atom, info.bits, tol)
    if key not in _BASIS_CACHE:
        N = 2**info.bits
        if atom.kind == "H0":
            E = np.zeros(N)
            E[: info.basis.dim] = info.basis.energies
            m = bops.diag(TensorTrain.from_dense(E.astype(np.complex128), [2] * info.bits, tol=tol))
        else:
            m = MPO.from_dense(info.basis.matrix(atom, N), [2] * info.bits, tol=tol)
        _BASIS_CACHE[key] = m
    return _BASIS_CACHE[key]


def _sequence_mpo(info: ModeInfo, seq: tuple[Atom, ...], tol: float) -> MPO:
    mpo: MPO | None = None
    for atom in reversed(seq):  # rightmost acts first
        m = _word_mpo(info.bits, atom.p, atom.q, tol) if isinstance(atom, Word) else _opaque_mpo(info, atom, tol)
        mpo = m if mpo is None else (m @ mpo).compress(tol)[0]
    assert mpo is not None
    return mpo


def clear_mpo_cache() -> None:
    """Forget every cached single-mode MPO (words, parity, custom-basis operators)."""
    _word_mpo.cache_clear()
    _parity_mpo.cache_clear()
    _BASIS_CACHE.clear()
