"""The System: named modes, their sizes and bases, and initial states."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..core.mpo import MPO
from ..core.tt import TensorTrain
from ..errors import InputError, ShapeError
from ..lindblad.comb import PurifiedComb
from .operators import CustomBasis, Mode, ModeInfo
from .states import StateSpec, fock

__all__ = ["System", "SystemState"]


class SystemState(PurifiedComb):
    """A :class:`PurifiedComb` that remembers its :class:`System` (operations return plain combs)."""

    __slots__ = ("system",)

    def __init__(self, comb: PurifiedComb, system: System):
        super().__init__(comb.tree, comb.layout)
        self.system = system


class System:
    """An ordered collection of bosonic modes (comb backbone order = insertion order)."""

    def __init__(self) -> None:
        self._modes: list[Mode] = []
        self._index: dict[str, int] = {}

    def add_mode(self, name: str, bits: int, basis: CustomBasis | None = None) -> Mode:
        """Add a mode with ``2**bits`` levels; returns its annihilation operator."""
        if not isinstance(name, str) or not name.isidentifier():
            raise InputError(f"mode names must be identifiers (used as keyword arguments), got {name!r}")
        if name in self._index:
            raise InputError(f"mode {name!r} already exists")
        if not isinstance(bits, (int, np.integer)) or not 1 <= bits <= 30:
            raise InputError(f"bits must be an integer in 1..30, got {bits!r}")
        if basis is not None and basis.dim > 2**bits:
            raise InputError(f"basis of {basis.dim} levels does not fit in 2**{bits}")
        m = Mode(ModeInfo(name, int(bits), basis))
        self._index[name] = len(self._modes)
        self._modes.append(m)
        return m

    # ------------------------------------------------------------- lookup
    @property
    def modes(self) -> tuple[Mode, ...]:
        return tuple(self._modes)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self._modes)

    @property
    def bits(self) -> tuple[int, ...]:
        return tuple(m.bits for m in self._modes)

    def index(self, name: str) -> int:
        if name not in self._index:
            raise InputError(f"unknown mode {name!r}; this system has {list(self._index)}")
        return self._index[name]

    def mode_info(self, name: str) -> ModeInfo:
        return self._modes[self.index(name)].info

    def __getitem__(self, name: str) -> Mode:
        return self._modes[self.index(name)]

    def __len__(self) -> int:
        return len(self._modes)

    def __repr__(self) -> str:
        return "System(" + ", ".join(f"{m.name}: 2^{m.bits}" for m in self._modes) + ")"

    def describe(self) -> dict:
        """JSON-friendly description (stored in result metadata)."""
        return {
            "modes": [{"name": m.name, "bits": m.bits, "custom_basis": m.info.basis is not None} for m in self._modes]
        }

    # ------------------------------------------------------------- states
    def product_state(self, **specs: StateSpec | TensorTrain) -> SystemState:
        """``|psi_0> (x) |psi_1> (x) ...``; unspecified modes start in ``|0>``."""
        if not self._modes:
            raise InputError("the system has no modes")
        unknown = set(specs) - set(self._index)
        if unknown:
            raise InputError(f"unknown mode(s) {sorted(unknown)}; this system has {list(self._index)}")
        legs = []
        for m in self._modes:
            spec = specs.get(m.name, fock(0))
            if isinstance(spec, TensorTrain):
                if spec.dims != (2,) * m.bits:
                    raise ShapeError(f"mode {m.name!r}: tensor train on {spec.dims} does not fit {m.bits} bits")
                leg = spec * (1.0 / spec.norm())
            else:
                leg = spec.build(m.bits)
            legs.append(leg)
        return SystemState(PurifiedComb.product_state(legs), self)

    def state_from_dense(
        self, array: np.ndarray, pad: Mapping[str, int] | None = None, *, tol: float = 1e-20
    ) -> SystemState:
        """From a dense ket of shape ``(d_0, d_1, ...)`` (or a purification ``(d_0, ..., chi_mu)``).

        Each axis is zero-padded to ``2**bits`` levels. ``pad`` documents intended sizes
        (``{"cavity": 1024}``); every padded size must fit the mode. A flat ket of length
        ``prod 2**bits`` is accepted too. The result is normalised.
        """
        A = np.asarray(array, dtype=np.complex128)
        M = len(self._modes)
        full = [2**b for b in self.bits]
        pad = dict(pad or {})
        for name, size in pad.items():
            i = self.index(name)
            if size > full[i]:
                raise ShapeError(f"pad {name}={size} exceeds the mode's 2**{self.bits[i]} levels")
        if A.ndim == 1 and A.size == int(np.prod(full)):
            A = A.reshape(full)
        if A.ndim not in (M, M + 1):
            raise ShapeError(f"expected an array with {M} (ket) or {M + 1} (purification) axes, got shape {A.shape}")
        if A.ndim == M:
            A = A[..., None]
        for i, d in enumerate(A.shape[:M]):
            if d > full[i]:
                raise ShapeError(f"mode {self._modes[i].name!r}: {d} levels do not fit in 2**{self.bits[i]}")
        out = np.zeros((*full, A.shape[-1]), dtype=np.complex128)
        out[tuple(slice(0, d) for d in A.shape[:M]) + (slice(None),)] = A
        nrm = np.linalg.norm(out)
        if nrm == 0:
            raise InputError("the state is zero")
        Psi = (out / nrm).reshape(-1, A.shape[-1])
        return SystemState(PurifiedComb.from_dense(Psi, self.bits, tol=tol).normalize(), self)

    def comb_to_chain(self, state: PurifiedComb | TensorTrain) -> TensorTrain:
        """A pure comb as one tensor train over all bits (modes in order, MSB first)."""
        if isinstance(state, TensorTrain):
            return state
        if state.bits != self.bits:
            raise ShapeError(f"state on bits {state.bits} does not match the system {self.bits}")
        if state.chi_mu != 1:
            raise InputError(f"closed-system evolution needs a pure state (chi_mu = 1), got chi_mu = {state.chi_mu}")
        if state.num_modes == 1 or state.bond_dims()["chi_e"] == 1:
            return _product_comb_to_chain(state)
        if sum(self.bits) <= 16:
            v = state.to_dense()[:, 0]
            return TensorTrain.from_dense(v, [2] * sum(self.bits), tol=1e-28)
        raise InputError("entangled multi-mode combs above 2^16 states cannot be converted to a chain yet")


def _product_comb_to_chain(state: PurifiedComb) -> TensorTrain:
    """Concatenate the legs; the backbone and ``mu`` bonds must have dimension 1."""
    state, _ = state.compress(1e-28)
    tree = state.tree
    lay = state.layout
    cores: list[np.ndarray] = []
    for a, leg in enumerate(lay.legs):
        b = lay.backbone[a]
        tb = tree.tensors[b][..., 0]  # drop the physical axis of dimension 1
        head = np.moveaxis(tb, tree.axis(b, leg[0]), -1)
        if int(np.prod(head.shape[:-1])) != 1:
            raise InputError("the comb is not a product state (a backbone bond is larger than 1)")
        head = head.reshape(1, -1)  # (1, chi_q)
        for j, q in enumerate(leg):
            t = tree.tensors[q]
            core = t[:, :, None] if t.ndim == 2 else t.transpose(0, 2, 1)  # (parent, s, child)
            if j == 0:
                core = np.tensordot(head, core, axes=([1], [0]))
            cores.append(core)
    mu = tree.tensors[lay.mu]
    if mu.size != 1:
        raise InputError("the comb is not a product state (purification bond larger than 1)")
    cores[0] = cores[0] * complex(mu.reshape(-1)[0])
    return TensorTrain(cores)


def chain_mpo(ops: Mapping[int, MPO], bits: Sequence[int]) -> MPO:
    """Tensor product of single-mode MPOs on a chain of all bits (identity on missing modes)."""
    cores: list[np.ndarray] = []
    for k, b in enumerate(bits):
        if k in ops:
            cores += list(ops[k].cores)
        else:
            cores += MPO.identity([2] * b).cores
    return MPO(cores)
