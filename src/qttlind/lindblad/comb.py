"""The purified comb: ``rho = Psi Psi^dagger`` stored as a tree tensor network (paper Sec. IV).

Layout (:class:`~qttlind.core.tree.CombLayout`): a backbone node per mode, the quantics leg of each
mode (MSB attached to the backbone), and one purification leaf ``mu`` attached to the last
backbone node. Bond kinds: ``backbone`` (``chi_e``), ``leg`` (``chi_q``), ``purification``;
the dimension of the ``mu`` index itself is ``chi_mu``.

* Single-mode operators act on their leg only (the MPO's outer bonds have dimension 1): no
  backbone or ``mu`` bond ever changes (decision D9).
* **Vector addition** ``Psi_1 + Psi_2``: block-diagonal at every node, ``mu`` index shared.
  **Matrix addition** ``rho_1 + rho_2``: the ``mu`` index is block-diagonal too, ``Psi = [Psi_1 | Psi_2]``.
* Truncating ``mu``: centre on the ``mu`` node, SVD ``P = U S V^dag``, keep ``U S`` (dropping
  ``V^dag`` is a gauge on ``mu``): the discarded weight is the trace-norm error on ``rho``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..core import linalg
from ..core.mpo import MPO
from ..core.tree import CombLayout, TreeTN, comb_product_state
from ..core.tt import TensorTrain
from ..errors import InputError, ShapeError

__all__ = ["PurifiedComb"]


class PurifiedComb:
    """A purified density matrix on a comb tensor network."""

    __slots__ = ("layout", "tree")
    tree: TreeTN
    layout: CombLayout

    def __init__(self, tree: TreeTN, layout: CombLayout):
        self.tree = tree
        self.layout = layout

    # ------------------------------------------------------------------ constructors
    @classmethod
    def product_state(cls, legs: Sequence[TensorTrain]) -> PurifiedComb:
        """``|psi_0> (x) |psi_1> (x) ...`` (pure: ``chi_mu = 1``) from single-mode quantics tensor trains."""
        tree, layout = comb_product_state(legs)
        return cls(tree, layout)

    @classmethod
    def from_dense(cls, Psi: np.ndarray, bits: Sequence[int], tol: float = 0.0) -> PurifiedComb:
        """From a dense purification ``Psi`` of shape ``(D, chi_mu)`` (or a ket of length ``D``).

        Modes in Kronecker order (mode 0 slowest), ``2**bits[k]`` levels each.
        """
        Psi = np.asarray(Psi, dtype=np.complex128)
        if Psi.ndim == 1:
            Psi = Psi[:, None]
        dims = [2**b for b in bits]
        if Psi.shape[0] != int(np.prod(dims)):
            raise ShapeError(f"Psi has {Psi.shape[0]} rows but the modes have {int(np.prod(dims))} states")
        layout, edges, kinds = CombLayout.build(list(bits))
        M = len(bits)
        coarse = TensorTrain.from_dense(Psi.reshape(-1), dims + [Psi.shape[1]], tol=tol)
        tensors: dict[int, np.ndarray] = {}
        for a in range(M):
            C = coarse.cores[a]  # (l, N, r)
            l_, N, r_ = C.shape
            mat = C.transpose(0, 2, 1).reshape(l_ * r_, N)
            leg = TensorTrain.from_dense(mat.reshape(-1), [l_ * r_] + [2] * bits[a], tol=tol)
            head = leg.cores[0].reshape(l_, r_, -1)  # (l, r, k): bonds to b_{a-1}, to b_{a+1} or mu, to q_1
            if a == 0:
                head = head[0]  # no left neighbour
            tensors[layout.backbone[a]] = head[..., None]  # physical axis of dimension 1
            for j, (c, q) in enumerate(zip(leg.cores[1:], layout.legs[a])):
                tensors[q] = c[:, :, 0] if j == bits[a] - 1 else c.transpose(0, 2, 1)
        tensors[layout.mu] = coarse.cores[M][:, :, 0]  # (l, chi_mu)
        return cls(TreeTN(tensors, edges, kinds), layout)

    # ------------------------------------------------------------------ serialisation
    def to_arrays(self) -> dict[str, np.ndarray]:
        """Flat ``{name: array}`` for ``np.savez`` (bits, centre and one array per node)."""
        out = {
            "comb_bits": np.array(self.bits, dtype=np.int64),
            "comb_center": np.array(-1 if self.tree.center is None else self.tree.center, dtype=np.int64),
        }
        for v, t in self.tree.tensors.items():
            out[f"comb_node_{v}"] = t
        return out

    @classmethod
    def from_arrays(cls, data: Mapping[str, np.ndarray]) -> PurifiedComb:
        """Inverse of :meth:`to_arrays` (bit-for-bit)."""
        bits = [int(b) for b in data["comb_bits"]]
        layout, edges, kinds = CombLayout.build(bits)
        nodes = [layout.mu, *layout.backbone, *(q for leg in layout.legs for q in leg)]
        tensors = {v: np.asarray(data[f"comb_node_{v}"]) for v in nodes}
        c = int(data["comb_center"])
        return cls(TreeTN(tensors, edges, kinds, None if c < 0 else c), layout)

    def tensor(self, other: PurifiedComb) -> PurifiedComb:
        """``rho_1 (x) rho_2``: the modes of ``other`` appended after those of ``self``.

        ``Psi = Psi_1 (x) Psi_2`` with ``mu = (mu_1, mu_2)``: the purification bond of ``self``
        is carried along the backbone of ``other`` (a delta), then both meet in the new ``mu``.
        """
        A, B = self.tree.canonicalize(self.layout.mu), other.tree
        la, lb = self.layout, other.layout
        M1, M2 = la.num_modes, lb.num_modes
        layout, edges, kinds = CombLayout.build([*la.bits, *lb.bits])
        e1 = A.bond_dim(la.backbone[-1], la.mu)
        tensors: dict[int, np.ndarray] = {}
        # modes of self: same tensors; the last backbone node now points to b'_0 instead of mu
        for a in range(M1):
            tensors[layout.backbone[a]] = A.tensors[la.backbone[a]]
            for q_old, q_new in zip(la.legs[a], layout.legs[a]):
                tensors[q_new] = A.tensors[q_old]
        I1 = np.eye(e1, dtype=np.complex128)
        for j in range(M2):
            t = B.tensors[lb.backbone[j]]  # axes: [left], right-or-mu, leg, phys(1)
            has_left = j > 0
            if not has_left:
                t = t[None]  # a left bond of dimension 1
            # carry the e1 index: left (l) -> (e, l), right (r) -> (e, r)
            T = np.einsum("lrqs,ef->elfrqs", t, I1)
            l_, r_ = t.shape[0], t.shape[1]
            T = T.reshape(e1 * l_, e1 * r_, t.shape[2], t.shape[3])
            if not has_left:  # left bond (e, 1) connects to the purification bond of self: dim e1
                T = T.reshape(e1, e1 * r_, t.shape[2], t.shape[3])
            tensors[layout.backbone[M1 + j]] = T
            for q_old, q_new in zip(lb.legs[j], layout.legs[M1 + j]):
                tensors[q_new] = B.tensors[q_old]
        mu1 = A.tensors[la.mu]  # (e1, chi1)
        mu2 = B.tensors[lb.mu]  # (e2, chi2)
        tensors[layout.mu] = np.kron(mu1, mu2)  # ((e1, e2), (chi1, chi2))
        return PurifiedComb(TreeTN(tensors, edges, kinds), layout)

    # ------------------------------------------------------------------ basic properties
    @property
    def num_modes(self) -> int:
        return self.layout.num_modes

    @property
    def bits(self) -> tuple[int, ...]:
        return self.layout.bits

    @property
    def chi_mu(self) -> int:
        return self.tree.phys_dim(self.layout.mu)

    def bond_dims(self) -> dict[str, int]:
        """Largest bond per kind: ``chi_e`` (backbone), ``chi_q`` (legs), ``chi_mu`` (purification index)."""
        return {
            "chi_e": self.tree.max_bond("backbone") if self.num_modes > 1 else 1,
            "chi_q": self.tree.max_bond("leg"),
            "chi_mu": self.chi_mu,
        }

    def __repr__(self) -> str:
        return f"PurifiedComb(bits={self.bits}, {self.bond_dims()})"

    def copy(self) -> PurifiedComb:
        return PurifiedComb(self.tree.copy(), self.layout)

    def to_dense(self) -> np.ndarray:
        """``Psi`` of shape ``(D, chi_mu)`` (modes in Kronecker order, mode 0 slowest)."""
        v = self.tree.to_dense(self.layout.phys_order())
        return v.reshape(-1, self.chi_mu)

    # ------------------------------------------------------------------ local operators
    def apply_local(
        self,
        ops: Mapping[int, MPO],
        *,
        tol: float | None = 0.0,
        caps: Mapping[str, int] | None = None,
    ) -> PurifiedComb:
        """Apply single-mode MPOs (``{mode: MPO}``) leg by leg; compress the changed legs.

        ``tol=None`` skips the compression (exact; leg bonds multiply by the MPO ranks).
        """
        tensors = dict(self.tree.tensors)
        changed: list[tuple[int, int]] = []
        for mode, op in ops.items():
            if not 0 <= mode < self.num_modes:
                raise InputError(f"mode {mode} out of range 0..{self.num_modes - 1}")
            leg = self.layout.legs[mode]
            if op.length != len(leg) or any(d != 2 for d in op.dims_in + op.dims_out):
                raise ShapeError(f"mode {mode}: MPO on {op.length} bits does not fit a leg of {len(leg)} bits")
            for j, (q, W) in enumerate(zip(leg, op.cores)):
                t = tensors[q]
                lw, o, _, rw = W.shape
                if t.ndim == 3:  # (parent, child, s)
                    T = linalg.contract(t, [2], W, [2])  # (p, c, lw, o, rw)
                    p, c = t.shape[:2]
                    tensors[q] = T.transpose(0, 2, 1, 4, 3).reshape(p * lw, c * rw, o)
                else:  # last bit: (parent, s)
                    T = linalg.contract(t, [1], W, [2])  # (p, lw, o, rw=1)
                    tensors[q] = T.reshape(t.shape[0] * lw, o)
                del j
            chain = [self.layout.backbone[mode], *leg]
            changed += list(zip(chain[:-1], chain[1:]))
        tree = self.tree._with(tensors, None)
        out = PurifiedComb(tree, self.layout)
        if tol is None or not changed:
            return out
        tree2, _ = tree.compress(tol, caps, only=changed)
        return PurifiedComb(tree2, self.layout)

    def scale(self, factor: complex) -> PurifiedComb:
        return PurifiedComb(self.tree.scale(factor), self.layout)

    # ------------------------------------------------------------------ sums
    def _direct_sum(self, other: PurifiedComb, mu_blocks: bool) -> PurifiedComb:
        return PurifiedComb._block_sum([self, other], mu_blocks)

    @staticmethod
    def _block_sum(parts: Sequence[PurifiedComb], mu_blocks: bool) -> PurifiedComb:
        """Block-diagonal sum of several combs (one pass, bond dimensions add)."""
        first = parts[0]
        for q in parts[1:]:
            if q.layout != first.layout:
                raise ShapeError("the combs have different layouts")
            if not mu_blocks and q.chi_mu != first.chi_mu:
                raise ShapeError(f"vector addition needs equal chi_mu, got {first.chi_mu} and {q.chi_mu}")
        if len(parts) == 1:
            return first
        mu = first.layout.mu
        tensors = {}
        for v in first.tree.nodes:
            blocks = [q.tree.tensors[v] for q in parts]
            nb = blocks[0].ndim - 1
            block_phys = mu_blocks and v == mu
            shape = [sum(b.shape[i] for b in blocks) for i in range(nb)]
            shape.append(sum(b.shape[-1] for b in blocks) if block_phys else blocks[0].shape[-1])
            T = np.zeros(shape, dtype=np.complex128)
            off = [0] * (nb + 1)
            for b in blocks:
                sl = tuple(slice(off[i], off[i] + b.shape[i]) for i in range(nb))
                sl += (slice(off[nb], off[nb] + b.shape[-1]),) if block_phys else (slice(None),)
                T[sl] = b
                for i in range(nb):
                    off[i] += b.shape[i]
                if block_phys:
                    off[nb] += b.shape[-1]
            tensors[v] = T
        return PurifiedComb(first.tree._with(tensors, None), first.layout)

    @staticmethod
    def vector_sum(parts: Sequence[PurifiedComb]) -> PurifiedComb:
        """``sum_k Psi_k`` in one block-diagonal pass (same ``mu`` index). Compress afterwards."""
        return PurifiedComb._block_sum(list(parts), mu_blocks=False)

    @staticmethod
    def matrix_sum(parts: Sequence[PurifiedComb]) -> PurifiedComb:
        """``sum_k rho_k``: ``Psi = [Psi_1 | Psi_2 | ...]`` in one pass. Compress afterwards."""
        return PurifiedComb._block_sum(list(parts), mu_blocks=True)

    def vector_add(self, other: PurifiedComb) -> PurifiedComb:
        """``Psi_1 + Psi_2`` (same ``mu`` index); bond dimensions add. Compress afterwards."""
        return self._direct_sum(other, mu_blocks=False)

    def matrix_add(self, other: PurifiedComb) -> PurifiedComb:
        """``rho_1 + rho_2``: ``Psi = [Psi_1 | Psi_2]`` (``chi_mu`` adds). Compress afterwards."""
        return self._direct_sum(other, mu_blocks=True)

    # ------------------------------------------------------------------ compression
    def compress(self, tol: float = 0.0, caps: Mapping[str, int] | None = None) -> tuple[PurifiedComb, float]:
        """Truncate every bond with the shared rule (per-kind ``caps``: ``chi_e``, ``chi_q``)."""
        tree, disc = self.tree.compress(tol, caps, root=self.layout.mu)
        return PurifiedComb(tree, self.layout), disc

    def _mu_svd(self) -> tuple[TreeTN, np.ndarray, np.ndarray, np.ndarray]:
        tree = self.tree.canonicalize(self.layout.mu)
        P = tree.tensors[self.layout.mu]  # (e, chi_mu)
        U, s, _ = np.linalg.svd(P, full_matrices=False)
        return tree, U, s, P

    def truncate_mu(self, tol: float = 0.0, chi_max: int | None = None) -> tuple[PurifiedComb, float]:
        """Truncate the purification index (eigenvalues of ``rho`` below the shared rule)."""
        from ..core.conventions import keep_count

        tree, U, s, _ = self._mu_svd()
        k, disc = keep_count(s, tol, chi_max)
        tensors = dict(tree.tensors)
        tensors[self.layout.mu] = U[:, :k] * s[:k]
        return PurifiedComb(tree._with(tensors, self.layout.mu), self.layout), disc

    # ------------------------------------------------------------------ observables
    def trace(self) -> float:
        """``Tr rho = ||Psi||^2``."""
        n = self.tree.norm() if self.tree.center is not None else self.tree.canonicalize(self.layout.mu).norm()
        return float(n**2)

    def normalize(self) -> PurifiedComb:
        tree = self.tree if self.tree.center is not None else self.tree.canonicalize(self.layout.mu)
        return PurifiedComb(tree.scale(1.0 / tree.norm()), self.layout)

    def expect(self, ops: Mapping[int, MPO] | Sequence[tuple[complex, Mapping[int, MPO]]]) -> complex:
        """``Tr(rho O) / Tr(rho)`` for a product ``{mode: MPO}`` or a sum ``[(coeff, {mode: MPO}), ...]``."""
        terms = [(1.0, ops)] if isinstance(ops, Mapping) else list(ops)
        tr = self.trace()
        total = 0.0 + 0j
        for c, prod in terms:
            out = self.apply_local(prod, tol=None)
            total += complex(c) * self.tree.inner(out.tree)
        return total / tr

    def purity(self) -> float:
        """``Tr rho^2 / (Tr rho)^2`` from the singular values on the ``mu`` index."""
        _, _, s, _ = self._mu_svd()
        w = s**2
        return float(np.sum(w**2) / np.sum(w) ** 2)

    def spectrum(self) -> np.ndarray:
        """Eigenvalues of ``rho / Tr rho`` (squared singular values on ``mu``), non-increasing."""
        _, _, s, _ = self._mu_svd()
        w = s**2
        return w / w.sum()

    def reduced_dm(self, mode: int) -> np.ndarray:
        """Dense reduced density matrix of ``mode`` (unit trace); ``2**bits[mode]`` levels."""
        if not 0 <= mode < self.num_modes:
            raise InputError(f"mode {mode} out of range 0..{self.num_modes - 1}")
        b = self.layout.backbone[mode]
        tree = self.tree.canonicalize(b)
        leg = self.layout.legs[mode]
        # contract the leg into one tensor (bond to b, N)
        acc = tree.tensors[leg[-1]]  # (parent, s)
        for q in reversed(leg[:-1]):
            t = tree.tensors[q]  # (parent, child, s)
            acc = np.einsum("pcs,cn->psn", t, acc).reshape(t.shape[0], -1)
        tb = tree.tensors[b]  # (..., to q1, 1)
        ax = tree.axis(b, leg[0])
        tb = np.moveaxis(tb[..., 0], ax, -1)  # (other bonds..., q1)
        X = np.tensordot(tb, acc, axes=([tb.ndim - 1], [0])).reshape(-1, acc.shape[1])  # (rest, N)
        rho = X.T @ X.conj()
        return rho / np.trace(rho).real

    def mean_photons(self) -> np.ndarray:
        """``<n>`` of every mode (exact: the number operator is a rank-2 diagonal MPO)."""
        from ..bosons.operators import number

        return np.array([self.expect({m: number(self.bits[m])}).real for m in range(self.num_modes)])
