"""Tree tensor networks, and the comb of the paper as one configuration of them.

A :class:`TreeTN` is a tree on nodes ``0..n-1``. Node ``v`` holds a tensor with **one axis per
neighbour, in ascending neighbour order, followed by one physical axis** (dimension 1 for nodes
without a physical index, like the backbone of the comb). Every edge has a *kind* (``"backbone"``,
``"leg"``, ``"purification"``, ...), so that bond dimensions can be capped per kind.

Because a tree has no loops, every edge splits the network into two parts: every edge has a
Schmidt decomposition and an optimal truncation, exactly as in a chain. With an orthogonality
centre ``c``, every other node is an isometry towards its neighbour on the path to ``c``.

* :meth:`TreeTN.canonicalize` -- QR sweeps in post-order towards the centre;
* :meth:`TreeTN.compress` -- depth-first truncated SVDs with the centre always adjacent to the
  edge being truncated (optimal per edge);
* :func:`comb_product_state` -- the comb of Moulinas & Waintal (backbone of one node per mode,
  one quantics leg per mode, one purification leaf) from single-mode tensor trains.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..errors import InputError, ShapeError
from . import linalg
from .tt import TensorTrain

__all__ = ["CAP_ALIASES", "CombLayout", "TreeTN", "comb_product_state"]

#: Accepted names for per-kind bond caps: the paper's symbols or the edge kinds.
CAP_ALIASES = {"chi_e": "backbone", "chi_q": "leg", "chi_mu": "purification"}


def _edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


class TreeTN:
    """A tree tensor network (see the module docstring for the axis convention).

    Parameters
    ----------
    tensors : mapping node -> ndarray
    edges : sequence of (u, v)
    edge_kinds : mapping (u, v) -> str, optional (default ``"bond"``)
    center : int, optional
        Orthogonality centre, if the caller guarantees one.
    """

    __slots__ = ("adj", "center", "kinds", "tensors")
    tensors: dict[int, np.ndarray]
    adj: dict[int, tuple[int, ...]]
    kinds: dict[tuple[int, int], str]
    center: int | None

    def __init__(
        self,
        tensors: Mapping[int, np.ndarray],
        edges: Iterable[tuple[int, int]],
        edge_kinds: Mapping[tuple[int, int], str] | None = None,
        center: int | None = None,
        *,
        copy: bool = True,
        validate: bool = True,
    ):
        nodes = sorted(tensors)
        adj: dict[int, list[int]] = {v: [] for v in nodes}
        elist = [_edge(int(u), int(v)) for u, v in edges]
        for u, v in elist:
            if u == v or u not in adj or v not in adj:
                raise ShapeError(f"invalid edge ({u}, {v}) for nodes {nodes}")
            adj[u].append(v)
            adj[v].append(u)
        self.adj = {v: tuple(sorted(n)) for v, n in adj.items()}
        self.kinds = {e: "bond" for e in elist}
        if edge_kinds:
            for (u, v), k in edge_kinds.items():
                e = _edge(u, v)
                if e not in self.kinds:
                    raise ShapeError(f"edge kind given for a missing edge {e}")
                self.kinds[e] = str(k)
        if copy:
            self.tensors = {v: np.array(tensors[v], dtype=np.complex128) for v in nodes}
        else:
            self.tensors = {v: np.asarray(tensors[v], dtype=np.complex128) for v in nodes}
        self.center = center
        if validate:
            self._validate(len(elist))

    def _validate(self, n_edges: int) -> None:
        n = len(self.tensors)
        if n == 0:
            raise ShapeError("a tree needs at least one node")
        if n_edges != n - 1 or len(set(self.kinds)) != n_edges:
            raise ShapeError(f"a tree on {n} nodes has {n - 1} distinct edges, got {n_edges}")
        start = next(iter(self.tensors))
        seen = {start}
        stack = [start]
        while stack:
            v = stack.pop()
            for u in self.adj[v]:
                if u not in seen:
                    seen.add(u)
                    stack.append(u)
        if len(seen) != n:
            raise ShapeError("the graph is not connected (or has a cycle)")
        for v, t in self.tensors.items():
            if t.ndim != len(self.adj[v]) + 1:
                raise ShapeError(f"node {v}: tensor has {t.ndim} axes, expected {len(self.adj[v])} bonds + 1 physical")
        for u, v in self.kinds:
            if self.bond_dim(u, v) != self.tensors[v].shape[self.axis(v, u)]:
                raise ShapeError(
                    f"bond ({u}, {v}): dimension {self.tensors[u].shape[self.axis(u, v)]} at node {u} but "
                    f"{self.tensors[v].shape[self.axis(v, u)]} at node {v}"
                )
        if self.center is not None and self.center not in self.tensors:
            raise InputError(f"center {self.center} is not a node")

    # ------------------------------------------------------------------ structure
    @property
    def nodes(self) -> list[int]:
        return sorted(self.tensors)

    @property
    def edges(self) -> list[tuple[int, int]]:
        return sorted(self.kinds)

    def neighbors(self, v: int) -> tuple[int, ...]:
        return self.adj[v]

    def axis(self, v: int, u: int) -> int:
        """Axis of node ``v``'s tensor that carries the bond to neighbour ``u``."""
        return self.adj[v].index(u)

    def bond_dim(self, u: int, v: int) -> int:
        return int(self.tensors[u].shape[self.axis(u, v)])

    def phys_dim(self, v: int) -> int:
        return int(self.tensors[v].shape[-1])

    def kind(self, u: int, v: int) -> str:
        return self.kinds[_edge(u, v)]

    def max_bond(self, kind: str | None = None) -> int:
        dims = [self.bond_dim(u, v) for (u, v), k in self.kinds.items() if kind is None or k == kind]
        return max(dims, default=1)

    def copy(self) -> TreeTN:
        return TreeTN(self.tensors, self.kinds, self.kinds, self.center, copy=True, validate=False)

    def _with(self, tensors: dict[int, np.ndarray], center: int | None) -> TreeTN:
        out = TreeTN.__new__(TreeTN)
        out.tensors = tensors
        out.adj = self.adj
        out.kinds = self.kinds
        out.center = center
        return out

    def __repr__(self) -> str:
        bonds = {k: self.max_bond(k) for k in sorted(set(self.kinds.values()))}
        return f"TreeTN(nodes={len(self.tensors)}, max bonds={bonds}, center={self.center})"

    # ------------------------------------------------------------------ traversals
    def parents(self, root: int) -> dict[int, int | None]:
        """Parent of every node when the tree is rooted at ``root`` (iterative BFS)."""
        par: dict[int, int | None] = {root: None}
        queue = [root]
        for v in queue:
            for u in self.adj[v]:
                if u not in par:
                    par[u] = v
                    queue.append(u)
        return par

    def postorder(self, root: int) -> list[int]:
        """Children before parents (the root last)."""
        par = self.parents(root)
        order: list[int] = []
        stack = [(root, False)]
        while stack:
            v, done = stack.pop()
            if done:
                order.append(v)
                continue
            stack.append((v, True))
            for u in reversed(self.adj[v]):
                if par.get(u) == v:
                    stack.append((u, False))
        return order

    def path(self, u: int, v: int) -> list[int]:
        """Nodes on the path from ``u`` to ``v`` (inclusive)."""
        par = self.parents(v)
        out = [u]
        while out[-1] != v:
            p = par[out[-1]]
            assert p is not None
            out.append(p)
        return out

    # ------------------------------------------------------------------ local moves
    def _qr_towards(self, tensors: dict[int, np.ndarray], v: int, p: int) -> None:
        """Make ``v`` an isometry towards ``p``; absorb the R factor into ``p`` (in place on ``tensors``)."""
        t = tensors[v]
        ax = self.axis(v, p)
        moved = np.moveaxis(t, ax, -1)
        q, r = linalg.qr(moved, split=t.ndim - 1)  # q: (..., k), r: (k, d_p)
        tensors[v] = np.moveaxis(q, -1, ax)
        axp = self.axis(p, v)
        tp = tensors[p]
        new = linalg.contract(r, [1], tp, [axp])  # (k, rest of p)
        tensors[p] = np.moveaxis(new, 0, axp)

    def _svd_across(self, tensors: dict[int, np.ndarray], p: int, c: int, tol: float, chi_max: int | None) -> float:
        """Centre at ``p``: truncated SVD of the bond (p, c); ``p`` keeps U, ``c`` absorbs S Vh (new centre)."""
        t = tensors[p]
        ax = self.axis(p, c)
        moved = np.moveaxis(t, ax, -1)
        res = linalg.svd(moved, split=t.ndim - 1, tol=tol, chi_max=chi_max)
        tensors[p] = np.moveaxis(res.U, -1, ax)
        sv = res.S[:, None] * res.Vh  # (k, d_c)
        axc = self.axis(c, p)
        new = linalg.contract(sv, [1], tensors[c], [axc])
        tensors[c] = np.moveaxis(new, 0, axc)
        return res.discarded

    # ------------------------------------------------------------------ canonical forms
    def canonicalize(self, center: int) -> TreeTN:
        """An equivalent network whose every node but ``center`` is an isometry towards ``center``."""
        if center not in self.tensors:
            raise InputError(f"center {center} is not a node")
        tensors = dict(self.tensors)
        if self.center is not None:  # only the path between the old and the new centre changes
            path = self.path(self.center, center)
            for v, p in zip(path[:-1], path[1:]):
                self._qr_towards(tensors, v, p)
            return self._with(tensors, center)
        par = self.parents(center)
        for v in self.postorder(center):
            parent = par[v]
            if parent is not None:
                self._qr_towards(tensors, v, parent)
        return self._with(tensors, center)

    def compress(
        self,
        tol: float = 0.0,
        caps: Mapping[str, int] | None = None,
        *,
        root: int | None = None,
        only: Iterable[tuple[int, int]] | None = None,
    ) -> tuple[TreeTN, float]:
        """Truncate every bond (or only the listed edges) with the shared rule; per-kind caps.

        Canonicalise to ``root``, then walk the tree depth first: the centre moves down an edge by
        a truncated SVD (so every truncation happens next to the centre, hence optimally) and back
        up by a QR. Subtrees without any edge to truncate are not visited. Returns the compressed
        network (centre at ``root``) and the summed discarded weight.
        """
        caps = {CAP_ALIASES.get(k, k): int(v) for k, v in (caps or {}).items()}
        if root is None:
            root = self.center if self.center is not None else self.nodes[0]
        net = self.canonicalize(root)
        tensors = dict(net.tensors)
        par = self.parents(root)
        children = {v: [u for u in self.adj[v] if par.get(u) == v] for v in self.adj}
        wanted = None if only is None else {_edge(u, v) for u, v in only}
        needed: dict[int, bool] = {}  # does the subtree hanging below edge (parent, v) contain work?
        for v in self.postorder(root):
            p = par[v]
            here = p is not None and (wanted is None or _edge(p, v) in wanted)
            needed[v] = here or any(needed[c] for c in children[v])
        discarded = 0.0
        stack: list[list[int]] = [[root, 0]]
        while stack:
            frame = stack[-1]
            v, i = frame
            if i < len(children[v]):
                frame[1] += 1
                c = children[v][i]
                if not needed[c]:
                    continue
                e = _edge(v, c)
                if wanted is None or e in wanted:
                    discarded += self._svd_across(tensors, v, c, tol, caps.get(self.kinds[e]))
                else:
                    self._qr_towards(tensors, v, c)
                stack.append([c, 0])
            else:
                stack.pop()
                if stack:
                    self._qr_towards(tensors, v, stack[-1][0])
        return self._with(tensors, root), discarded

    # ------------------------------------------------------------------ contractions
    def _contract_all(self, phys_order: Sequence[int]) -> np.ndarray:
        root = self.nodes[0]
        par = self.parents(root)
        order = list(par)  # BFS order
        acc = self.tensors[root]
        labels: list[tuple] = [("e", root, u) for u in self.adj[root]] + [("p", root)]
        for v in order[1:]:
            p = par[v]
            assert p is not None
            t = self.tensors[v]
            tl = [("e", v, u) for u in self.adj[v]] + [("p", v)]
            ia = labels.index(("e", p, v))
            ib = tl.index(("e", v, p))
            acc = linalg.contract(acc, [ia], t, [ib])
            labels = labels[:ia] + labels[ia + 1 :] + tl[:ib] + tl[ib + 1 :]
        perm = [labels.index(("p", v)) for v in phys_order]
        rest = [i for i in range(len(labels)) if i not in perm]
        if any(acc.shape[i] != 1 for i in rest):
            raise ShapeError("phys_order must list every node with a physical dimension > 1")
        return acc.transpose(perm + rest).reshape([acc.shape[i] for i in perm])

    def to_dense(self, phys_order: Sequence[int] | None = None) -> np.ndarray:
        """Dense vector with the physical axes in ``phys_order`` (default: ascending node ids)."""
        order = list(phys_order) if phys_order is not None else self.nodes
        return self._contract_all(order).reshape(-1)

    def inner(self, other: TreeTN) -> complex:
        """``<self|other>`` for two networks on the same tree (bond dimensions may differ)."""
        if self.adj != other.adj:
            raise ShapeError("inner: the two networks are not on the same tree")
        root = self.nodes[0]
        par = self.parents(root)
        envs: dict[int, np.ndarray] = {}
        for v in self.postorder(root):
            nb = self.adj[v]
            deg = len(nb)
            A = list(range(deg))  # bra bond labels
            B = list(range(deg, 2 * deg))  # ket bond labels
            P = 2 * deg  # physical label
            operands: list = [self.tensors[v].conj(), A + [P], other.tensors[v], B + [P]]
            out: list[int] = []
            for k, u in enumerate(nb):
                if par.get(u) == v:
                    operands += [envs.pop(u), [A[k], B[k]]]
                else:  # the parent bond stays open
                    out = [A[k], B[k]]
            operands.append(out)
            t = np.einsum(*operands, optimize=True)
            if par[v] is None:
                return complex(t)
            envs[v] = t
        raise AssertionError("unreachable")

    def norm(self) -> float:
        if self.center is not None:
            return float(np.linalg.norm(self.tensors[self.center]))
        return float(np.sqrt(max(self.inner(self).real, 0.0)))

    def scale(self, factor: complex) -> TreeTN:
        tensors = dict(self.tensors)
        v = self.center if self.center is not None else self.nodes[0]
        tensors[v] = tensors[v] * factor
        return self._with(tensors, self.center)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_tensor_train(cls, tt: TensorTrain, kind: str = "bond") -> TreeTN:
        """The chain ``0 - 1 - ... - L-1`` (core ``(l, s, r)`` -> node tensor ``(l, r, s)``)."""
        L = tt.length
        tensors = {}
        for i, c in enumerate(tt.cores):
            if L == 1:
                tensors[i] = c.reshape(-1)  # a single node: only the physical axis
            elif i == 0:
                tensors[i] = c[0].T.copy()  # (r, s)
            elif i == L - 1:
                tensors[i] = c[:, :, 0]  # (l, s)
            else:
                tensors[i] = c.transpose(0, 2, 1)  # (l, r, s)
        edges = [(i, i + 1) for i in range(L - 1)]
        return cls(tensors, edges, {e: kind for e in edges}, center=tt.center)


# ============================================================================= the comb


@dataclass(frozen=True)
class CombLayout:
    """Node ids of a comb: backbone ``b_alpha``, legs ``q_{alpha, 1..R_alpha}`` (MSB first), ``mu``."""

    backbone: tuple[int, ...]
    legs: tuple[tuple[int, ...], ...]
    mu: int

    @property
    def num_modes(self) -> int:
        return len(self.backbone)

    @property
    def bits(self) -> tuple[int, ...]:
        return tuple(len(leg) for leg in self.legs)

    def phys_order(self) -> list[int]:
        """Dense ordering: modes in order, bits MSB first within a mode, ``mu`` last."""
        return [q for leg in self.legs for q in leg] + [self.mu]

    @classmethod
    def build(cls, bits: Sequence[int]) -> tuple[CombLayout, list[tuple[int, int]], dict[tuple[int, int], str]]:
        M = len(bits)
        if M == 0 or any(b < 1 for b in bits):
            raise InputError(f"a comb needs >= 1 mode with >= 1 bit each, got {list(bits)}")
        backbone = tuple(range(M))
        mu = M
        nxt = M + 1
        legs = []
        for b in bits:
            legs.append(tuple(range(nxt, nxt + b)))
            nxt += b
        edges: list[tuple[int, int]] = []
        kinds: dict[tuple[int, int], str] = {}
        for a in range(M - 1):
            edges.append((a, a + 1))
            kinds[(a, a + 1)] = "backbone"
        edges.append((M - 1, mu))
        kinds[(M - 1, mu)] = "purification"
        for a, leg in enumerate(legs):
            chain = [a, *leg]
            for u, v in zip(chain[:-1], chain[1:]):
                edges.append((u, v))
                kinds[(u, v)] = "leg"
        return cls(backbone, tuple(legs), mu), edges, kinds


def comb_product_state(legs: Sequence[TensorTrain]) -> tuple[TreeTN, CombLayout]:
    """The comb of a product state ``|psi_0> (x) |psi_1> (x) ...`` with ``chi_mu = 1``.

    Each leg is a single-mode quantics tensor train with physical dimension 2.
    """
    for i, tt in enumerate(legs):
        if any(d != 2 for d in tt.dims):
            raise ShapeError(f"leg {i} must be a quantics tensor train (physical dimension 2), got dims {tt.dims}")
    layout, edges, kinds = CombLayout.build([tt.length for tt in legs])
    tensors: dict[int, np.ndarray] = {}
    adj: dict[int, list[int]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    for b in layout.backbone:
        tensors[b] = np.ones((1,) * len(adj[b]) + (1,), dtype=np.complex128)
    tensors[layout.mu] = np.ones((1, 1), dtype=np.complex128)
    for tt, leg in zip(legs, layout.legs):
        for c, q in zip(tt.cores, leg):
            n_nb = len(adj[q])
            if n_nb == 1:  # last bit of the leg: only the parent bond
                tensors[q] = c[:, :, 0]  # (l, s)
            else:
                tensors[q] = c.transpose(0, 2, 1)  # (l, r, s): parent id < child id
    return TreeTN(tensors, edges, kinds), layout
