"""cvsim.ir: operations, programs, binding, JSON round trips, validation."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cvsim.errors import InputError, ShapeError
from cvsim.ir import SCHEMA, SPECS, Mode, Op, Parameter, Program, op

SETTINGS = settings(max_examples=60, deadline=None)

# one valid value per parameter of every registered operation
_VALUES = {
    "alpha": 0.3 - 0.1j,
    "r": 0.4,
    "phi": 0.7,
    "nbar": 0.2,
    "n": 2,
    "theta": 0.5,
    "s": -0.3,
    "g": 0.6,
    "kappa": 0.1,
    "eta": 0.8,
    "gain": 1.5,
    "sigma2": 0.05,
    "sigma": 0.02,
    "efficiency": 0.9,
    "electronic": 0.01,
    "outcome": None,
}


def sample_op(name: str, first_mode: int, rng: np.random.Generator) -> Op:
    spec = SPECS[name]
    params = {k: _VALUES[k] for k in spec.checks if k in _VALUES}
    if name == "Interferometer":
        q, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        params["U"] = q
    if name == "Symplectic":
        params["S"] = np.eye(4)
    n = spec.num_modes or 2
    return Op(name, params, tuple(range(first_mode, first_mode + n)))


@st.composite
def programs(draw):
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    names = draw(st.lists(st.sampled_from(sorted(n for n in SPECS if SPECS[n].kind != "measure")), max_size=8))
    ops = [sample_op(n, 0, rng) for n in names]
    if draw(st.booleans()):
        ops.append(op("Homodyne", 2, phi=draw(st.floats(-3, 3)), outcome=draw(st.floats(-2, 2))))
    if draw(st.booleans()):
        ops.append(op("S", 1, r=Parameter("r_free")))
    return Program(3, ops, name=draw(st.sampled_from(["", "test"])))


# ------------------------------------------------------------------------------- ops
class TestOps:
    def test_defaults_and_order(self):
        o = op("S", 0, r=0.5)
        assert dict(o.params) == {"r": 0.5, "phi": 0.0} and o.kind == "gate" and o.modes == (0,)
        assert repr(o) == "S(r=0.5, phi=0.0) on [0]"

    def test_every_spec_accepts_its_sample(self):
        rng = np.random.default_rng(0)
        for name in SPECS:
            o = sample_op(name, 0, rng)
            assert o.kind == SPECS[name].kind

    @pytest.mark.parametrize(
        ("call", "exc", "match"),
        [
            (lambda: op("Squeeze", 0, r=0.1), InputError, "unknown operation"),
            (lambda: op("S", 0, rr=0.1), InputError, "unknown parameter"),
            (lambda: op("BS", (0, 1)), InputError, "missing"),
            (lambda: op("Loss", 0, eta=1.5), InputError, "eta"),
            (lambda: op("Homodyne", 0, efficiency=0.0), InputError, "efficiency"),
            (lambda: op("R", 0, theta=math.nan), InputError, "theta"),
            (lambda: op("R", 0, theta=True), InputError, "real number"),
            (lambda: op("Fock", 0, n=-1), InputError, "integer"),
            (lambda: op("BS", 0, theta=0.1), ShapeError, "2 mode"),
            (lambda: op("BS", (1, 1), theta=0.1), InputError, "distinct"),
            (lambda: op("Interferometer", (0, 1), U=np.eye(3)), ShapeError, "3 mode"),
            (lambda: op("Symplectic", (0,), S=np.eye(3)), ShapeError, "2m"),
            (lambda: op("Symplectic", (0,), S=np.eye(2) * 1j), InputError, "real"),
            (lambda: Parameter("1x"), InputError, "identifier"),
        ],
    )
    def test_validation(self, call, exc, match):
        with pytest.raises(exc, match=match):
            call()

    def test_arrays_are_frozen_copies(self):
        U = np.eye(2, dtype=complex)
        o = op("Interferometer", (0, 1), U=U)
        U[0, 0] = 5
        assert o.params["U"][0, 0] == 1
        with pytest.raises(ValueError):
            o.params["U"][0, 0] = 2

    def test_bind_validates(self):
        o = op("Loss", 0, eta=Parameter("t"))
        assert o.free_parameters == {"t"}
        assert o.bind({"t": 0.5}).params["eta"] == 0.5
        with pytest.raises(InputError, match="eta"):
            o.bind({"t": 2.0})


# ------------------------------------------------------------------------------- programs
class TestProgram:
    def test_composition_is_immutable_and_ordered(self):
        p = Program(2)
        q = p >> op("S", 0, r=0.3)
        r = q >> Program(2, [op("BS", (0, 1), theta=0.4)])
        assert len(p) == 0 and len(q) == 1 and [o.name for o in r] == ["S", "BS"]
        assert (q >> [op("R", 1, theta=0.1), op("F", 0)]).ops[-1].name == "F"

    def test_modes(self):
        p = Program(["signal", Mode("idler", "the other arm")])
        assert p.mode_index("idler") == 1 and p.modes[1].description == "the other arm"
        with pytest.raises(InputError):
            p.mode_index("pump")
        with pytest.raises(InputError):
            Program(["a", "a"])
        with pytest.raises(InputError):
            Program(0)

    def test_measured_modes_and_reuse(self):
        p = Program(2) >> op("Homodyne", 0) >> op("Vacuum", 0) >> op("S", 0, r=0.1) >> op("Heterodyne", 1)
        assert p.measured_modes == (1,)
        with pytest.raises(InputError, match="measured mode"):
            p >> op("BS", (0, 1), theta=0.1)
        with pytest.raises(InputError, match="has 2"):
            p >> op("R", 2, theta=0.1)

    def test_gaussianity(self):
        assert (Program(1) >> op("S", 0, r=0.1)).is_gaussian
        assert not (Program(1) >> op("Kerr", 0, kappa=0.1)).is_gaussian

    def test_bind(self):
        p = Program(2) >> op("S", 0, r=Parameter("r")) >> op("BS", (0, 1), theta=Parameter("t"), phi=Parameter("r"))
        assert p.parameters == {"r", "t"} and not p.is_bound
        b = p.bind({"r": 0.2}, t=0.4)
        assert b.is_bound and b.ops[1].params["phi"] == 0.2 and p.parameters == {"r", "t"}
        with pytest.raises(InputError, match="no free parameter"):
            p.bind(x=1.0)

    @SETTINGS
    @given(p=programs())
    def test_json_round_trip(self, p):
        text = p.to_json()
        assert json.loads(text)["schema"] == SCHEMA
        q = Program.from_json(text)
        assert q == p and q.parameters == p.parameters
        assert Program.from_json(p.to_json(indent=None)) == p

    def test_json_rejects(self):
        good = (Program(1) >> op("R", 0, theta=0.1)).to_dict()
        with pytest.raises(InputError, match="schema"):
            Program.from_dict({**good, "schema": "other/9"})
        bad = json.loads(json.dumps(good))
        bad["ops"][0]["name"] = "Teleport"
        with pytest.raises(InputError, match="unknown name"):
            Program.from_dict(bad)
        with pytest.raises(InputError, match="JSON"):
            Program.from_json("{not json")
        bad = json.loads(json.dumps(good))
        bad["ops"][0]["params"]["theta"] = {"weird": 1}
        with pytest.raises(InputError, match="decode"):
            Program.from_dict(bad)

    def test_provenance_survives_json(self):
        p = Program(1) >> op("Loss", 0, eta=0.9, component="detector", noise_kind="loss")
        assert Program.from_json(p.to_json()).ops[0].component == "detector"

    def test_not_an_op(self):
        with pytest.raises(InputError, match="not an Op"):
            Program(1, ["S"])  # type: ignore[list-item]

    def test_repr(self):
        assert repr(Program(2, name="x") >> op("S", 0, r=Parameter("a"))) == "Program 'x'(modes=2, ops=1, free=['a'])"
