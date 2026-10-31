"""Hardware models (layer 3): sources, mode bases, detectors, phase locks, setups.

* :mod:`.jsa` — joint spectral amplitude of degenerate down-conversion (pump × phase matching) on
  a frequency grid, and its Schmidt supermodes (S11);
* :mod:`.sources` — synchronously pumped OPO (cavity, threshold, escape efficiency, squeezing
  spectrum) and single-pass waveguide (distributed loss), producing the supermode state with
  every imperfection booked in a :class:`~cvsim.gaussian.NoiseLedger` (S11);
* :mod:`.excess` — excess antisqueezing (phase-sensitive added noise, gain fluctuations) and the
  calibration of squeezing / antisqueezing measurements (S11).

Mode bases and the pulse shaper (S12), detectors and phase noise (S13) and serialisable setups
(S14) complete the layer.
"""

from __future__ import annotations

from . import excess, jsa, sources

__all__ = ["excess", "jsa", "sources"]
