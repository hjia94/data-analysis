"""data_analysis.io — readers grouped by data shape, plus output-path resolver.

The only place in the package where a file format is parsed. Output locations are
resolved centrally via :mod:`data_analysis.io.paths`.

The unified LAPD HDF5 reader is the public entry point for experiment code:
``from data_analysis.io import open_lapd``. It auto-detects the file's provenance
and returns a :class:`~data_analysis.io.lapd_hdf5.LapdRun`; the per-provenance
backends under :mod:`data_analysis.io._backends` are private implementation detail.
"""

from .lapd_hdf5 import open_lapd, LapdRun, LapdSession, compare_runs, gas_puff

__all__ = ["open_lapd", "LapdRun", "LapdSession", "compare_runs", "gas_puff"]
