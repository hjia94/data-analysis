"""LAPD HDF5 reader backends — provenance-specific parsers hidden behind the
``open_lapd`` dispatcher in :mod:`data_analysis.io.lapd_hdf5`.

Each backend is a near-verbatim move of an existing reader (wrap, don't rewrite):

- :mod:`~data_analysis.io._backends.bapsflib_daq` — LabVIEW DAQ + C translator (bapsflib)
- :mod:`~data_analysis.io._backends.pydaq`        — Python LAPD_DAQ pipeline
- :mod:`~data_analysis.io._backends.legacy_2018`  — 2018-2020 process-plasma layout

Alongside the readers, :mod:`~data_analysis.io._backends.run_description` is a
tolerant parser + diff for the hand-written pydaq ``description`` attribute (pure
text, no HDF5), reached via ``LapdRun.description()`` and ``compare_runs``.

Import these through ``open_lapd`` / ``compare_runs`` rather than directly; they
are an internal (underscore) package.
"""
