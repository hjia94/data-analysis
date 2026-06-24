"""Unified LAPD HDF5 reader: :func:`open_lapd` + :class:`LapdRun`.

Three provenances of LAPD HDF5 file are parsed by three backends under
:mod:`data_analysis.io._backends` (moved near-verbatim from the old readers in
Step 3 of the reorg). :func:`open_lapd` sniffs the file's group signatures,
picks a backend, and returns a :class:`LapdRun` that delegates to that backend's
existing functions. Behavior is preserved (wrap, don't rewrite); the old reader
module names still resolve through thin shims, so existing experiment code is
unaffected.

Handle lifecycle
----------------
``open_lapd`` does **not** keep the HDF5 file open. It records the path + backend
only; every :class:`LapdRun` method opens the handle, reads the requested slice,
and closes it. (The bapsflib backend's ``lapd.File`` is likewise opened and
closed inside each call.) This matches the no-leaked-handle requirement and the
way the pydaq/legacy readers already worked.

Shot selection
--------------
Large runs must be readable in part or streamed shot-by-shot rather than loaded
whole. ``channel(name, shots=...)`` reads a chosen subset into a rectangular
``(nshot, nsamples)`` stack; ``iter_shots(name, shots=...)`` yields one shot at a
time (the read-and-analyze / process-and-discard pattern); ``shots()`` lists the
available shot numbers. The pydaq backend supports these natively via
``scope_io``; the bapsflib backend maps ``shots`` onto its ``index`` argument;
the legacy 2018 layout has no per-shot concept and raises ``NotImplementedError``.
"""

import os
from contextlib import contextmanager

import h5py

# Backend modules and scope_io are imported lazily inside the methods that need
# them (see _backend_module / _open_pydaq_scope): bapsflib_daq pulls in bapsflib
# + matplotlib, so importing them eagerly would load those heavy deps even when
# opening a pydaq/legacy file. Matches the lazy-import convention in io/scope.py.

__all__ = ["open_lapd", "LapdRun", "detect_backend"]


# --------------------------------------------------------------------------- #
# format sniffing
# --------------------------------------------------------------------------- #
def detect_backend(path):
    """Return the backend name for ``path``: 'bapsflib' | 'pydaq' | 'legacy'.

    Sniffs group signatures without holding the file open:

    - ``Raw data + config`` present  -> 'bapsflib' (LabVIEW DAQ + C translator)
    - ``Acquisition/LeCroy_scope``   -> 'legacy'   (2018-2020 process-plasma)
    - else, if a LAPD_DAQ layout is recognized (``Control/Positions`` or a scope
      group containing ``shot_*`` subgroups) -> 'pydaq'

    Raises ``ValueError`` if the file matches no known layout.
    """
    with h5py.File(path, "r") as f:
        if "Raw data + config" in f:
            return "bapsflib"
        if "LeCroy_scope" in f.get("Acquisition", {}):
            return "legacy"
        if "Positions" in f.get("Control", {}):
            return "pydaq"
        # LAPD_DAQ scope groups: a top-level group with shot_* subgroups.
        if any(_has_shot_groups(item) for item in f.values()):
            return "pydaq"
        top = sorted(f.keys())

    raise ValueError(
        f"Unrecognized LAPD HDF5 layout: {path!r}. Top-level groups: {top}. "
        f"Expected a bapsflib ('Raw data + config'), legacy "
        f"('Acquisition/LeCroy_scope'), or LAPD_DAQ pydaq (scope groups / "
        f"'Control/Positions') file."
    )


def _has_shot_groups(item):
    """True if ``item`` is an HDF5 group containing ``shot_*`` subgroups."""
    return isinstance(item, h5py.Group) and any(
        name.startswith("shot_") for name in item
    )


def open_lapd(path):
    """Open an LAPD HDF5 file and return a :class:`LapdRun` for it.

    Detects the file's provenance and selects the matching backend. Does not keep
    the file open -- see the module docstring for the handle lifecycle.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LAPD HDF5 file not found: {path!r}")
    backend = detect_backend(path)
    return LapdRun(path, backend)


# --------------------------------------------------------------------------- #
# unified run interface
# --------------------------------------------------------------------------- #
class LapdRun:
    """A single LAPD data run behind one interface, regardless of provenance.

    Construct via :func:`open_lapd`. Methods delegate to the selected backend and
    return that backend's native output (documented per method); where a backend
    lacks a concept the method raises ``NotImplementedError``. No HDF5 handle is
    held between calls.
    """

    def __init__(self, path, backend):
        self.path = path
        self._backend = backend

    @property
    def backend(self):
        """Backend name: ``'bapsflib'`` | ``'pydaq'`` | ``'legacy'``."""
        return self._backend

    def __repr__(self):
        return f"LapdRun(path={self.path!r}, backend={self._backend!r})"

    # -- lazy backend-module loaders (keep bapsflib/matplotlib off import path) - #
    @staticmethod
    def _bapsflib_daq():
        from ._backends import bapsflib_daq
        return bapsflib_daq

    @staticmethod
    def _pydaq():
        from ._backends import pydaq
        return pydaq

    @staticmethod
    def _legacy_2018():
        from ._backends import legacy_2018
        return legacy_2018

    # -- bapsflib needs an open lapd.File; open/close it around a callback ---- #
    def _with_bapsflib_file(self, fn):
        from bapsflib import lapd
        f = lapd.File(self.path)
        try:
            return fn(f)
        finally:
            f.close()

    # -- pydaq: open the file, resolve the scope group, hand over scope wrappers #
    @contextmanager
    def _open_pydaq_scope(self, scope_name):
        """Yield ``(f, resolved_scope_name, scope)`` for a pydaq read.

        Centralizes the open + scope-group resolution + ``data_analysis.io.scope``
        import that every pydaq method otherwise repeats. The HDF5 handle is open
        only for the ``with`` body (and across yields, so generators stay valid).
        """
        from . import scope
        with h5py.File(self.path, "r") as f:
            yield f, self._resolve_scope_name(f, scope_name), scope

    # ----------------------------------------------------------------------- #
    # describe
    # ----------------------------------------------------------------------- #
    def info(self):
        """Print a human-readable overview of the file.

        Delegates to ``show_info`` (bapsflib), ``print_info`` (pydaq), or
        ``print_data_objects`` (legacy). Returns ``None`` (prints), matching the
        original readers.
        """
        if self._backend == "bapsflib":
            return self._with_bapsflib_file(self._bapsflib_daq().show_info)
        if self._backend == "pydaq":
            return self._pydaq().print_info(self.path)
        return self._legacy_2018().print_data_objects(self.path)

    # ----------------------------------------------------------------------- #
    # positions
    # ----------------------------------------------------------------------- #
    def positions(self, **kwargs):
        """Read probe motion / positions.

        Returns the selected backend's native output:

        - bapsflib: ``read_probe_motion_bmotion(f)`` (falls back to
          ``read_probe_motion_6k(f)`` if no bmotion group) ->
          ``(pos_array, xpos, ypos, zpos, npos, nshot)``
        - pydaq: ``read_positions(path, motion_group_name=...)`` ->
          ``(motion_list, pos_array, npos, nshot)``
        - legacy: ``read_position_data(path)`` -> ``(pos_array, xpos, ypos, zpos)``
        """
        if self._backend == "bapsflib":
            bapsflib_daq = self._bapsflib_daq()
            def _read(f):
                result = bapsflib_daq.read_probe_motion_bmotion(f)
                if result is None:
                    result = bapsflib_daq.read_probe_motion_6k(f)
                return result
            return self._with_bapsflib_file(_read)
        if self._backend == "pydaq":
            return self._pydaq().read_positions(self.path, **kwargs)
        return self._legacy_2018().read_position_data(self.path)

    # ----------------------------------------------------------------------- #
    # shots
    # ----------------------------------------------------------------------- #
    def shots(self, scope_name=None):
        """Return the available shot numbers (LAPD_DAQ pydaq files only).

        Delegates to ``scope.scope_shot_numbers``. ``scope_name`` selects the
        scope group (required if the file has more than one). Raises
        ``NotImplementedError`` for the bapsflib and legacy layouts, which have no
        per-shot scope-group concept.
        """
        self._require_pydaq("shots()")
        with self._open_pydaq_scope(scope_name) as (f, scope_name, scope):
            return scope.scope_shot_numbers(f[scope_name])

    def _require_pydaq(self, what):
        """Raise NotImplementedError if this is not a pydaq file."""
        if self._backend != "pydaq":
            raise NotImplementedError(
                f"{what} is only available for LAPD_DAQ (pydaq) files; this is a "
                f"{self._backend!r} file."
            )

    @staticmethod
    def _resolve_scope_name(f, scope_name):
        """Pick the scope group: the given name, or the sole scope group."""
        if scope_name is not None:
            if scope_name not in f:
                raise KeyError(f"Scope group {scope_name!r} not found in file.")
            return scope_name
        candidates = [k for k in f if _has_shot_groups(f[k])]
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(
            f"Multiple or no scope groups found ({candidates}); pass scope_name=."
        )

    # ----------------------------------------------------------------------- #
    # channel data
    # ----------------------------------------------------------------------- #
    def channel(self, name, shots=None, scope_name=None, **kwargs):
        """Read channel data.

        Backend-specific ``name``/return (native, preserved from the old readers):

        - pydaq: ``name`` is the scope channel name (e.g. ``'C2'``). ``shots`` may
          be ``None`` (all shots in the scope group), an int (one shot), or a
          sequence/slice (that subset). Returns ``(stack, dt, t0)`` where ``stack``
          is ``(nshot, nsamples)`` float64 (one row per shot; NaN rows for
          unreadable shots), via ``scope.read_hdf5_scope_channel_shots``.
        - bapsflib: ``name`` must be ``(board_num, chan_num)``; ``shots`` maps to
          the digitizer ``index`` argument (None = all). Returns ``(data, tarr)``
          from ``read_data`` (``data`` is a bapsflib structured array).
        - legacy: ``name`` is the channel number; ``shots`` is not supported (the
          2018 layout stores a single trace per channel). Returns the channel
          ndarray from ``read_channel_data``.
        """
        if self._backend == "pydaq":
            with self._open_pydaq_scope(scope_name) as (f, scope_name, scope):
                shot_numbers = self._pydaq_shot_list(f, scope_name, shots, scope)
                return scope.read_hdf5_scope_channel_shots(
                    f, scope_name, name, shot_numbers, **kwargs
                )

        if self._backend == "bapsflib":
            bapsflib_daq = self._bapsflib_daq()
            try:
                board_num, chan_num = name
            except (TypeError, ValueError):
                raise TypeError(
                    "bapsflib channel() expects name=(board_num, chan_num); "
                    f"got {name!r}."
                )
            return self._with_bapsflib_file(
                lambda f: bapsflib_daq.read_data(
                    f, board_num, chan_num, index_arr=shots, **kwargs
                )
            )

        # legacy
        if shots is not None:
            raise NotImplementedError(
                "shot selection is not available for legacy 2018 files (one trace "
                "per channel); call channel(name) without shots."
            )
        return self._legacy_2018().read_channel_data(self.path, name, **kwargs)

    def iter_shots(self, name, shots=None, scope_name=None):
        """Yield one shot of ``name`` at a time (LAPD_DAQ pydaq files only).

        For the process-and-discard pattern on large runs: each iteration yields
        ``(shot_number, data, dt, t0)`` for one shot via
        ``scope.read_hdf5_scope_data``, without loading the whole run. The HDF5
        handle is held only for the duration of iteration and closed when the
        generator is exhausted or closed.

        Raises ``NotImplementedError`` for bapsflib/legacy files.
        """
        self._require_pydaq("iter_shots()")
        with self._open_pydaq_scope(scope_name) as (f, scope_name, scope):
            for s in self._pydaq_shot_list(f, scope_name, shots, scope):
                data, dt, t0 = scope.read_hdf5_scope_data(f, scope_name, name, s)
                yield s, data, dt, t0

    @staticmethod
    def _pydaq_shot_list(f, scope_name, shots, scope):
        """Normalize ``shots`` (None/int/slice/sequence) to a list of shot numbers."""
        if shots is None:
            return list(scope.scope_shot_numbers(f[scope_name]))
        if isinstance(shots, int):
            return [shots]
        if isinstance(shots, slice):
            return list(scope.scope_shot_numbers(f[scope_name]))[shots]
        return list(shots)

    # ----------------------------------------------------------------------- #
    # time array
    # ----------------------------------------------------------------------- #
    def time_array(self, scope_name=None):
        """Return the time array for the run.

        - pydaq: ``scope.read_hdf5_scope_tarr(f, scope_name)``
        - bapsflib / legacy: not derivable without a channel read; raises
          ``NotImplementedError`` (use ``channel()``, which returns the time array
          or the scaling needed to build it, preserving the old readers' behavior).
        """
        if self._backend == "pydaq":
            with self._open_pydaq_scope(scope_name) as (f, scope_name, scope):
                return scope.read_hdf5_scope_tarr(f, scope_name)
        raise NotImplementedError(
            f"time_array() requires a channel read for {self._backend!r} files; "
            f"the time array comes back from channel() (bapsflib: (data, tarr); "
            f"legacy: build from get_header())."
        )
