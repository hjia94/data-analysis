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

Output normalization (Step 3b)
------------------------------
``channel()`` and ``time_array()`` return a single schema regardless of
provenance, so cross-provenance analysis code no longer branches on
``.backend``: ``channel()`` -> ``(stack, tarr)`` with ``stack`` a
``(nshot, nsamples)`` float array and ``tarr`` the 1-D time axis; ``time_array()``
-> ``tarr`` for every backend. Each backend's native reader output is adapted at
the seam (the readers themselves are unchanged from Step 3 -- wrap, don't
rewrite). ``positions()`` and ``info()`` remain backend-native: the three
position layouts share no clean common schema, so a forced unification would be
lossy; see ``positions()``.
"""

import os
from contextlib import contextmanager

import h5py
import numpy as np

# Backend modules and scope_io are imported lazily inside the methods that need
# them (see _backend_module / _open_pydaq_scope): bapsflib_daq pulls in bapsflib
# + matplotlib, so importing them eagerly would load those heavy deps even when
# opening a pydaq/legacy file. Matches the lazy-import convention in io/scope.py.

__all__ = ["open_lapd", "LapdRun", "LapdSession", "detect_backend"]


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


def _read_bapsflib_positions(f, bapsflib_daq):
    """Probe motion from an open bapsflib ``lapd.File``: bmotion, else 6K.

    Shared by ``LapdRun.positions()`` (which opens/closes the file around it) and
    ``LapdSession.positions()`` (which already holds the file open).
    """
    result = bapsflib_daq.read_probe_motion_bmotion(f)
    if result is None:
        result = bapsflib_daq.read_probe_motion_6k(f)
    return result


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

    Construct via :func:`open_lapd`. ``channel()`` and ``time_array()`` return a
    normalized schema for every backend (see the module docstring);
    ``positions()`` and ``info()`` return each backend's native output (documented
    per method). Where a backend lacks a concept the method raises
    ``NotImplementedError``. No HDF5 handle is held between calls.
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

    # ----------------------------------------------------------------------- #
    # held-handle session (bapsflib): one open file, many reads
    # ----------------------------------------------------------------------- #
    @contextmanager
    def session(self):
        """Yield a :class:`LapdSession` holding one open handle for many reads.

        This is the single bapsflib open/close path: the per-call ``LapdRun``
        methods (``info``/``digitizer_config``/``positions``/``channel``) each open
        a fresh one-read session and close it (the no-lingering-handle rule). Some
        experiment routines instead open the file once and make several reads
        against that one handle -- read the digitizer config, the probe motion,
        then a handful of channels -- which is what calling ``session()`` directly
        gives them, without leaking a handle: the file is open only for the
        ``with`` body and closed on exit. ::

            with open_lapd(path).session() as sess:
                adc, digi = sess.digitizer_config()
                pos, *_ = sess.positions()
                data, tarr = sess.read_data(4, 5, index_arr=slice(0, n), adc=adc)

        Only the bapsflib backend holds a handle (its ``lapd.File`` is the
        expensive open); pydaq/legacy reads are per-path and cheap, so calling
        ``session()`` on those backends raises ``NotImplementedError`` -- use the
        :class:`LapdRun` methods directly.
        """
        if self._backend != "bapsflib":
            raise NotImplementedError(
                f"session() holds one open bapsflib handle; the {self._backend!r} "
                "backend reads per-path, so call the LapdRun methods directly."
            )
        from bapsflib import lapd
        f = lapd.File(self.path)
        try:
            yield LapdSession(f, self._bapsflib_daq())
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
            with self.session() as sess:
                return sess.info()
        if self._backend == "pydaq":
            return self._pydaq().print_info(self.path)
        return self._legacy_2018().print_data_objects(self.path)

    def digitizer_config(self):
        """Read the digitizer configuration (bapsflib files only).

        Delegates to ``read_digitizer_config`` -> ``(adc, digi_dict)`` where
        ``adc`` is the ADC name (e.g. ``'SIS 3302'``) and ``digi_dict`` maps board
        number to the list of enabled channels. Pass the returned ``adc`` to
        :meth:`channel` (``adc=``) to read from that digitizer. Raises
        ``NotImplementedError`` for the pydaq/legacy backends, which have no SIS
        crate config.
        """
        if self._backend != "bapsflib":
            raise NotImplementedError(
                f"digitizer_config() reads a bapsflib SIS-crate config; this is a "
                f"{self._backend!r} file."
            )
        with self.session() as sess:
            return sess.digitizer_config()

    def scope_channels(self, scope_name):
        """Print channel descriptions for a scope group (pydaq files only).

        Delegates to ``print_scope_channels(path, scope_name)``. Raises
        ``NotImplementedError`` for bapsflib/legacy files.
        """
        self._require_pydaq("scope_channels()")
        return self._pydaq().print_scope_channels(self.path, scope_name)

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
            with self.session() as sess:
                return sess.positions()
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
    @staticmethod
    def _tarr(nsamples, dt, t0):
        """Build the uniform 1-D time axis from sample count + sampling (dt, t0)."""
        return np.arange(nsamples) * dt + t0

    def channel(self, name, shots=None, scope_name=None, **kwargs):
        """Read channel data, normalized to ``(stack, tarr)`` for every backend.

        ``stack`` is a ``(nshot, nsamples)`` float array (one row per shot) and
        ``tarr`` is the matching 1-D time axis. The backend reader output is
        adapted to this schema at the seam; ``name`` is still backend-specific:

        - pydaq: ``name`` is the scope channel name (e.g. ``'C2'``). ``shots`` may
          be ``None`` (all shots in the scope group), an int (one shot), or a
          sequence/slice (that subset). The reader returns ``(stack, dt, t0)``
          (``stack`` is ``(nshot, nsamples)`` float64, NaN rows for unreadable
          shots); ``tarr`` is built from ``dt``/``t0``.
        - bapsflib: ``name`` must be ``(board_num, chan_num)``; ``shots`` maps to
          the digitizer ``index`` argument (None = all). The reader returns
          ``(data, tarr)`` with ``data['signal']`` the ``(nshot, nsamples)`` stack.
        - legacy: ``name`` is the channel number; ``shots`` is not supported (the
          2018 layout stores one trace per channel). The single trace is returned
          as a 1-row stack with ``tarr`` built from the channel header.
        """
        if self._backend == "pydaq":
            with self._open_pydaq_scope(scope_name) as (f, scope_name, scope):
                shot_numbers = self._pydaq_shot_list(f, scope_name, shots, scope)
                stack, dt, t0 = scope.read_hdf5_scope_channel_shots(
                    f, scope_name, name, shot_numbers, **kwargs
                )
            if stack is None:
                return None, None
            return stack, self._tarr(stack.shape[1], dt, t0)

        if self._backend == "bapsflib":
            try:
                board_num, chan_num = name
            except (TypeError, ValueError):
                raise TypeError(
                    "bapsflib channel() expects name=(board_num, chan_num); "
                    f"got {name!r}."
                )
            with self.session() as sess:
                data, tarr = sess.read_data(
                    board_num, chan_num, index_arr=shots, **kwargs
                )
            return data["signal"], tarr

        # legacy
        if shots is not None:
            raise NotImplementedError(
                "shot selection is not available for legacy 2018 files (one trace "
                "per channel); call channel(name) without shots."
            )
        legacy = self._legacy_2018()
        trace = legacy.read_channel_data(self.path, name, **kwargs)
        if trace is None:
            return None, None
        stack = np.atleast_2d(trace)
        header = legacy.get_header(self.path)
        return stack, self._tarr(stack.shape[1], header.dt, header.t0)

    def iter_shots(self, name, shots=None, scope_name=None):
        """Yield one shot of ``name`` at a time (LAPD_DAQ pydaq files only).

        For the process-and-discard pattern on large runs: each iteration yields
        ``(shot_number, data, tarr)`` for one shot via ``scope.read_hdf5_scope_data``,
        without loading the whole run. ``data`` is the 1-D trace and ``tarr`` the
        matching time axis (built from the shot's ``dt``/``t0``), consistent with
        :meth:`channel`. The HDF5 handle is held only for the duration of
        iteration and closed when the generator is exhausted or closed.

        Raises ``NotImplementedError`` for bapsflib/legacy files.
        """
        self._require_pydaq("iter_shots()")
        tarr = None
        with self._open_pydaq_scope(scope_name) as (f, scope_name, scope):
            for s in self._pydaq_shot_list(f, scope_name, shots, scope):
                data, dt, t0 = scope.read_hdf5_scope_data(f, scope_name, name, s)
                # dt/t0/length are the same for every shot; build tarr once and
                # reuse it (only rebuild if a shot's length differs).
                if tarr is None or len(tarr) != len(data):
                    tarr = self._tarr(len(data), dt, t0)
                yield s, data, tarr

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
    def time_array(self, name=None, scope_name=None):
        """Return the 1-D time axis ``tarr`` for the run (uniform across backends).

        - pydaq: read directly from the scope group via
          ``scope.read_hdf5_scope_tarr`` -- ``name`` is not needed.
        - bapsflib / legacy: the axis length comes from a channel, so pass
          ``name`` (bapsflib ``(board_num, chan_num)``, legacy channel number).
          This reads a single shot via :meth:`channel` (not the whole run) and
          returns just its ``tarr``.
        """
        if self._backend == "pydaq":
            with self._open_pydaq_scope(scope_name) as (f, scope_name, scope):
                return scope.read_hdf5_scope_tarr(f, scope_name)
        if name is None:
            raise TypeError(
                f"time_array() needs a channel name for {self._backend!r} files "
                "(the axis length comes from a channel read); pass name= "
                "(bapsflib: (board_num, chan_num); legacy: channel number)."
            )
        # Read only the first shot: the time axis is shot-independent, so there's
        # no need to pull the whole (nshot, nsamples) stack just to discard it.
        # (legacy ignores shots -- one trace per channel -- so this is a no-op there.)
        shots = None if self._backend == "legacy" else 0
        _, tarr = self.channel(name, shots=shots)
        return tarr


# --------------------------------------------------------------------------- #
# held-handle session (bapsflib only)
# --------------------------------------------------------------------------- #
class LapdSession:
    """One open bapsflib ``lapd.File`` exposing the per-handle reads as methods.

    Obtained from :meth:`LapdRun.session`; valid only inside that ``with`` block
    (the handle is closed on exit). The methods mirror the bapsflib backend
    functions with the ``f`` argument dropped, and return each reader's **native**
    output unchanged (not the normalized ``(stack, tarr)`` of
    :meth:`LapdRun.channel`) so routines that open the file once and make several
    reads against that handle migrate with no behavior change::

        rh.read_data(f, board, chan, ...)  ->  sess.read_data(board, chan, ...)
    """

    def __init__(self, f, bapsflib_daq):
        self._f = f
        self._b = bapsflib_daq

    @property
    def file(self):
        """The held ``bapsflib.lapd.File`` -- escape hatch for raw bapsflib calls
        (e.g. ``sess.file.read_msi('Discharge')``) the wrapper methods don't cover.
        Valid only inside the ``session()`` ``with`` block."""
        return self._f

    def info(self):
        """``show_info(f)`` -- print a human-readable overview."""
        return self._b.show_info(self._f)

    def digitizer_config(self):
        """``read_digitizer_config(f)`` -> ``(adc, digi_dict)``."""
        return self._b.read_digitizer_config(self._f)

    def positions(self):
        """Probe motion via ``read_probe_motion_bmotion`` (falls back to
        ``read_probe_motion_6k`` if there is no bmotion group) ->
        ``(pos_array, xpos, ypos, zpos, npos, nshot)``."""
        return _read_bapsflib_positions(self._f, self._b)

    def positions_6k(self):
        """``read_probe_motion_6k(f)`` -- force the 6K reader (no bmotion fallback)."""
        return self._b.read_probe_motion_6k(self._f)

    def read_data(self, board_num, chan_num, index_arr=None, adc="SIS 3302",
                  control=None):
        """``read_data(f, ...)`` -> native ``(data, tarr)`` (``data['signal']`` is
        the ``(nshot, nsamples)`` stack). ``index_arr``/``adc``/``control`` are
        passed through unchanged."""
        return self._b.read_data(
            self._f, board_num, chan_num, index_arr=index_arr, adc=adc,
            control=control,
        )

    def datarun_sequence(self):
        """``unpack_datarun_sequence(f)`` -> ``(messages, statuses, timestamps)``."""
        return self._b.unpack_datarun_sequence(self._f)

    def magnetic_field(self):
        """``read_magnetic_field(f)`` -> ``(Bdata, port_ls)``."""
        return self._b.read_magnetic_field(self._f)

    def interferometer_old(self):
        """``read_interferometer_old(f)`` -> ``(int_arr, int_tarr, den_factor)``.

        CAVEAT (reorg item B4, unresolved): ``den_factor`` is not physics-verified
        -- treat it as provisional, not a calibrated density scaling. See the
        backend function's docstring.
        """
        return self._b.read_interferometer_old(self._f)
