"""Reader for interferometer traces merged into LAPD_DAQ (pydaq) datarun files.

``interf_merge_lapd_daq.py`` (bapsf_interferometer repo) merges TWO interferometer
traces per channel into a datarun HDF5 -- the trace nearest the run's FIRST shot
and the one nearest its LAST shot -- under::

    diagnostics/interferometer/<phase_pNN>/<shot_number>   phase [rad]
    diagnostics/interferometer/time_array                  time [ms] (LeCroy channels)
    diagnostics/interferometer/time_array_p40              time [ms] (phase_p40 / Rigol)

Each ``phase_*`` group carries the attrs ``calibration factor (m^-3/rad)`` (line-
averaged density = factor x phase, assuming a 40 cm plasma length) and
``Microwave frequency (Hz)``.  ``phase_p40`` shots may be zero-filled placeholders
flagged with a ``rigol_missing`` attr; those are skipped here.
"""

from dataclasses import dataclass, field

import h5py
import numpy as np

_INTERF_GROUP = "diagnostics/interferometer"


@dataclass
class InterferometerChannel:
    """One merged interferometer channel: kept traces + calibration metadata."""

    name: str                 # e.g. "phase_p29"
    cal: float                # calibration factor [m^-3 / rad]
    f_uwave: float            # microwave frequency [Hz] (nan if absent)
    t_ms: np.ndarray          # (nt,) time axis [ms], as stored
    phase: np.ndarray         # (nshots_kept, nt) phase [rad]
    shots: list = field(default_factory=list)    # kept shot numbers, sorted
    when: list = field(default_factory=list)     # local acquisition time per kept shot
    skipped: dict = field(default_factory=dict)  # shot -> reason for dropped shots

    @property
    def ne_line_cm3(self):
        """(nshots_kept, nt) line-averaged density [cm^-3] (40 cm plasma length)."""
        return self.phase * (self.cal / 1e6)

    def ne_line_avg_cm3(self):
        """(nt,) line-averaged density [cm^-3], averaged over the kept shots."""
        return self.ne_line_cm3.mean(axis=0)


def read_interferometer(fn, channels=None):
    """Read merged interferometer traces from a pydaq datarun HDF5.

    ``channels=None`` reads every ``phase_*`` group; otherwise only the named
    ones.  Returns ``{name: InterferometerChannel}``.  ``rigol_missing``
    placeholder shots are dropped and recorded in ``.skipped``; a channel whose
    shots are all missing is returned with an empty ``phase`` array.
    """
    out = {}
    with h5py.File(fn, "r") as f:
        interf = f.get(_INTERF_GROUP)
        if interf is None:
            raise KeyError(
                f"No {_INTERF_GROUP} group in {fn} -- "
                "merge with interf_merge_lapd_daq.py first.")

        if channels is None:
            channels = sorted(k for k in interf if k.startswith("phase_"))

        for name in channels:
            g = interf[name]
            # Per-channel time group when one exists (e.g. the Rigol-scoped
            # phase_p40 -> time_array_p40), else the shared time_array.
            suffix = name.split("_", 1)[1]
            t_grp = interf.get(f"time_array_{suffix}") or interf["time_array"]

            ch = InterferometerChannel(
                name=name,
                cal=float(g.attrs["calibration factor (m^-3/rad)"]),
                f_uwave=float(g.attrs.get("Microwave frequency (Hz)", np.nan)),
                t_ms=np.empty(0),
                phase=np.empty((0, 0)),
            )
            traces = []
            for shot in sorted(g, key=int):
                ds = g[shot]
                if ds.attrs.get("rigol_missing", False):
                    ch.skipped[int(shot)] = str(
                        ds.attrs.get("rigol_missing_reason", "rigol missing"))
                    continue
                if not traces:
                    ch.t_ms = t_grp[shot][:]
                traces.append(ds[:])
                ch.shots.append(int(shot))
                ch.when.append(str(ds.attrs.get("interferometer time (local)", "?")))
            if traces:
                ch.phase = np.stack(traces)
            out[name] = ch
    return out
