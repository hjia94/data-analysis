"""Jun-2026 LAPD ion-saturation-current (Isat) fluctuation analysis.

Companion to ``Jun2026_IV.py``.  Where that module pulls the *swept* Langmuir
tips (complete I+V pairs) to extract Vp/Te/ne, this one reads a **fixed-bias
ion-saturation channel** and looks at how the Isat signal *fluctuates* in time
(raw trace + FFT), rather than fitting an IV curve.

Which channel is Isat is **not** guessed here -- you read the printed channel
descriptions (:func:`list_all_channels`) and the run's probe description
(:func:`print_run_description`) and name the scope + channel yourself.

Reading
-------
Same pydaq read path as ``Jun2026_IV``: channels are read by scope-channel name
via ``run.channel(name, scope_name=...)``.  Positions come from
``Jun2026_IV.read_lp_positions``.  Isat is a single fixed-bias current trace per
shot, so there is no sweep detection / reshape -- we read the raw per-shot
signal at one position and FFT it.

Current scaling reuses the ``RESISTOR`` / ``Aprobe`` knobs from ``Jun2026_IV``;
for fluctuation work the *shape* of the spectrum is what matters, not the
absolute scaling.  The IV pipeline's ``I_SIGN`` is irrelevant to Isat (a sign
flip doesn't change the fluctuation spectrum), so it is not applied here.
"""

import numpy as np
import h5py

from data_analysis.io import open_lapd
from data_analysis.io.scope_reader import read_scope_channel_descriptions

import Jun2026_IV as jiv

# Scope groups that may hold probe signals.  ``Configuration`` / ``Control`` are
# never scopes; everything else is a scope group.
_NON_SCOPE_GROUPS = {"Configuration", "Control"}


def list_all_channels(fn):
    """Print every scope group's channel descriptions and return them.

    Returns ``{scope_name: {chan: description}}`` for every scope group in the
    file (``Configuration`` / ``Control`` excluded).  Use the printout to decide
    which scope + channel is the Isat channel you want -- nothing is classified
    or guessed.
    """
    out = {}
    with h5py.File(fn, "r") as f:
        scope_groups = [g for g in f.keys() if g not in _NON_SCOPE_GROUPS]
        print("Scope groups and channel descriptions:")
        for scope_name in scope_groups:
            desc = read_scope_channel_descriptions(f, scope_name)
            if not desc:
                continue
            out[scope_name] = dict(desc)
            print(f"\n  scope '{scope_name}':")
            for chan in sorted(desc):
                print(f"    {chan}: {desc[chan]!r}")
    if not out:
        print("  (no scope groups with channel descriptions found)")
    return out


def print_run_description(fn):
    """Print the run's hand-written description (plasma / bias / probe settings).

    Reads the pydaq ``description`` attribute via ``open_lapd(fn).description()``
    and prints its raw text so the probe wiring / bias settings the operator
    wrote are visible alongside the channel list.  Returns the parsed
    ``RunDescription`` (or ``None`` if it can't be read).
    """
    try:
        desc = open_lapd(fn).description()
    except (OSError, ValueError, NotImplementedError, KeyError) as e:
        print(f"(could not read run description -- {e})")
        return None
    print("=== Run description ===")
    print(desc.raw)
    return desc


def get_isat_at_position(run, scope_name, chan, npos, nshot, pos_index):
    """Read the scaled Isat signal for ONE probe position.

    Reads only that position's ``nshot`` shots off disk (a positional shot slice,
    same ordering as ``Jun2026_IV._read_reshaped``).  Returns ``(tarr, Iarr)``
    where ``Iarr`` is ``(nshot, nsamples)`` scaled current density (via
    ``jiv.RESISTOR`` / ``jiv.Aprobe``).  No sign flip is applied -- it doesn't
    matter for Isat fluctuations.
    """
    shots = slice(pos_index * nshot, (pos_index + 1) * nshot)
    Istack, tarr = run.channel(chan, scope_name=scope_name, shots=shots)
    Iarr = Istack[:nshot] / (jiv.RESISTOR * jiv.Aprobe)
    return tarr, Iarr


def isat_fft(I, tarr, detrend=True):
    """Single-sided amplitude spectrum of an Isat trace.

    ``I`` is a 1-D trace (e.g. one shot, or the shot mean); ``tarr`` its time
    axis (seconds, uniform sampling).  Returns ``(freq, amp)`` with ``freq`` in
    Hz and ``amp`` the single-sided amplitude (DC bin dropped).  ``detrend``
    removes the mean first so the DC level doesn't swamp the fluctuation
    spectrum.
    """
    I = np.asarray(I, float)
    if detrend:
        I = I - np.mean(I)
    n = I.size
    dt = float(np.mean(np.diff(tarr)))
    freq = np.fft.rfftfreq(n, d=dt)
    amp = np.abs(np.fft.rfft(I)) * (2.0 / n)
    # Drop the DC bin (freq == 0); meaningless after detrend, distracting on a log plot.
    return freq[1:], amp[1:]
