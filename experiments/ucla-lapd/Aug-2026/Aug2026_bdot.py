"""Aug-2026 Alfvén zonal-flow campaign: reduce a B-dot run to per-shot wave phasors.

Two trombone antennas (south, north) launch a shear Alfvén wave in ~188 kHz
bursts; shots cycle A (south) / B (north) / AB at each position. For every shot
this reads probe P28's three B-dot channels and both antenna currents, then

* integrates dB/dt with :func:`data_analysis.plasma.bdot.integrate_bdot` under
  all three baselines (``pretrigger``, ``smooth``, ``periodic`` on the burst);
* takes 188 kHz phasors (:func:`data_analysis.plasma.fields.phasor`) over the
  steady part of the burst, phase-referenced to the record start;
* keeps per-cycle wave power (:func:`~data_analysis.plasma.fields.cycle_means`)
  and drift / noise-floor diagnostics.

Burst timing is found from the antenna currents, and the drive frequency from a
phasor scan, so the script runs unchanged on any B-dot run of the campaign
(03, 05, 17, 18, 21, 29, 30, 35).

**Baseline choice.** Integrated B in these runs is dominated by a ~1.5 kHz
plasma mode and small in-burst amplifier offset shifts (integration weights a
component by 1/f, so the slow content is 10-100x the wave). The wave is the
phasor (identical under all three baselines to <= 2e-5 on run 30) or the
``smooth``-baseline waveform; ``pretrigger`` B keeps the slow content.

**Units.** The C11/C13 effective areas are not applied yet, so B is per unit
coil area (``inverse_area_matrix(1.0, GAIN)``): shapes, phases, ratios and
curl/div structure are real, the absolute scale is not. Replace ``AREA`` with
the calibration (a 3x3 matrix is accepted) when it is available.

    .venv/bin/python experiments/ucla-lapd/Aug-2026/Aug2026_bdot.py 30
    .venv/bin/python experiments/ucla-lapd/Aug-2026/Aug2026_bdot.py 35 --workers 16

Output: ``<out-dir>/<run>-bdot-phasors.npz`` (default out-dir
``~/data-analysis-output/Aug2026``, outside the repo).
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from data_analysis.io import open_lapd
from data_analysis.plasma import bdot, fields

DATA_DIR = Path("/data/BAPSF_Data/Alfven_zonal_flow/aug2026")
OUT_DIR = Path.home() / "data-analysis-output" / "Aug2026"
SCOPE = "bdot_scope"
CH_B = ["C1", "C2", "C3"]          # P28 Bx, By, Bz behind X10 amplifiers
CH_I = ["C7", "C8"]                # antenna current south, north
I_PER_V = 20.0                     # A/V on the antenna current monitors
GAIN = 10.0
AREA = 1.0                         # m^2 -- placeholder until the C11 calibration is applied
F_SCAN = (186e3, 190e3)            # drive-frequency search band, Hz


def run_file(run):
    hits = sorted(DATA_DIR.glob(f"{int(run):02d}-*.hdf5"))
    if len(hits) != 1:
        raise FileNotFoundError(f"run {run}: expected one file in {DATA_DIR}, found {hits}")
    return hits[0]


def burst_windows(I, t, period_s=1 / 188e3):
    """Sample windows from antenna currents ``I`` (nshot, nt) covering A, B and AB shots.

    The burst is where the cycle-RMS current exceeds half its maximum. Returns
    slices: ``pre`` (quiet, before), ``burst`` (driven), ``steady`` (driven,
    past the first ms of turn-on), ``post`` (quiet, after; may be empty).
    """
    dt = t[1] - t[0]
    n = max(int(round(period_s / dt)), 1)
    p = ((I - I[:, :n].mean(1, keepdims=True)) ** 2).max(0)    # loudest antenna at each sample
    env = np.sqrt(np.convolve(p, np.ones(n) / n, mode="same"))  # cycle RMS
    on = np.flatnonzero(env > 0.5 * env.max())
    i0, i1 = on[0], on[-1]
    k = lambda s: int(round(s / dt))
    pre = slice(k(0.05e-3), max(i0 - k(0.2e-3), k(0.05e-3) + 1))
    burst = slice(i0 + k(0.2e-3), i1 - k(0.1e-3))
    steady = slice(min(i0 + k(1e-3), burst.stop - k(0.5e-3)), i1 - k(0.3e-3))
    post = slice(min(i1 + k(0.5e-3), len(t)), max(len(t) - k(0.1e-3), min(i1 + k(0.5e-3), len(t))))
    if pre.stop - pre.start < 10 * n:
        raise RuntimeError(f"less than 10 drive periods before the burst ({(pre.stop - pre.start) * dt * 1e3:.2f} ms)")
    return dict(pre=pre, burst=burst, steady=steady, post=post)


def drive_frequency(I, dt, steady):
    f = np.linspace(*F_SCAN, 801)
    return float(f[np.argmax([abs(fields.phasor(I[steady], dt, fi)) for fi in f])])


def reduce_chunk(args):
    path, shots, freq, w = args
    run = open_lapd(path)
    V = {}
    for ch in CH_B + CH_I:
        V[ch], t = run.channel(ch, shots=shots, scope_name=SCOPE)
    dt = t[1] - t[0]
    period = 1 / (freq * dt)
    vb = np.stack([V[c] for c in CH_B])                          # (3, nshot, nt)
    bd = bdot.voltage_to_bdot(vb, bdot.inverse_area_matrix(AREA, GAIN))
    B = bdot.integrate_bdot(bd, dt, "pretrigger", pretrigger=w["pre"])
    Braw = bdot.integrate_bdot(bd, dt, None)
    Bsm = bdot.integrate_bdot(bd, dt, "smooth", period=period, smooth_periods=4)
    Bper = bdot.integrate_bdot(bd[..., w["burst"]], dt, "periodic", period=period)

    st = w["steady"]

    def ph(x, sl=st, t0=st.start):                               # phase origin = record start
        return fields.phasor(x[..., sl], dt, freq) * np.exp(-2j * np.pi * freq * t0 * dt)

    Bph = ph(B)
    off = w["burst"].start
    a = np.abs(Bph)
    post = w["post"]
    return dict(
        shots=np.asarray(shots),
        Iamp=np.stack([np.sqrt(2) * V[c][:, st].std(-1) * I_PER_V for c in CH_I]),   # A
        Iph=np.stack([ph(V[c]) * I_PER_V for c in CH_I]),                            # A
        Bph=Bph, Bdph=ph(bd), Bph_sm=ph(Bsm),
        Bph_per=ph(Bper, slice(st.start - off, st.stop - off), st.start),
        end_raw=np.abs(Braw[..., -1]) / a, end_pre=np.abs(B[..., -1]) / a,
        pre_sm=np.sqrt(2) * Bsm[..., w["pre"]].std(-1) / a,
        post_sm=(np.sqrt(2) * Bsm[..., post].std(-1) / a) if post.stop > post.start
        else np.full(a.shape, np.nan),
        cyc=fields.cycle_means(Bsm[..., w["burst"]] ** 2, period),               # (3, nshot, ncyc)
        offset_V=vb[..., w["pre"]].mean(-1),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=int)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=36, help="shots per worker task")
    args = ap.parse_args()

    t0 = time.time()
    path = run_file(args.run)
    run = open_lapd(path)
    _setup, pos, _npos, _nshot = run.positions()                 # pydaq: (setup, pos_array, npos, nshot)
    pos = pos[pos["shot_num"] != 0]
    shots = [int(s) for s in pos["shot_num"]]
    I = np.stack([run.channel(c, shots=shots[:3], scope_name=SCOPE)[0] for c in CH_I])
    t = run.time_array(CH_I[0], scope_name=SCOPE)
    I = I.reshape(-1, I.shape[-1])                               # (2*3, nt): A, B and AB shots
    w = burst_windows(I, t)
    freq = drive_frequency(I[np.argmax(I.std(-1))], t[1] - t[0], w["steady"])
    ms = {k: f"{t[s.start]*1e3:.2f}-{t[s.stop-1]*1e3:.2f} ms" for k, s in w.items() if s.stop > s.start}
    print(f"{path.name}\n  {len(shots)} shots, dt {(t[1]-t[0])*1e9:.0f} ns, drive {freq/1e3:.3f} kHz\n  windows {ms}",
          flush=True)

    chunks = [shots[i:i + args.chunk] for i in range(0, len(shots), args.chunk)]
    parts = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, r in enumerate(ex.map(reduce_chunk, [(path, c, freq, w) for c in chunks])):
            parts.append(r)
            print(f"  chunk {k + 1}/{len(chunks)}  {time.time() - t0:.0f}s", flush=True)
    res = {k: np.concatenate([p[k] for p in parts], axis=0 if k == "shots" else 1) for k in parts[0]}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.run:02d}-bdot-phasors.npz"
    np.savez(out, source=str(path), freq=freq, dt=t[1] - t[0], t_burst0=t[w["burst"].start],
             area=AREA, gain=GAIN, pos_x=pos["x"], pos_y=pos["y"], pos_shot=pos["shot_num"], **res)
    print(f"saved {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
