"""Group the Jun-2026 LAPD HDF5 runs by their gas-puff setting.

The operator records the puff in the (free-text) ``Plasma condition`` bullet of
each run's ``description`` attribute, e.g.::

    Helium backside pressure 40 Psi, Puff voltage 75V for 24ms West+East

The puff voltage (V) and duration (ms) are pulled out via the shared
``data_analysis.io.gas_puff`` helper. Runs that share a ``(puff_v, puff_t)``
pair are collected into one group.

Writes ``gas_puff_groups.npy`` next to the data: a dict keyed by a human label
(``"75V-24ms"``) -> ``{"puff_v": <volts>, "puff_t": <ms>, "files": [names...]}``.
Load with ``np.load(path, allow_pickle=True).item()``.
"""

import glob
import os

import numpy as np

from data_analysis.io import gas_puff

DATA_DIR = r"D:\data\LAPD\jun2026-jia"
OUT_NAME = "gas_puff_groups.npy"


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    groups = {}
    ungrouped = []

    for path in files:
        name = os.path.basename(path)
        try:
            puff = gas_puff(path)
        except Exception as exc:  # non-pydaq / unreadable -> skip, but report
            print(f"  ! {name}: {exc}")
            puff = None
        if puff is None:
            ungrouped.append(name)
            continue
        puff_v, puff_t = puff
        # ``:g`` drops a trailing ``.0`` (24.0 -> '24', 24.5 -> '24.5').
        label = f"{puff_v:g}V-{puff_t:g}ms"
        grp = groups.setdefault(
            label, {"puff_v": puff_v, "puff_t": puff_t, "files": []}
        )
        grp["files"].append(name)

    # Stable ordering: by voltage then duration.
    groups = dict(
        sorted(groups.items(), key=lambda kv: (kv[1]["puff_v"], kv[1]["puff_t"]))
    )

    out_path = os.path.join(DATA_DIR, OUT_NAME)
    np.save(out_path, groups, allow_pickle=True)

    print(f"\nGrouped {sum(len(g['files']) for g in groups.values())} "
          f"file(s) into {len(groups)} gas-puff setting(s):")
    for label, g in groups.items():
        print(f"\n  [{label}]  puff_v={g['puff_v']} V, puff_t={g['puff_t']} ms"
              f"  ({len(g['files'])} files)")
        for n in g["files"]:
            print(f"      {n}")
    if ungrouped:
        print(f"\n  No puff setting found in {len(ungrouped)} file(s):")
        for n in ungrouped:
            print(f"      {n}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
