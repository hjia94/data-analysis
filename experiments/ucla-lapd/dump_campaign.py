"""Phase-1 extractor for run-log generation: campaign metadata -> JSONL.

Companion to RUN_LOG_PROMPT.md. One line of JSON per run, written to stdout or
``-o``. Observes and reports; it never interprets. Classifying probe motion,
joining probes to channels, and judging discrepancies are the agent's job in
phase 2 -- this script only supplies the raw material, because every one of
those decisions has campaign-specific exceptions that code gets wrong silently.

Three tiers of data, and only the first two are ever read here:
  metadata  -- attrs, group names, config text, dataset shapes   (emitted whole)
  positions -- per-motion-group shot ranges and grids            (emitted summarized)
  waveforms -- per-shot signal arrays                            (NEVER read)

Reading a waveform would make the pass O(campaign size) instead of O(runs); the
timeout that motivated this split came from bulk that had no business in a
metadata scan.

Unknown structure is recorded as ``layout_unknown`` / ``"__unreadable__"``, never
as an empty field. An agent reads absence as fact, so a silent null corrupts the
run log invisibly -- a loud marker does not.

Usage:
    python dump_campaign.py <DATA_DIR> [-o runs.jsonl]
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import h5py

from data_analysis.io.lapd_hdf5 import detect_backend  # not re-exported by io/
# The port parse and the [channels] parse are the SAME operation the analysis
# path joins on (data_analysis.io.probe_map). Importing them rather than keeping
# a copy is what stops this log and the analysis from disagreeing about which
# channel belongs to which probe -- the one thing the log exists to state.
from data_analysis.io.probe_map import config_text, parse_config_channels, ports_in

# Bump when a field's meaning changes, so a regenerated log that differs can be
# traced to the extractor rather than to the data.
# v2: parse_channels reads the whole [channels] section as `<prefix>_C<n>`
# instead of requiring a literal `_scope_` infix, which matched no channel in
# configs that write `LPScope_C1` (the pre-v2 join saw zero channel ports).
# v3: the port/[channels] parsers moved to data_analysis.io.probe_map; same
# output, one implementation shared with the analysis path.
SCHEMA_VERSION = 3

# `source_code` embeds a whole Python source file; `description` is long free
# text the agent pulls per-run when it actually needs it. Both blow the budget
# of a bulk scan.
SKIP_ATTRS = {"source_code", "description"}
ATTR_MAXLEN = 400


def attrs_of(obj):
    """Attributes, minus the oversized ones, truncated, JSON-safe."""
    out = {}
    for k, v in obj.attrs.items():
        if k in SKIP_ATTRS:
            continue
        s = str(v)
        out[k] = s[:ATTR_MAXLEN] + "..." if len(s) > ATTR_MAXLEN else s
    return out


def parse_channels(cfg):
    """`{'<scope>:C1': 'description'}` from the config's [channels] section.

    :func:`~data_analysis.io.probe_map.parse_config_channels` keyed for JSON:
    the tuple key it returns is not serializable, and the `'<scope>:C1'` spelling
    is what the emitted JSONL and the run log use.
    """
    return {f"{prefix}:{chan}": desc
            for (prefix, chan), desc in parse_config_channels(cfg).items()}


def summarize_group(g):
    """Motion summary for one group. Shot ranges, grid, and null-row accounting.

    ``shot_num == 0`` is the pydaq padding convention, but it is REPORTED, not
    assumed: ``null_pos_collides`` says whether the null coordinate (0,0) is also
    a real measured position. Where it collides, the convention is unsafe for
    that run and the agent must find another discriminator.
    """
    pa = g.get("positions_array")
    su = g.get("positions_setup_array")
    out = {"planned_positions": None if su is None else int(len(su))}

    if pa is None:
        out["error"] = "no positions_array"
        return out

    a = pa[:]
    names = a.dtype.names or ()
    if "shot_num" not in names:
        out["error"] = f"positions_array has no shot_num; fields={list(names)}"
        return out

    sn = a["shot_num"]
    live = sn != 0
    n_live = int(live.sum())
    out.update(rows=int(len(sn)), live_rows=n_live, null_rows=int((~live).sum()))

    if n_live:
        out["live_shot_range"] = [int(sn[live].min()), int(sn[live].max())]
        if "x" in names and "y" in names:
            xy = {(float(x), float(y)) for x, y in zip(a["x"][live], a["y"][live])}
            out["unique_positions"] = len(xy)
            out["shots_per_position"] = round(n_live / len(xy), 3)
            out["null_pos_collides"] = (0.0, 0.0) in xy
    else:
        out["live_shot_range"] = None
    return out


def pair_relations(groups):
    """Pairwise live-shot-range relation for every motion-group pair.

    'disjoint' vs 'overlap' vs 'identical' is what separates sequential from
    simultaneous probe motion. Computed rather than eyeballed -- reading two
    ranges and judging by hand is where this goes wrong.
    """
    rels = []
    named = [(n, s.get("live_shot_range")) for n, s in groups.items()]
    for i, (n1, r1) in enumerate(named):
        for n2, r2 in named[i + 1:]:
            if not r1 or not r2:
                rel = "unknown"
            elif r1 == r2:
                rel = "identical"
            elif r1[1] < r2[0] or r2[1] < r1[0]:
                rel = "disjoint"
            else:
                rel = "overlap"
            rels.append({"groups": [n1, n2], "relation": rel,
                         "ranges": [r1, r2]})
    return rels


def scope_summary(f):
    """Per scope group: shot count, and which channels exist on shot 1.

    Channel presence on disk is ground truth for what was acquired; the config's
    [channels] is only intent. Shot 1 stands in for all shots -- verifying every
    shot would defeat the point of a metadata-only pass.
    """
    out = {}
    for name, node in f.items():
        if not isinstance(node, h5py.Group):
            continue
        shots = [k for k in node if k.startswith("shot_")]
        if not shots:
            continue
        first = node.get(sorted(shots, key=lambda s: int(s.split("_")[1]))[0])
        out[name] = {
            "n_shot_groups": len(shots),
            "attrs": attrs_of(node),
            "channels_on_disk": sorted(
                k[:-5] for k in (first or {}) if k.endswith("_data")),
        }
    return out


def join_material(motion_groups, channels):
    """Ports parsed from both sides, plus what failed to join.

    Deliberately does NOT decide the mapping. A truncated port in a channel
    description and two probes sharing one port both need judgement, and both
    are invisible unless the unmatched sets are reported: an unmatched probe
    appearing alongside an unmatched channel in the same run is the signature
    of a typo, not of a missing probe.
    """
    mg_ports = {n: ports_in(n) for n in motion_groups}
    ch_ports = {k: ports_in(v) for k, v in channels.items()}
    moving = {p for ps in mg_ports.values() for p in ps}
    # A port claimed by two probes joins every one of its channels to both, so
    # the unmatched sets stay empty and the ambiguity is otherwise invisible.
    shared = sorted({p for p in moving
                     if sum(p in ps for ps in mg_ports.values()) > 1})
    return {
        "motion_group_ports": mg_ports,
        "channel_ports": ch_ports,
        "contested_ports": shared,
        "unmatched_motion_groups": sorted(
            n for n, ps in mg_ports.items()
            if not any(p in {q for k, qs in ch_ports.items() for q in qs}
                       for p in ps)),
        "unmatched_channels": sorted(
            k for k, ps in ch_ports.items() if ps and not (set(ps) & moving)),
        "motion_groups_without_port": sorted(
            n for n, ps in mg_ports.items() if not ps),
        "channels_without_port": sorted(
            k for k, ps in ch_ports.items() if not ps),
    }


def extract(path):
    """One run -> one JSON-safe dict. Never raises; failures become fields."""
    rec = {"schema_version": SCHEMA_VERSION,
           "file": os.path.basename(path),
           "size_bytes": os.path.getsize(path),
           "mtime": os.path.getmtime(path)}
    try:
        rec["backend"] = detect_backend(path)
    except Exception as e:                      # unrecognized layout: say so
        rec["backend"] = "layout_unknown"
        rec["backend_error"] = str(e)[:ATTR_MAXLEN]

    try:
        with h5py.File(path, "r") as f:
            rec["root_attrs"] = attrs_of(f)
            rec["top_level"] = sorted(f.keys())
            cfg = config_text(f)
            rec["config_text"] = cfg
            channels = parse_channels(cfg)
            rec["config_channels"] = channels
            rec["scopes"] = scope_summary(f)

            pos = f.get("Control/Positions")
            if pos is None:
                rec["motion_groups"] = {}
                rec["motion_note"] = "no Control/Positions group"
            else:
                rec["motion_groups"] = {n: summarize_group(pos[n]) for n in pos}
                rec["motion_pairs"] = pair_relations(rec["motion_groups"])

            rec["join"] = join_material(rec["motion_groups"], channels)
    except Exception as e:
        rec["read_error"] = f"{type(e).__name__}: {e}"[:ATTR_MAXLEN]
        rec.setdefault("motion_groups", "__unreadable__")
    return rec


def coverage(records):
    """How many runs produced each field -- the first thing phase 2 should read.

    A field present in 31 of 37 runs is either a real campaign change or a gap in
    this script, and the difference matters. Cheap to compute, and it turns
    "did I miss something" into a check rather than a worry.
    """
    counts = {}
    for r in records:
        for k, v in r.items():
            if v not in (None, {}, [], ""):
                counts[k] = counts.get(k, 0) + 1
    return {"n_runs": len(records),
            "field_coverage": dict(sorted(counts.items())),
            "backends": sorted({r.get("backend", "?") for r in records}),
            "runs_with_read_error": [r["file"] for r in records
                                     if "read_error" in r],
            "runs_with_unknown_layout": [r["file"] for r in records
                                         if r.get("backend") == "layout_unknown"]}


def git_hash():
    try:
        return subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)),
             "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("-o", "--out", help="output JSONL (default: stdout)")
    ap.add_argument("--glob", default="*.hdf5")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, args.glob)))
    if not paths:
        sys.exit(f"no files matching {args.glob!r} in {args.data_dir}")

    records = [extract(p) for p in paths]
    summary = {"summary": coverage(records),
               "schema_version": SCHEMA_VERSION,
               "extractor_git": git_hash(),
               "data_dir": os.path.abspath(args.data_dir)}

    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    if out is sys.stdout:
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        for r in records:
            out.write(json.dumps(r, default=str) + "\n")
        out.write(json.dumps(summary, default=str) + "\n")
    finally:
        if args.out:
            out.close()
            print(json.dumps(summary["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
