"""Self-contained HTML slider pages from a validated *bundle* of frames.

A slider page is one ``.html`` file with **zero runtime dependencies**: no
server, no kernel, no plotly. Double-click it and scrub. It is *additive* --
the publication PNG stays whatever it was.

The bundle is the seam between analysis and plotting
--------------------------------------------------------------------------
Analysis code builds a plain ``dict`` (a *bundle*) and hands it here; this
module knows nothing about any diagnostic, and no analysis module imports it.
Every preliminary-data product decomposes into three independent parts, and the
bundle is exactly those three:

**geometry** -- the spatial layout, decided *solely by the position data*
(``plane`` -> heatmap panels, ``line`` -> profile panels). This is the same rule
:func:`data_analysis.viz.plot_utils.grid_frames` already applies.

**axis** -- the one scan dimension the slider scrubs, *typed and
self-describing* (``name``/``unit``/``values``). Frequency (a cross-spectral
estimator) and time (an IV sweep, an emissive trace) come from different
analysis schemas at different time scales; they are never interchangeable and
never share a generic dimension name. A bundle carries one axis, which a
group may override with its own when groups were recorded over different spans
(IV runs differ in both sweep count and duration).

**fields** -- named physical quantities, one frame cube each, *all* sharing that
one geometry and that one axis. Each field draws as its own panel, and every
panel moves together on the single slider.

A bundle may carry several **groups** of those fields -- one per channel, probe
pair, tip, or run -- selected by a dropdown on the page. A group is *not* a
fourth dimension: every group shares the one geometry, and differs only in which
signal was measured or when. That is what makes them comparable by switching
rather than by opening two files.

Groups share the *spatial* axes always -- they are measured on the same probe
line or plane, which is what makes switching a comparison rather than a change
of subject. The *scan* axis is per group: a group may carry its own ``axis``,
falling back to the bundle's. Where groups differ in length the slider keeps the
**index** across a switch and the readout shows the real value for the group now
selected.

A field draws one curve by default (``frames``). For ``line`` geometry it may
instead carry several named **traces** (``traces``), drawn on one panel against
one shared y-axis -- two probe tips of the same quantity, say. Overlaying is
what makes them comparable; that is also why traces are refused for ``plane``
geometry, where stacked heatmaps would simply hide one another.

Schema v1
--------------------------------------------------------------------------
::

    bundle = {
        "schema": 1,
        "title": str,                       # page heading
        "geometry": "plane" | "line",
        "axis": {"name": "frequency",       # THE scan axis (exactly one)
                 "unit": "kHz",
                 "values": (n_axis,) float},        # monotonic
        "x": {"label": "X position", "unit": "cm", "values": (nx,) float},
        "y": {...},                         # plane only; omitted for a line
        "fields": [                         # one or more, order = panel order
            {"name": "coherence",
             "unit": "",                    # "" for dimensionless
             "frames": (n_axis, ny, nx) | (n_axis, nx) float,
             "cmap": "viridis",             # any matplotlib colormap name
             "vmin": 0.0, "vmax": 1.0},     # both None -> page scale toggle
            # ... OR "traces" instead of "frames" (line geometry only):
            {"name": "Vp", "unit": "V", "cmap": "viridis",
             "vmin": None, "vmax": None,
             "traces": [{"name": "tip L", "frames": ndarray},
                        {"name": "tip R", "frames": ndarray}]},
        ],
        # ... OR, for a multi-channel page, "groups" instead of "fields":
        "groups": [                         # order = dropdown order
            {"name": "bdotscope-C1 vs lpscope-C3",   # dropdown entry
             "axis": {...},                 # optional; defaults to the above
             "fields": [...]},              # same field list as above
        ],
        "provenance": {"source": str, "params": {...}},   # rendered as a banner
        "warning": str | None,              # optional caveat banner
    }

Rules, all enforced by :func:`validate_bundle`:

* One axis per group (the bundle's, unless the group overrides it). Every
  field's ``frames.shape[0]`` equals that axis's ``values.size``.
* ``frames`` XOR ``traces`` on a field. ``traces`` needs ``line`` geometry, and
  every trace carries a full frame cube of the same shape.
* Every field shares the geometry: ``(n_axis, ny, nx)`` for a plane,
  ``(n_axis, nx)`` for a line, matching ``x`` (and ``y``).
* ``fields`` and ``groups`` are alternatives -- exactly one of the two. A bare
  ``fields`` list is the single-group case, and is normalized to one unnamed
  group internally, so the page has one code path rather than two.
* Groups need matching field *layouts*: same count, same names, same units,
  same trace names, in the same order. This is structural, not stylistic: the
  panel row (names, units, colormaps, fixed scales) is serialized **once** for
  the whole page and the page builds its canvases from it, so a group whose
  layout differed would be drawn under another group's captions and colorbars. Fields carrying a
  fixed ``vmin``/``vmax`` are additionally comparable across a switch;
  autoscaled ones are rescaled per group and are not.
* ``vmin``/``vmax`` both set -> that fixed physical scale is used (coherence
  ``0..1``, phase ``-180..180``). Both ``None`` -> the page's global/per-frame
  scale toggle applies to that field.
* NaN is legal anywhere in ``frames`` -- unvisited grid cells, masked values.
  It is serialized as JSON ``null`` and drawn grey.
* Frames arrive **prepared**: band-limited, windowed, gridded, any masking
  already applied. No analysis happens in JavaScript, ever.

Writing an adapter
--------------------------------------------------------------------------
One small function per diagnostic, living with that experiment (it is the only
code that knows the experiment's storage layout). It loads, slices the axis,
grids the frames with ``plot_utils.grid_frames``, builds the dict above, and calls
:func:`write_slider_html`. See ``Jun2026_slider.emit_xcorr_slider``.

Re-rendering without re-analysis
--------------------------------------------------------------------------
:func:`write_slider_html` also writes ``<stem>-bundle.npz`` beside the page.
After a template fix, re-render any saved bundle without touching the analysis::

    python -m data_analysis.viz.slider_html <stem>-bundle.npz

The bundle npz is a small *standalone* file of prepared frames. The experiment's
own data ``.npz`` is never read or written by this module.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from html import escape
from pathlib import Path

import matplotlib
import numpy as np

#: Bundle schema version. Bump only on an incompatible change; the renderer
#: refuses a bundle whose ``schema`` it does not recognise.
SCHEMA_VERSION = 1

#: Warn above this rendered size. 16 MB is also the Artifact publish ceiling,
#: so a page over it cannot be shared to a phone.
ARTIFACT_MAX_MB = 16.0

_TEMPLATE = Path(__file__).with_name("slider_template.html")

_GEOMETRIES = ("plane", "line")


# =========================================================================== #
#  Validation -- every error names the offending key
# =========================================================================== #

def bundle_groups(bundle):
    """The bundle's groups as a ``[{"name", "fields"}, ...]`` list.

    One accessor for both spellings: a bundle carrying a bare ``fields`` list is
    the single-group case and comes back as one unnamed group, so validation,
    serialization, and the round-trip all walk the same structure instead of
    branching on which key the adapter happened to use.
    """
    groups = bundle.get("groups")
    if groups is not None:
        return list(groups)
    return [{"name": "", "fields": bundle.get("fields")}]


def group_axis(bundle, group):
    """The axis a group scrubs: its own if it has one, else the bundle's.

    Groups that were recorded separately need not share a sweep -- the IV runs
    differ in both sweep count and time span -- so the axis is per group, with
    the bundle-level one as the shared default. The *spatial* axes stay
    bundle-level: every group is measured on the same probe line, which is what
    makes switching between them a comparison rather than a change of subject.
    """
    return group.get("axis") or bundle["axis"]


def field_traces(field):
    """A field's curves as a ``[{"name", "frames"}, ...]`` list.

    One accessor for both spellings, mirroring :func:`bundle_groups`: a field
    carrying a bare ``frames`` cube is the single-curve case and comes back as
    one unnamed trace, so validation, serialization and drawing walk the same
    structure whichever the adapter wrote.
    """
    traces = field.get("traces")
    if traces is not None:
        return list(traces)
    return [{"name": "", "frames": field.get("frames")}]


def validate_bundle(bundle):
    """Check a bundle against schema v1; raise ``ValueError`` naming the key.

    Returns the bundle unchanged so it can wrap a construction expression.
    Called automatically by :func:`write_slider_html`; call it directly in an
    adapter's tests to fail early with a precise message.
    """
    def _require(cond, msg):
        if not cond:
            raise ValueError(f"bundle: {msg}")

    _require(isinstance(bundle, dict), f"expected a dict, got {type(bundle).__name__}")
    _require(bundle.get("schema") == SCHEMA_VERSION,
             f"'schema' must be {SCHEMA_VERSION}, got {bundle.get('schema')!r}")

    geometry = bundle.get("geometry")
    _require(geometry in _GEOMETRIES,
             f"'geometry' must be one of {list(_GEOMETRIES)}, got {geometry!r}")

    def _check_axis(axis, where):
        """Validate one axis dict and return its length. Shared by the
        bundle-level axis and any per-group override, so both are held to the
        same rules instead of the override being the lenient path."""
        _require(isinstance(axis, dict), f"'{where}' must be a dict {{name, unit, values}}")
        for k in ("name", "unit", "values"):
            _require(k in axis, f"'{where}' is missing '{k}'")
        values = np.asarray(axis["values"], float)
        _require(values.ndim == 1 and values.size > 0,
                 f"'{where}.values' must be 1-D and non-empty, got shape {values.shape}")
        # The slider steps through these in order, so an unsorted axis would
        # scrub to frames that jump around the scan rather than sweeping it.
        _require(np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0),
                 f"'{where}.values' must be monotonic (it is the slider's order); "
                 f"got {values[:4]}...")
        return values.size

    n_axis = _check_axis(bundle.get("axis"), "axis")

    # Spatial axes: 'x' always; 'y' only (and always) for a plane.
    spatial = ["x", "y"] if geometry == "plane" else ["x"]
    sizes = {}
    for key in spatial:
        entry = bundle.get(key)
        _require(isinstance(entry, dict), f"'{key}' must be a dict {{label, unit, values}}")
        for k in ("label", "unit", "values"):
            _require(k in entry, f"'{key}' is missing '{k}'")
        values = np.asarray(entry["values"], float)
        _require(values.ndim == 1 and values.size > 0,
                 f"'{key}.values' must be 1-D and non-empty, got shape {values.shape}")
        sizes[key] = values.size
    if geometry == "line":
        _require("y" not in bundle, "'y' is meaningless for geometry 'line'; drop it")

    _require(("fields" in bundle) != ("groups" in bundle),
             "set exactly one of 'fields' (single channel) or 'groups' "
             "(several channels behind a dropdown), not both and not neither")
    if "groups" in bundle:
        _require(isinstance(bundle["groups"], (list, tuple)) and bundle["groups"],
                 "'groups' must be a non-empty list of {name, fields}")

    grouped = "groups" in bundle
    groups = bundle_groups(bundle)
    layout = None
    for g, group in enumerate(groups):
        # Errors name the key the adapter actually wrote, so a single-channel
        # bundle is not told about a 'groups' list it never set.
        stem = f"groups[{g}]" if grouped else "fields"
        prefix = f"{stem}.fields" if grouped else stem
        _require(isinstance(group, dict), f"{stem} must be a dict {{name, fields}}")
        # A name labels a dropdown entry, so it is required exactly when there
        # is a dropdown. A lone group has nothing to label -- and requiring one
        # there would reject a single-channel bundle read back from disk, where
        # save_bundle writes the grouped spelling regardless.
        if len(groups) > 1:
            _require(group.get("name"), f"{stem} is missing a non-empty 'name' "
                                        "(it labels the dropdown entry)")

        # A group may scrub its own axis (runs of differing length), so the
        # expected cube shape is per group, not per bundle.
        group_ax = group.get("axis")
        n = _check_axis(group_ax, f"{stem}.axis") if group_ax else n_axis
        want = (n, sizes["y"], sizes["x"]) if geometry == "plane" else (n, sizes["x"])

        fields = group.get("fields")
        _require(isinstance(fields, (list, tuple)) and len(fields) > 0,
                 f"{stem}: 'fields' must be a non-empty list")
        for i, field in enumerate(fields):
            where = f"{prefix}[{i}]"
            _require(isinstance(field, dict), f"{where} must be a dict")
            for k in ("name", "unit", "cmap"):
                _require(k in field, f"{where} is missing '{k}'")
            _require(("frames" in field) != ("traces" in field),
                     f"{where} ('{field['name']}') must set exactly one of "
                     "'frames' (one curve) or 'traces' (several curves on one "
                     "panel), not both and not neither")
            if "traces" in field:
                _require(geometry == "line",
                         f"{where} ('{field['name']}') uses 'traces', which "
                         "only draws for geometry 'line' -- overlaid heatmaps "
                         "would hide one another")
                _require(field["traces"],
                         f"{where}: 'traces' must be a non-empty list of "
                         "{name, frames}")
            traces = field_traces(field)
            # Both are properties of the field, not of the trace being checked.
            labelled = "traces" in field
            named = len(traces) > 1
            for t, trace in enumerate(traces):
                spot = f"{where}.traces[{t}]" if labelled else where
                _require(isinstance(trace, dict), f"{spot} must be a dict {{name, frames}}")
                # A name labels a legend entry, so it is required exactly when
                # there is a legend to label.
                if named:
                    _require(trace.get("name"),
                             f"{spot} is missing a non-empty 'name' "
                             "(it labels the trace in the legend)")
                # np.shape reads the shape without materializing/copying the cube.
                shape = np.shape(trace["frames"])
                _require(shape == want,
                         f"{spot} ('{field['name']}') frames have shape {shape}, "
                         f"expected {want} = (n_axis, "
                         + ("ny, nx)" if geometry == "plane" else "nx)"))
            # A colormap typo is the likeliest adapter mistake; catch it here rather
            # than as a KeyError from deep inside rendering.
            _require(field["cmap"] in matplotlib.colormaps,
                     f"{where} ('{field['name']}') has unknown cmap "
                     f"{field['cmap']!r}; see matplotlib.colormaps")
            vmin, vmax = field.get("vmin"), field.get("vmax")
            _require((vmin is None) == (vmax is None),
                     f"{where} ('{field['name']}') must set both 'vmin' and 'vmax' "
                     "or neither (neither -> the page's scale toggle)")
            _require(vmin is None or vmin < vmax,
                     f"{where} ('{field['name']}') needs vmin < vmax, "
                     f"got vmin={vmin}, vmax={vmax}")

        # _payload emits the panel row once, from the first group, and the
        # page builds its canvases from that row. A group whose layout differed
        # would therefore render under another group's captions and colorbars,
        # so the layouts must agree.
        signature = [(f["name"], f.get("unit", ""),
                      tuple(t.get("name", "") for t in field_traces(f)))
                     for f in fields]
        if layout is None:
            layout, layout_name = signature, group.get("name")
        else:
            _require(signature == layout,
                     f"{stem} ('{group.get('name')}') has fields {signature}, "
                     f"but '{layout_name}' has {layout}; every group must carry "
                     "the same fields, in the same order, so the dropdown swaps "
                     "data under fixed panels")

    return bundle


# =========================================================================== #
#  Serialization -- NaN -> null, colormaps -> LUTs
# =========================================================================== #

def _jsonable(value):
    """Recursively convert to JSON-safe Python, mapping non-finite floats to None.

    ``json.dumps`` emits a bare ``NaN`` literal, which ``JSON.parse`` rejects --
    so every NaN (unvisited grid cell, masked value) becomes ``null`` here and
    the page draws it grey.
    """
    if isinstance(value, np.ndarray):
        # Map non-finite entries at the array level, then convert once. Walking
        # the nested lists element-by-element instead costs ~13x more on a
        # full-size frame cube, for identical output.
        if value.dtype.kind == "f":
            boxed = value.astype(object)
            boxed[~np.isfinite(value)] = None
            return boxed.tolist()
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


#: Decimal places kept for floats in the page payload (the ``.npz`` beside it
#: keeps full precision). Frames are drawn into a 256-step colormap and printed
#: by the hover readout at far coarser resolution than a float64 carries, so
#: full ``repr`` spends ~19 characters per cell encoding differences no reader
#: can see. At 4 dp the worst-case error is 5e-5 -- 1/80th of one colormap step
#: on a unit-span field, and below the last printed digit -- while a run-26
#: payload drops 2.4x (0.38 -> 0.16 MB). Axis and coordinate values are rounded
#: on the same terms: they are shown as tick and readout text, which is already
#: formatted to 3 significant figures, and they are ~0.3% of the payload either
#: way, so exempting them would buy precision nothing downstream reads.
PAYLOAD_DECIMALS = 4


def _frames_array(frames):
    """A frame cube as a plain float array, masked entries becoming NaN.

    ``np.asarray`` on a masked array drops the mask and hands back the values
    *underneath* it, so a masked point would plot as whatever was there rather
    than as the gap the mask asked for -- wrong, and silently so. NaN is what
    both the page and the colormap already treat as missing.
    """
    if np.ma.isMaskedArray(frames):
        return np.ma.filled(frames.astype(float), np.nan)
    return np.asarray(frames, float)


def _round_floats(value):
    """``value`` with every float rounded to :data:`PAYLOAD_DECIMALS`.

    Applied to the whole payload rather than to the frame cubes alone, so there
    is one precision rule to state instead of a per-key exemption list. Kept
    apart from :func:`_jsonable`, which also serializes ``save_bundle``'s spine
    -- the bundle is the full-precision copy and must not be rounded.
    """
    if isinstance(value, float):
        return round(value, PAYLOAD_DECIMALS)
    if isinstance(value, dict):
        return {k: _round_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_floats(v) for v in value]
    return value


def _colormap_lut(name, n=256):
    """``name`` as an ``n x 3`` uint8 RGB lookup table, as nested lists.

    Exporting matplotlib's real LUT keeps the page's colors identical to the
    PNGs, instead of eyeballing RGB anchors in JavaScript.
    """
    rgba = matplotlib.colormaps[name](np.linspace(0.0, 1.0, n))
    return np.round(rgba[:, :3] * 255).astype(np.uint8).tolist()


def _payload(bundle):
    """The JSON object the page reads, built from a validated bundle.

    Panel *appearance* (name, unit, colormap, fixed scale) is shared by every
    group and emitted once as ``panels``; only the frame cubes vary per group.
    Emitting a 256x3 colormap LUT per channel instead would repeat ~6.6 kB that
    carries no new information -- about 3.6% of a channel's cost at run-26 size,
    where the frame cubes dominate. The point is less the bytes than that the
    page has one panel row to build its canvases from, whichever channel is
    selected.

    Every float here is rounded to :data:`PAYLOAD_DECIMALS`; the ``.npz`` beside
    the page keeps full precision, so the bundle -- not the page -- is what to
    re-read for anything quantitative.
    """
    groups = bundle_groups(bundle)

    # Validation has already established every group shares this layout, so the
    # first group defines the panels for all of them.
    panels = []
    for field in groups[0]["fields"]:
        vmin, vmax = field.get("vmin"), field.get("vmax")
        panels.append({
            "name": field["name"],
            "unit": field.get("unit", ""),
            "lut": _colormap_lut(field["cmap"]),
            # None/None tells the page this field follows the scale toggle.
            "vmin": None if vmin is None else float(vmin),
            "vmax": None if vmax is None else float(vmax),
            # Trace *names* are panel furniture (the legend) and identical
            # across groups by the layout rule, so they ride with the panel;
            # only the cubes below vary per group.
            "traces": [t.get("name", "") for t in field_traces(field)],
        })

    def _axis_payload(axis):
        return {"name": axis["name"], "unit": axis["unit"],
                "values": _jsonable(np.asarray(axis["values"], float))}

    # A group ships its own axis only when it actually overrides the bundle's;
    # the page falls back to the bundle axis otherwise. Emitting it
    # unconditionally duplicated one identical array per group -- on a
    # many-channel xcorr page that is the frequency axis serialized N times.
    def _group_payload(group):
        entry = {"name": group.get("name", "")}
        if group.get("axis"):
            entry["axis"] = _round_floats(_axis_payload(group["axis"]))
        entry["frames"] = [[_jsonable(np.round(_frames_array(t["frames"]),
                                               PAYLOAD_DECIMALS))
                            for t in field_traces(f)]
                           for f in group["fields"]]
        return entry

    # Rounding is applied per part rather than to the finished payload: the
    # cubes are already rounded by np.round above, and re-walking them in
    # Python costs ~15x the serialization itself to provably change nothing.
    payload = {
        "geometry": bundle["geometry"],
        "axis": _round_floats(_axis_payload(bundle["axis"])),
        "x": _round_floats(_jsonable(bundle["x"])),
        "panels": _round_floats(panels),
        "groups": [_group_payload(group) for group in groups],
    }
    if bundle["geometry"] == "plane":
        payload["y"] = _round_floats(_jsonable(bundle["y"]))
    return payload


def _script_json(obj):
    """``obj`` as compact JSON, safe to embed inside an inline ``<script>``.

    An HTML parser ends a script block at the first literal ``</script>``,
    wherever it appears -- including inside a JSON string. A field name or axis
    label carrying one would truncate the payload mid-literal and the page would
    die with a ``SyntaxError``, rendering as a bare heading. Escaping ``<`` as
    ``\\u003c`` cannot form that sequence; ``JSON.parse`` decodes the escape, so
    every string arrives at the page unchanged.
    """
    return json.dumps(obj, separators=(",", ":")).replace("<", "\\u003c")


def _banner_html(provenance):
    """Provenance dict -> the page's banner markup (escaped, one row per item)."""
    if not provenance:
        return ""
    rows = []
    source = provenance.get("source")
    if source:
        rows.append(f"<b>source</b> {escape(str(source))}")
    for key, value in (provenance.get("params") or {}).items():
        rows.append(f"<b>{escape(str(key))}</b> {escape(str(value))}")
    return " &nbsp;·&nbsp; ".join(rows)


# =========================================================================== #
#  Bundle round-trip -- a small standalone npz, never the experiment's data file
# =========================================================================== #

def save_bundle(bundle, path):
    """Write a validated bundle to a standalone ``.npz`` (arrays + a JSON spine).

    Small (prepared frames only) and independent of the experiment's own data
    ``.npz``, which this module never touches. :func:`load_bundle` reverses it.
    """
    path = Path(path)
    arrays = {"__axis_values__": np.asarray(bundle["axis"]["values"], float),
              "__x_values__": np.asarray(bundle["x"]["values"], float)}
    if bundle["geometry"] == "plane":
        arrays["__y_values__"] = np.asarray(bundle["y"]["values"], float)

    spine = {k: v for k, v in bundle.items()
             if k not in ("axis", "x", "y", "fields", "groups")}
    spine["axis"] = {"name": bundle["axis"]["name"], "unit": bundle["axis"]["unit"]}
    spine["x"] = {k: v for k, v in bundle["x"].items() if k != "values"}
    if bundle["geometry"] == "plane":
        spine["y"] = {k: v for k, v in bundle["y"].items() if k != "values"}

    # Frame cubes are keyed by (group, field) so a multi-channel bundle round
    # trips as one file. Everything is written in the grouped spelling, single
    # channel included: back-compat is a *read* requirement (npz files written
    # before groups existed are on disk), and honouring it on write too would
    # mean two on-disk formats kept in sync by hand. load_bundle carries the
    # one legacy branch instead.
    spine["groups"] = []
    for g, group in enumerate(bundle_groups(bundle)):
        entry = {"name": group.get("name", ""), "fields": []}
        # A per-group axis is stored like the bundle's: values to an array, the
        # rest to the spine.
        if group.get("axis"):
            arrays[f"__axis_{g}__"] = np.asarray(group["axis"]["values"], float)
            entry["axis"] = {k: v for k, v in group["axis"].items() if k != "values"}
        for i, field in enumerate(group["fields"]):
            spec = {k: v for k, v in field.items() if k not in ("frames", "traces")}
            # Cubes are always keyed per trace, so there is one array layout on
            # disk. The *spine* keeps the field's own spelling: a plane field
            # is single-curve by rule, and handing it back as 'traces' would
            # make a reloaded plane bundle fail its own validation.
            traces = field_traces(field)
            for t, trace in enumerate(traces):
                arrays[f"__frames_{g}_{i}_{t}__"] = _frames_array(trace["frames"])
            if "traces" in field:
                spec["traces"] = [{"name": t.get("name", "")} for t in traces]
            entry["fields"].append(spec)
        spine["groups"].append(entry)

    # Through _jsonable first: adapters routinely put values read back from an
    # npz into 'provenance', and np.int64 is not JSON-serializable (np.float64
    # is, since it subclasses float -- so the gap is easy to miss in testing).
    np.savez_compressed(path, __spine__=np.array(json.dumps(_jsonable(spine))),
                        **arrays)
    return path


def load_bundle(path):
    """Read back a bundle written by :func:`save_bundle`.

    Also reads the pre-groups layout (a ``fields`` spine with ``__frames_{i}__``
    keys) and the pre-traces one (``__frames_{g}_{i}__``), which files written
    by earlier versions still use -- the only place the old formats are
    understood, so the rest of the module sees one shape.

    Groups are normalized (a ``fields`` bundle comes back as one group), but a
    field keeps the spelling it was saved with -- ``frames`` stays ``frames``.
    Cubes are keyed per trace on disk either way; only the spine records which
    spelling applies, so a reloaded bundle re-validates exactly as it was.
    """
    with np.load(path, allow_pickle=False) as data:
        bundle = json.loads(str(data["__spine__"]))
        bundle["axis"]["values"] = data["__axis_values__"]
        bundle["x"]["values"] = data["__x_values__"]
        if bundle["geometry"] == "plane":
            bundle["y"]["values"] = data["__y_values__"]

        if "groups" not in bundle:                      # pre-groups npz
            for i, field in enumerate(bundle["fields"]):
                field["frames"] = data[f"__frames_{i}__"]
        else:
            for g, group in enumerate(bundle["groups"]):
                if "axis" in group:
                    group["axis"]["values"] = data[f"__axis_{g}__"]
                for i, field in enumerate(group["fields"]):
                    if "traces" in field:
                        for t, trace in enumerate(field["traces"]):
                            trace["frames"] = data[f"__frames_{g}_{i}_{t}__"]
                    elif f"__frames_{g}_{i}_0__" in data:
                        field["frames"] = data[f"__frames_{g}_{i}_0__"]
                    else:                       # pre-traces npz: one cube/field
                        field["frames"] = data[f"__frames_{g}_{i}__"]
    return bundle


# =========================================================================== #
#  Rendering
# =========================================================================== #

def write_slider_html(bundle, out_path, save_bundle_npz=True):
    """Render a bundle to a self-contained ``.html`` page. Returns its path.

    The bundle is validated first, so a schema mistake fails here with a message
    naming the key rather than rendering something silently wrong.

    Args:
        bundle (dict): A schema-v1 bundle (see the module docstring).
        out_path: Destination ``.html``. Parent directories are created.
        save_bundle_npz (bool): Also write ``<stem>-bundle.npz`` beside the page
            so it can be re-rendered later without re-running the analysis.

    Warns if the page exceeds :data:`ARTIFACT_MAX_MB`, above which it cannot be
    published as an Artifact for viewing on a phone.
    """
    validate_bundle(bundle)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Titles and warnings are prose that routinely carries "<", ">" and "&"
    # (e.g. "Te>5", "gamma2 < 0.2"), so escape them like the banner.
    fills = {
        "__TITLE__": escape(bundle.get("title", "slider")),
        "__BANNER__": _banner_html(bundle.get("provenance")),
        "__WARNING__": escape(bundle.get("warning") or ""),
        "__PAYLOAD__": _script_json(_payload(bundle)),
    }
    # One pass, so substituted text is never itself rescanned: a title of
    # literally "__PAYLOAD__" stays that text instead of being replaced by the
    # whole payload on the next .replace().
    html = re.sub("|".join(map(re.escape, fills)), lambda m: fills[m.group(0)],
                  _TEMPLATE.read_text(encoding="utf-8"))
    out_path.write_text(html, encoding="utf-8")

    size_mb = out_path.stat().st_size / 1e6
    if size_mb > ARTIFACT_MAX_MB:
        print(f"  WARNING: {out_path.name} is {size_mb:.1f} MB "
              f"(> {ARTIFACT_MAX_MB:.0f} MB) "
              "-- too large to publish as an Artifact; narrow the axis range "
              "or coarsen the grid.")
    groups = bundle_groups(bundle)
    channels = f"{len(groups)} channel(s) x " if len(groups) > 1 else ""
    print(f"Slider page: {out_path}  ({size_mb:.2f} MB, {channels}"
          f"{len(groups[0]['fields'])} field(s) x "
          f"{np.size(group_axis(bundle, groups[0])['values'])} frames)")

    if save_bundle_npz:
        npz_path = save_bundle(bundle, out_path.with_name(out_path.stem + "-bundle.npz"))
        print(f"  bundle: {npz_path}")
    return out_path


def main(argv=None):
    """CLI: re-render a saved bundle npz after a template change."""
    parser = argparse.ArgumentParser(
        description="Re-render a saved slider bundle to a self-contained HTML page.")
    parser.add_argument("bundle", help="a <stem>-bundle.npz written by write_slider_html")
    parser.add_argument("-o", "--out", default=None,
                        help="output .html (default: alongside the bundle)")
    args = parser.parse_args(argv)

    bundle_path = Path(args.bundle)
    out = Path(args.out) if args.out else bundle_path.with_name(
        bundle_path.stem.removesuffix("-bundle") + ".html")
    # The bundle npz was just re-read; re-writing it here would be a no-op copy.
    write_slider_html(load_bundle(bundle_path), out, save_bundle_npz=False)


if __name__ == "__main__":
    main()
