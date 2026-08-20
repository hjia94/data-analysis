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
never share a generic dimension name. One bundle carries exactly one axis.

**fields** -- named physical quantities, one frame cube each, *all* sharing that
one geometry and that one axis. Each field draws as its own panel, and every
panel moves together on the single slider.

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
        ],
        "provenance": {"source": str, "params": {...}},   # rendered as a banner
        "warning": str | None,              # optional caveat banner
    }

Rules, all enforced by :func:`validate_bundle`:

* Exactly one axis. Every field's ``frames.shape[0]`` equals ``axis.values.size``.
* Every field shares the geometry: ``(n_axis, ny, nx)`` for a plane,
  ``(n_axis, nx)`` for a line, matching ``x`` (and ``y``).
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

    axis = bundle.get("axis")
    _require(isinstance(axis, dict), "'axis' must be a dict {name, unit, values}")
    for k in ("name", "unit", "values"):
        _require(k in axis, f"'axis' is missing '{k}'")
    axis_values = np.asarray(axis["values"], float)
    _require(axis_values.ndim == 1 and axis_values.size > 0,
             f"'axis.values' must be 1-D and non-empty, got shape {axis_values.shape}")
    # The slider steps through these in order, so an unsorted axis would scrub
    # to frames that jump around the scan rather than sweeping it.
    _require(np.all(np.diff(axis_values) > 0) or np.all(np.diff(axis_values) < 0),
             f"'axis.values' must be monotonic (it is the slider's order); "
             f"got {axis_values[:4]}...")
    n_axis = axis_values.size

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

    want = ((n_axis, sizes["y"], sizes["x"]) if geometry == "plane"
            else (n_axis, sizes["x"]))

    fields = bundle.get("fields")
    _require(isinstance(fields, (list, tuple)) and len(fields) > 0,
             "'fields' must be a non-empty list")
    for i, field in enumerate(fields):
        where = f"fields[{i}]"
        _require(isinstance(field, dict), f"{where} must be a dict")
        for k in ("name", "unit", "frames", "cmap"):
            _require(k in field, f"{where} is missing '{k}'")
        # np.shape reads the shape without materializing/copying the cube.
        shape = np.shape(field["frames"])
        _require(shape == want,
                 f"{where} ('{field['name']}') frames have shape {shape}, "
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


def _colormap_lut(name, n=256):
    """``name`` as an ``n x 3`` uint8 RGB lookup table, as nested lists.

    Exporting matplotlib's real LUT keeps the page's colors identical to the
    PNGs, instead of eyeballing RGB anchors in JavaScript.
    """
    rgba = matplotlib.colormaps[name](np.linspace(0.0, 1.0, n))
    return np.round(rgba[:, :3] * 255).astype(np.uint8).tolist()


def _payload(bundle):
    """The JSON object the page reads, built from a validated bundle."""
    fields = []
    for field in bundle["fields"]:
        vmin, vmax = field.get("vmin"), field.get("vmax")
        fields.append({
            "name": field["name"],
            "unit": field.get("unit", ""),
            "frames": _jsonable(np.asarray(field["frames"], float)),
            "lut": _colormap_lut(field["cmap"]),
            # None/None tells the page this field follows the scale toggle.
            "vmin": None if vmin is None else float(vmin),
            "vmax": None if vmax is None else float(vmax),
        })

    payload = {
        "geometry": bundle["geometry"],
        "axis": {"name": bundle["axis"]["name"],
                 "unit": bundle["axis"]["unit"],
                 "values": _jsonable(np.asarray(bundle["axis"]["values"], float))},
        "x": _jsonable(bundle["x"]),
        "fields": fields,
    }
    if bundle["geometry"] == "plane":
        payload["y"] = _jsonable(bundle["y"])
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
             if k not in ("axis", "x", "y", "fields")}
    spine["axis"] = {"name": bundle["axis"]["name"], "unit": bundle["axis"]["unit"]}
    spine["x"] = {k: v for k, v in bundle["x"].items() if k != "values"}
    if bundle["geometry"] == "plane":
        spine["y"] = {k: v for k, v in bundle["y"].items() if k != "values"}
    spine["fields"] = []
    for i, field in enumerate(bundle["fields"]):
        arrays[f"__frames_{i}__"] = np.asarray(field["frames"], float)
        spine["fields"].append({k: v for k, v in field.items() if k != "frames"})

    # Through _jsonable first: adapters routinely put values read back from an
    # npz into 'provenance', and np.int64 is not JSON-serializable (np.float64
    # is, since it subclasses float -- so the gap is easy to miss in testing).
    np.savez_compressed(path, __spine__=np.array(json.dumps(_jsonable(spine))),
                        **arrays)
    return path


def load_bundle(path):
    """Read back a bundle written by :func:`save_bundle`."""
    with np.load(path, allow_pickle=False) as data:
        bundle = json.loads(str(data["__spine__"]))
        bundle["axis"]["values"] = data["__axis_values__"]
        bundle["x"]["values"] = data["__x_values__"]
        if bundle["geometry"] == "plane":
            bundle["y"]["values"] = data["__y_values__"]
        for i, field in enumerate(bundle["fields"]):
            field["frames"] = data[f"__frames_{i}__"]
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
    print(f"Slider page: {out_path}  ({size_mb:.2f} MB, "
          f"{len(bundle['fields'])} field(s) x {np.size(bundle['axis']['values'])} frames)")

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
