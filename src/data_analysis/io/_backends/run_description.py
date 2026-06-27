"""Tolerant parser + diff for the hand-written LAPD ``description`` attribute.

LAPD pydaq HDF5 files carry a free-text ``description`` root attribute that the
operator writes by hand (purpose prose -> ``Operator:`` -> a ``Setup:`` block of
indented bullets: plasma condition, magnetic field, multi-electrode bias, probe).
During a run series usually **one** setting changes (``uniform 800G`` ->
``uniform 600G``, a probe resistor, a bias duration). The text is *almost* but not
*exactly* consistent run to run: unit spacing (``800G`` vs ``800 G``), tabs vs
spaces, ``(NOT USED)`` markers, trailing commas, and typo bullets (``-Resistor``).

This module turns that text into a structured :class:`RunDescription` and diffs
two of them, classifying each setting as changed / added / removed. The diff
compares on a *normalized* form (so formatting drift is not a difference) but
reports the *raw* text the operator wrote. :meth:`RunDiff.summary` renders all
changes on one line, suitable for a plot title or a printed banner.

Pure text in, pure data out -- no HDF5/h5py dependency, so it is unit-testable
without data files. The HDF5 read path lives in :mod:`..pydaq`
(``_find_description_attribute`` -> this parser); experiment code reaches it via
``data_analysis.io.open_lapd(path).description()`` and ``compare_runs``.
"""

import re
from collections import OrderedDict

# Reuse the heading canonicalizer already proven on these files (maps "Magnetic
# field" / "Plasma condition" / "Probe" -> canonical names). The sectioning itself
# is done here, indentation-aware, because the descriptions use bulleted headers
# ("- Magnetic field") rather than the underline/colon headings that
# _split_description_sections keys on.
from .pydaq import _canonical_description_section

__all__ = [
    "RunDescription",
    "RunDiff",
    "Section",
    "Item",
    "parse_description",
    "diff_descriptions",
]


# --------------------------------------------------------------------------- #
# normalization -- the "stay vague" core
# --------------------------------------------------------------------------- #
_BULLET_RE = re.compile(r"^[\s\-\*•>#]+")
_NOT_USED_RE = re.compile(r"\(?\bnot\s+used\b\)?", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_PORT_RE = re.compile(r"^p\d{1,3}$", re.IGNORECASE)  # probe-port label, e.g. "P29"
# Recognized unit tokens (shared by the two regexes below). Multi-char units are
# listed before their single-char prefixes (kg before g, ka before a, kv before v)
# so the alternation's leftmost-match picks the longer unit ("1kG", "4kA").
_UNITS = r"ka|kg|kv|psi|ms|us|ohm|cm\^-3|cm|mm|hz|g|a|v|w"
# A number immediately followed (optionally via spaces) by a unit token; used to
# collapse "800 G" -> "800g" so spacing drift is not a diff.
_NUM_UNIT_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*"
    rf"({_UNITS})\b",
    re.IGNORECASE,
)


def _strip_not_used(text):
    """Return ``(text_without_marker, was_marked)`` for a ``(NOT USED)`` bullet."""
    if _NOT_USED_RE.search(text):
        return _NOT_USED_RE.sub("", text).strip(" ()"), True
    return text, False


def _strip_bullet(line):
    """Remove leading bullet glyphs/whitespace from a line (keep inner text)."""
    return _BULLET_RE.sub("", line).strip()


def _normalize_key(text):
    """Canonical form of a key/label for path building and comparison.

    Lowercase, whitespace-collapsed, bullet- and trailing-punctuation-stripped.
    Robust to the typo/spacing drift seen across runs.
    """
    text = _strip_bullet(text)
    text = text.strip().rstrip(":").strip().rstrip(",").strip()
    return _WS_RE.sub(" ", text).lower()


def _normalize_value(text):
    """Canonical form of a value for *comparison* (display uses the raw text).

    Collapses whitespace, drops a trailing comma, and removes the space between a
    number and its unit (``800 G`` -> ``800g``) with unit casing unified, so
    cosmetic formatting differences do not register as a changed setting.
    """
    if text is None:
        return None
    text = _WS_RE.sub(" ", text.strip()).rstrip(",").strip()
    text = _NUM_UNIT_RE.sub(lambda m: f"{m.group(1)}{m.group(2).lower()}", text)
    return text.lower()


def _indent_width(line, tabsize=4):
    """Leading-whitespace width of ``line`` with tabs expanded (tabs vs spaces)."""
    width = 0
    for ch in line:
        if ch == "\t":
            width += tabsize
        elif ch == " ":
            width += 1
        else:
            break
    return width


def _split_key_value(text):
    """Split ``"key: value"`` on the first colon -> ``(key, value)``.

    Returns ``(None, None)`` when there is no usable colon (a free-text bullet) or
    when the colon is only a trailing label marker (``"P29:"`` with no value).
    """
    if ":" not in text:
        return None, None
    key, value = text.split(":", 1)
    key, value = key.strip(), value.strip()
    if not key:
        return None, None
    return key, value


def _looks_like_label(text):
    """True if a line introduces a nested subgroup (e.g. a probe port ``"P29:"``).

    A label is a short head token ending in a colon whose value part is short or
    empty -- ``"P29:"``, ``"P28: Bdot-10T-C16"`` -- as opposed to a setting line
    like ``"Resistor: 43/300 for L/R respectively"``.
    """
    key, value = _split_key_value(text)
    if key is None:
        return False
    head = key.split()[0] if key.split() else ""
    # Port-style head (P29, P28, Port 35) or an empty/very short value -> label.
    return bool(_PORT_RE.match(head)) or value == ""


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
class Item:
    """One parsed line: a ``key: value`` setting or a free-text bullet.

    ``key``/``value`` are the raw (display) strings; ``key`` is ``None`` for a
    pure free-text bullet. ``not_used`` records a stripped ``(NOT USED)`` marker.
    """

    __slots__ = ("key", "value", "raw", "not_used")

    def __init__(self, key, value, raw, not_used=False):
        self.key = key
        self.value = value
        self.raw = raw
        self.not_used = not_used

    def __repr__(self):
        flag = " (NOT USED)" if self.not_used else ""
        if self.key is None:
            return f"Item(text={self.raw!r}{flag})"
        return f"Item({self.key!r}: {self.value!r}{flag})"


class Section:
    """A ``Setup:`` section: flat ``items`` plus ``subgroups`` (e.g. probe ports)."""

    __slots__ = ("items", "subgroups")

    def __init__(self):
        self.items = []
        self.subgroups = OrderedDict()  # label -> list[Item]

    def __repr__(self):
        return f"Section(items={len(self.items)}, subgroups={list(self.subgroups)})"


class RunDescription:
    """Structured view of a parsed ``description`` string.

    Attributes:
        header: free-text intro lines (purpose / one-line summary).
        operator: the ``Operator:`` value, or ``None``.
        sections: ``OrderedDict`` of canonical section name -> :class:`Section`.
        raw: the original description text.
    """

    def __init__(self, header, operator, sections, raw):
        self.header = header
        self.operator = operator
        self.sections = sections
        self.raw = raw

    def __repr__(self):
        return (
            f"RunDescription(operator={self.operator!r}, "
            f"sections={list(self.sections)})"
        )

    def to_flat(self):
        """Flatten to ``{path: (raw_value, normalized_value, not_used)}``.

        ``path`` is a normalized ``"section / [subgroup /] key"`` string so that
        spacing/typo drift does not create spurious diff entries. Free-text bullets
        are keyed by their normalized text (value ``None``) so a bullet appearing
        on only one side still shows up as added/removed. ``raw_value`` is what the
        operator wrote (for display); ``normalized_value`` is what the diff compares.
        """
        flat = OrderedDict()

        def add(path, raw_value, not_used):
            # Disambiguate repeated free-text bullets / keys under one parent.
            key = path
            n = 2
            while key in flat:
                key = f"{path} #{n}"
                n += 1
            flat[key] = (raw_value, _normalize_value(raw_value), not_used)

        def add_items(prefix, items):
            for it in items:
                if it.key is not None:
                    add(f"{prefix} / {_normalize_key(it.key)}", it.value, it.not_used)
                else:
                    # Free-text bullet: key is the normalized text, no value.
                    add(f"{prefix} / {_normalize_key(it.raw)}", None, it.not_used)

        for sec_name, sec in self.sections.items():
            sec_key = _normalize_key(sec_name)
            add_items(sec_key, sec.items)
            for label, items in sec.subgroups.items():
                add_items(f"{sec_key} / {_normalize_key(label)}", items)
        return flat


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def parse_description(description):
    """Parse a raw ``description`` string into a :class:`RunDescription`.

    The descriptions are: free-text prose, an ``Operator:`` line, then a
    ``Setup:`` block of nested bullets. Sectioning is indentation-aware -- the
    shallowest bullet level inside ``Setup:`` are the section headers
    (``- Plasma condition:``, ``- Magnetic field``, ``- Probe``), canonicalized via
    :func:`..pydaq._canonical_description_section`; deeper bullets are that
    section's items; a label bullet (``P29:``) opens a nested subgroup that
    collects the bullets indented beneath it.
    """
    if description is None:
        description = ""

    lines = description.splitlines()
    operator = None
    header = []
    sections = OrderedDict()

    # Locate the Setup: block; everything before it (minus Operator:) is header.
    setup_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _is_operator_line(line):
            _, operator = _split_key_value(stripped)
            operator = operator or None
            continue
        if _clean_heading(stripped).lower() == "setup":
            setup_idx = i
            break
        if stripped:
            header.append(_strip_bullet(stripped))

    body = lines if setup_idx is None else lines[setup_idx + 1:]
    _parse_setup_block(body, sections)

    return RunDescription(header, operator, sections, description)


def _clean_heading(text):
    """Strip bullet glyphs and a trailing colon from a candidate heading line."""
    return _strip_bullet(text).rstrip(":").strip()


def _is_operator_line(line):
    return line.strip().lower().startswith("operator")


def _parse_setup_block(lines, sections):
    """Fill ``sections`` from the indented bullet body of a ``Setup:`` block.

    Three indent levels matter: section header (shallowest non-blank bullet),
    item, and subgroup-content (under a label item). The header level is taken
    from the first non-blank line so tab- and space-indented files behave alike.
    """
    header_indent = None
    current_section = None
    current_label = None
    label_indent = None

    for raw_line in lines:
        if not raw_line.strip():
            continue
        indent = _indent_width(raw_line)
        text = _strip_bullet(raw_line)
        text, not_used = _strip_not_used(text)
        if not text:
            continue

        if header_indent is None:
            header_indent = indent

        # Shallowest level -> a section header.
        if indent <= header_indent:
            current_section = _start_section(sections, text)
            current_label = None
            label_indent = None
            continue

        if current_section is None:  # bullet before any header -> synthesize one
            current_section = _start_section(sections, "Other")

        # Under an open label and more indented -> that label's subgroup.
        if current_label is not None and label_indent is not None and indent > label_indent:
            current_section.subgroups[current_label].append(_make_item(text, not_used))
            continue

        # Item level: a label opens a subgroup, otherwise it's a flat item.
        if _looks_like_label(text):
            label, rest = _split_key_value(text)
            current_label = label
            label_indent = indent
            current_section.subgroups.setdefault(label, [])
            if rest:  # "P28: Bdot-10T-C16" -- keep the inline descriptor
                current_section.subgroups[label].append(_make_item(rest, not_used))
        else:
            current_label = None
            label_indent = None
            current_section.items.append(_make_item(text, not_used))


def _start_section(sections, header_text):
    """Get/create the canonical :class:`Section` for a header line."""
    # The header may carry an inline value ("Multi-electrode bias: Port 35 top");
    # canonicalize on the key part only.
    key, _ = _split_key_value(header_text)
    title = key if key is not None else header_text
    name = _canonical_description_section(_clean_heading(title))
    if name not in sections:
        sections[name] = Section()
    return sections[name]


# A free-text bullet ending in a measurement / range, e.g. "Bias on 12-17ms",
# "Bias on for 5ms". The leading words become a stable key and the numeric tail
# the value, so a change to just the number diffs as *changed* (same path) rather
# than as add+remove (a path that embeds the number).
_TRAILING_MEASURE_RE = re.compile(
    r"^(?P<key>.*?[a-zA-Z])\s+"
    r"(?P<val>(?:for\s+|~)?\d[\d.\s\-–to]*"
    rf"(?:{_UNITS})?)\s*$",
    re.IGNORECASE,
)


def _make_item(text, not_used):
    key, value = _split_key_value(text)
    if key is not None:
        return Item(key, value, text, not_used)
    # No colon: try to peel a trailing measurement off a free-text bullet so a
    # numeric-only change is comparable. Require the value to actually contain a
    # digit and the key to have a couple of words (avoid splitting "P28" etc.).
    m = _TRAILING_MEASURE_RE.match(text)
    if m and any(ch.isdigit() for ch in m.group("val")) and len(m.group("key").split()) >= 2:
        return Item(m.group("key").strip(), m.group("val").strip(), text, not_used)
    return Item(None, None, text, not_used)


# --------------------------------------------------------------------------- #
# diff
# --------------------------------------------------------------------------- #
# Ranking for the one-line summary: most physically significant settings first.
# Matched as substrings against the normalized diff path.
_RANK_HINTS = (
    ("uniform", 0),       # magnetic field strength ("...uniform 800g")
    ("magnetic", 0),
    ("helium", 1),        # gas / fill
    ("gas", 1),
    ("bias", 2),          # biasing
    ("density", 3),
    ("resistor", 4),
    ("heater", 5),
)

# Friendly short labels for the summary, matched as substrings against the path.
_FRIENDLY = (
    ("magnetic field / yellow", "B-field"),
    ("uniform", "B-field"),
    ("magnetic", "B-field"),
    ("helium", "gas"),
    ("density", "density"),
    ("bias on", "bias"),
    ("bias voltage", "bias V"),
    ("bias", "bias"),
    ("resistor", "resistor"),
    ("heater", "heater"),
    ("puff", "puff"),
)


class RunDiff:
    """The classified difference between two :class:`RunDescription` s.

    Attributes:
        changed: list of ``(path, raw_a, raw_b)`` -- present on both sides with
            differing normalized values, ranked most-significant first.
        only_in_a / only_in_b: list of ``(path, raw_value)`` present on one side.
        label_a / label_b: optional short labels for the two runs (e.g. filenames).
    """

    def __init__(self, changed, only_in_a, only_in_b, label_a=None, label_b=None):
        self.changed = changed
        self.only_in_a = only_in_a
        self.only_in_b = only_in_b
        self.label_a = label_a
        self.label_b = label_b

    def __bool__(self):
        return bool(self.changed or self.only_in_a or self.only_in_b)

    def __repr__(self):
        return (
            f"RunDiff(changed={len(self.changed)}, "
            f"only_in_a={len(self.only_in_a)}, only_in_b={len(self.only_in_b)})"
        )

    def summary(self, include_only_one_sided=True, max_changes=None, arrow="->"):
        """One-line human summary of all changes, joined by ``"; "``.

        Each *changed* setting renders as ``"<label> <raw_a>-><raw_b>"`` using a
        short friendly label (the most common case, e.g. ``"B-field 800G->600G"``).
        When ``include_only_one_sided``, settings present on only one side that map
        to a *recognized* field (B-field, gas, bias, ...) are listed as
        ``"+<label>"`` / ``"-<label>"``; the remaining one-sided items (operator
        prose reworded between runs) are collapsed into a trailing
        ``"(+N/-M other)"`` count so a genuinely different pair does not flood the
        line. ``arrow`` is ASCII by default (console-safe); pass ``"→"`` for a
        real arrow in plot titles. Returns ``"no differences"`` when the runs match.
        ``max_changes`` truncates the changed list with a ``"(+N more)"`` tail.
        """
        parts = []
        for path, raw_a, raw_b in self.changed:
            parts.append(f"{_friendly_path(path)} {_short(raw_a)}{arrow}{_short(raw_b)}")

        if max_changes is not None and len(parts) > max_changes:
            extra = len(parts) - max_changes
            parts = parts[:max_changes] + [f"(+{extra} more)"]

        if include_only_one_sided:
            other = []
            for side, sign in ((self.only_in_b, "+"), (self.only_in_a, "-")):
                named = [_friendly_path(p) for p, _ in side if _is_recognized(p)]
                parts += [f"{sign}{lbl}" for lbl in named]
                if len(side) - len(named):
                    other.append(f"{sign}{len(side) - len(named)}")
            if other:
                parts.append(f"({'/'.join(other)} other)")

        if not parts:
            return "no differences"
        return "; ".join(parts)


def diff_descriptions(a, b, label_a=None, label_b=None):
    """Diff two :class:`RunDescription` s into a :class:`RunDiff`.

    Compares the flattened, normalized settings of ``a`` and ``b``: a path present
    on both with a differing normalized value -- or a toggled ``(NOT USED)`` marker
    -- is *changed*; present on one side only is added/removed. ``changed`` is
    ranked most-significant first (B-field, gas, bias, density, ...).
    ``label_a``/``label_b`` are carried through for display (e.g. the two filenames).
    """
    fa, fb = a.to_flat(), b.to_flat()

    changed = []
    only_in_a = []
    only_in_b = []

    for path, (raw_a, norm_a, nu_a) in fa.items():
        if path in fb:
            raw_b, norm_b, nu_b = fb[path]
            if norm_a != norm_b:
                changed.append((path, raw_a, raw_b))
            elif nu_a != nu_b:
                # Same value, but the (NOT USED) marker was toggled between runs
                # (e.g. a probe disconnected) -- surface that in the raw values.
                changed.append(
                    (path, _mark_used(raw_a, nu_a), _mark_used(raw_b, nu_b))
                )
        else:
            only_in_a.append((path, raw_a))
    for path, (raw_b, _, _) in fb.items():
        if path not in fa:
            only_in_b.append((path, raw_b))

    changed.sort(key=lambda c: _rank(c[0]))
    return RunDiff(changed, only_in_a, only_in_b, label_a, label_b)


def _rank(path):
    """Sort key for a diff path: lower = more significant (see ``_RANK_HINTS``)."""
    for hint, rank in _RANK_HINTS:
        if hint in path:
            return rank
    return 99


def _friendly_path(path):
    """Short label for the summary; falls back to the path's last segment."""
    for hint, label in _FRIENDLY:
        if hint in path:
            return label
    return path.rsplit("/", 1)[-1].strip()


def _is_recognized(path):
    """True if ``path`` maps to a known significant field (has a friendly label)."""
    return any(hint in path for hint, _ in _FRIENDLY)


def _mark_used(raw_value, not_used):
    """Render a value with its in-use / NOT-USED state for a flag-only change.

    Used when a path's value is unchanged but its ``(NOT USED)`` marker toggled
    between runs, so the diff/summary can show *which* way it flipped.
    """
    state = "NOT USED" if not_used else "in use"
    if raw_value is None:
        return state
    return f"{raw_value} ({state})"


def _short(value):
    """Compact a raw value for the summary: pull out a trailing number+unit.

    Drops surrounding *words* ("configured for uniform 800G" -> "800G") but
    preserves a *numeric* prefix so a range or list is not truncated to its last
    term ("12-17ms" stays "12-17ms", not "17ms"). Falls back to the
    whitespace-collapsed full value when there is no number+unit to pull out.
    """
    if value is None:
        return "(none)"  # a free-text item present on one side without a value
    m = _NUM_UNIT_RE.search(value)
    # Only collapse to the bare number+unit when it is not part of a larger
    # numeric expression (range/list/decimal) -- i.e. the char before it is not a
    # digit or a range/list separator.
    if m and (m.start() == 0 or value[m.start() - 1] not in "-–/.0123456789"):
        # Keep the operator's own unit casing (so "1kG"/"4kA" stay readable),
        # only dropping any space between number and unit.
        return f"{m.group(1)}{m.group(2)}"
    return _WS_RE.sub(" ", value.strip())
