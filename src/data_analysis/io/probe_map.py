"""Map moving probes (motion groups) to the digitizer channels that recorded them.

Nothing in an LAPD HDF5 file links a motion group to its channels structurally.
Both sides are operator-written free text, and the only key they share is the
**port number**::

    motion group  '<Hermes>    p29_LP'   -> port 29
    channel       'Isat, LP@P29-R'       -> port 29

So the join is: parse ports from both sides, match. That is campaign-independent
-- it is a property of how LAPD_DAQ records wiring, not of any one experiment --
which is why it lives here rather than in an experiment script.

Where the channel wiring is written
-----------------------------------
In pydaq files, two places, and they disagree in real campaigns:

* ``Configuration/experiment_config``, ``[channels]`` -- operator-entered at
  setup. Authoritative in practice, and the ONLY source in runs whose scope
  attrs were never filled in.
* ``<scope>.attrs['<CH>_description']`` -- can lag the config, and is empty or a
  placeholder in whole blocks of runs.

:func:`channel_wiring` reads the config block and falls back to the scope attrs,
because a campaign that has only the latter should still join.

bapsflib files keep both sides somewhere else entirely -- wiring on the digitizer
configuration groups, motion groups in ``bmotion`` -- so each function here
dispatches on the layout. The join itself is unchanged, because both schemas
spell the port the same way in both places; only the addressing differs, with a
channel keyed ``(board, chan)`` rather than ``'C1'``.

What this module refuses to do
------------------------------
It does not guess. Three cases return "unknown" rather than a plausible answer,
because a wrong probe->channel mapping is silent: the shapes still match and the
spectra still plot, they are just labelled with the other probe's coordinates.

* a channel naming no port (e.g. ``'Bx'``) -- :data:`UNMAPPED`
* a channel whose port matches no motion group -- :data:`STATIONARY`, the usual
  cause being a fixed reference probe that is digitized but never moved
* a port claimed by two motion groups -- raises :class:`AmbiguousProbeMap`

The caller decides what to do with those. An empty answer is a correct statement
that the file does not say; a guessed one is indistinguishable from a verified
mapping.
"""

import re

import h5py

#: A channel that names no port: the file does not say which probe it belongs to.
UNMAPPED = "unmapped"

#: A channel whose port matches no *moving* probe -- a stationary diagnostic.
STATIONARY = "stationary"

# `p28_Bdot`: a trailing \b fails because `8_` is not a word boundary, so the
# port never matches. Require a non-digit instead.
_PORT_RE = re.compile(r"[pP](\d{1,2})(?!\d)")

# Campaigns write the scope prefix as `LPScope_C1`, `biasScope_C8` or
# `Bdot_scope_C1` -- all `<prefix>_C<n>`.
_CHAN_RE = re.compile(r"\s*(\w+?)_(C\d+)\s*=\s*(.+?)\s*$")
_SECTION_RE = re.compile(r"\s*\[(\w+)\]\s*$")


class AmbiguousProbeMap(ValueError):
    """A port is claimed by more than one motion group, so the join cannot decide.

    Carries ``port`` and ``candidates`` so a caller can report both names or pass
    an explicit override, rather than re-deriving them from the message.
    """

    def __init__(self, port, candidates, channel=None):
        self.port = port
        self.candidates = list(candidates)
        self.channel = channel
        where = f"channel {channel!r} (port {port})" if channel else f"port {port}"
        super().__init__(
            f"{where} is claimed by {len(self.candidates)} motion groups: "
            f"{self.candidates}. Name one explicitly to disambiguate."
        )


def ports_in(text):
    """Port numbers named in a probe or channel string. ``'p28_Bdot'`` -> ``[28]``."""
    return sorted({int(m) for m in _PORT_RE.findall(text or "")})


def config_text(f):
    """Raw ``Configuration/experiment_config`` text, or ``None`` if absent.

    Takes an OPEN h5py file, so a caller already walking the file (the campaign
    extractor) does not reopen it.
    """
    node = f.get("Configuration/experiment_config")
    if node is None or not isinstance(node, h5py.Dataset):
        return None
    raw = node[()]
    return raw.decode(errors="replace") if isinstance(raw, bytes) else str(raw)


def parse_config_channels(cfg):
    """``{(scope_prefix, 'C1'): description}`` from the ``[channels]`` section.

    The scope prefix is whatever the config writes, which need not match the HDF5
    group name (``LPScope`` vs ``lpscope``); :func:`channel_wiring` reconciles
    them case-insensitively. Only lines inside ``[channels]`` are read, so an
    unrelated ``foo_C1`` elsewhere in the INI cannot leak in. A leading ``;``
    comments a line out -- these configs keep previous setups in place, commented
    -- and the leading ``\\s*`` does not match it, so that wiring is skipped.
    """
    out = {}
    in_channels = False
    for line in (cfg or "").splitlines():
        sec = _SECTION_RE.match(line)
        if sec:
            in_channels = sec.group(1).lower() == "channels"
            continue
        if not in_channels:
            continue
        m = _CHAN_RE.match(line)
        if m:
            out[(m.group(1), m.group(2))] = m.group(3)
    return out


def _scope_attr_channels(f, scopes):
    """``{(scope, 'C1'): description}`` from each scope group's descriptions.

    The fallback source: campaigns that filled in scope descriptions but wrote no
    ``[channels]`` config block still join. Delegates to
    :func:`~data_analysis.io.scope_reader.read_scope_channel_descriptions`, which
    also handles the old per-shot ``<CH>_data`` layout that a hand-rolled attr
    scan would miss. Blank and placeholder descriptions ("No description
    available" -- what the writer emits when the operator left it empty) are
    dropped: they carry no port and would otherwise mask a usable config entry.
    """
    from .scope_reader import read_scope_channel_descriptions

    out = {}
    for scope in scopes:
        for chan, val in (read_scope_channel_descriptions(f, scope) or {}).items():
            text = val.decode(errors="replace") if isinstance(val, bytes) else str(val)
            if not text.strip() or "no description" in text.lower():
                continue
            out[(scope, chan)] = text
    return out


def channel_wiring(fn):
    """Every channel's wiring description: ``{(group, chan): description}``.

    Prefers the ``[channels]`` config block and falls back to scope attrs for
    channels the config does not mention (see the module docstring on why both
    exist). Scope names are returned as the config writes them unless an actual
    HDF5 group matches case-insensitively, in which case the on-disk spelling
    wins -- callers address channels by the group name they read data from.

    bapsflib files keep their wiring somewhere else entirely (on the digitizer
    configuration groups) and are delegated to that backend, which returns the
    same ``{(group, chan): description}`` shape with ``chan`` as the
    ``(board, chan)`` tuple its reader addresses. Dispatching here rather than
    leaving those files to fall through matters: this function found no scope
    groups in them and returned ``{}``, which reads as "the file records no
    wiring" -- indistinguishable from a run whose operator left it blank.
    """
    from .lapd_hdf5 import _has_shot_groups, detect_backend

    if detect_backend(fn) == "bapsflib":
        from ._backends import bapsflib_daq
        return bapsflib_daq.channel_wiring(fn)

    with h5py.File(fn, "r") as f:
        # Scope groups are the ones holding shot_* subgroups -- the repo's single
        # definition of that test. A plain "is a group" check would let
        # Configuration/Control into the name map below.
        scopes = [name for name in f if _has_shot_groups(f[name])]
        by_lower = {name.lower(): name for name in scopes}
        wiring = {}
        for (prefix, chan), desc in parse_config_channels(config_text(f)).items():
            wiring[(by_lower.get(prefix.lower(), prefix), chan)] = desc
        for key, desc in _scope_attr_channels(f, scopes).items():
            wiring.setdefault(key, desc)
    return wiring


#: Where each schema names its motion groups. pydaq gives one subgroup per
#: group; bapsflib names them in a `motion_group_name` column instead, with one
#: row per shot -- hence the dedup. Both spell the port the same way
#: (`p29_Nxy21-dxy1cm`), so :func:`ports_in` reads either.
_MOTION_GROUPS = "Control/Positions"
_BMOTION_AXES = "Raw data + config/bmotion/bmotion_axis_names"


def motion_group_ports(fn):
    """``{motion_group_name: [ports]}`` for every moving probe in the file.

    Empty only when the file truly moved nothing. Reading just the pydaq path
    would return ``{}`` for every bapsflib run, and an empty mapping does not
    fail -- it silently relabels every channel :data:`STATIONARY` in
    :func:`probe_channel_map`, which is a wrong answer rather than a refusal.
    """
    with h5py.File(fn, "r") as f:
        if _MOTION_GROUPS in f:
            return {name: ports_in(name) for name in f[_MOTION_GROUPS]}
        axes = f.get(_BMOTION_AXES)
        if axes is None:
            return {}
        names = {n.decode(errors="replace") if isinstance(n, bytes) else str(n)
                 for n in axes["motion_group_name"]}
        return {name: ports_in(name) for name in sorted(names)}


def probe_channel_map(fn):
    """Join every channel to the moving probe that recorded it.

    Returns ``{(scope, chan): motion_group_name | STATIONARY | UNMAPPED}`` --
    every channel the file describes, so a caller can see the stationary and
    unmapped ones rather than inferring them from absence.

    Raises :class:`AmbiguousProbeMap` if a channel's port is claimed by two
    motion groups: that is a real ambiguity in the file, and picking either
    would be a guess.
    """
    mg_ports = motion_group_ports(fn)
    out = {}
    for key, desc in channel_wiring(fn).items():
        chan_ports = ports_in(desc)
        if not chan_ports:
            out[key] = UNMAPPED
            continue
        owners = [g for g, ps in mg_ports.items()
                  if any(p in ps for p in chan_ports)]
        if len(owners) > 1:
            raise AmbiguousProbeMap(chan_ports, owners, channel=f"{key[0]}:{key[1]}")
        out[key] = owners[0] if owners else STATIONARY
    return out


def moving_group(mapping, channel):
    """One channel's moving probe from an already-built :func:`probe_channel_map`.

    ``None`` covers both "stationary" and "no port in the description" -- a caller
    that needs to tell them apart reads ``mapping`` directly. Raises ``KeyError``
    naming the described channels if ``channel`` is not one of them, since
    returning ``None`` for a typo'd channel name would read as "stationary".

    Takes the mapping rather than a path so a caller resolving several channels
    of one file parses it once.
    """
    try:
        group = mapping[tuple(channel)]
    except KeyError:
        raise KeyError(
            f"{channel!r} has no wiring description in this file; "
            f"described channels: {sorted(f'{s}:{c}' for s, c in mapping)}"
        ) from None
    return None if group in (STATIONARY, UNMAPPED) else group


def motion_group_for_channel(fn, scope, chan):
    """The motion group that recorded one channel, or ``None`` if the file does not say.

    Convenience wrapper over :func:`probe_channel_map` + :func:`moving_group` for
    a caller resolving a single channel; resolving several channels of one file
    should build the mapping once and call :func:`moving_group`.
    """
    return moving_group(probe_channel_map(fn), (scope, chan))
