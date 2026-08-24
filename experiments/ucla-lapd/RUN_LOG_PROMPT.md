# Run-log generation prompt

Reusable prompt for having an agent catalogue an LAPD campaign into a run log.
Paste the block below at the start of a session, filling in the three
placeholders. Everything outside the block is guidance for *you*, not the agent.

**Fill in:** `<DATA_DIR>`, `<START>`, `<END>` (dates as `YYYY-MM-DD`).

**Optional, saves a discovery pass:** if the campaign has an obvious split — a
configuration change partway through, two probe setups — say so when you paste.
The agent will look regardless.

---

```
Read every HDF5 file in <DATA_DIR> and write a run log at
<DATA_DIR>/RUN_LOG_<START>_to_<END>.md covering the campaign.

Use the data-analysis repo (helpers in src/data_analysis/, venv at
.venv/Scripts/python.exe) to read files and descriptions. Search for an existing
helper before writing anything new. Scratch scripts go in the scratchpad, not
the data directory.

## Work in two phases

Phase 1 — extract. Run the existing extractor; do not write a new one:

    .venv/Scripts/python.exe experiments/ucla-lapd/dump_campaign.py \
        <DATA_DIR> -o <scratchpad>/runs.jsonl

It emits one JSON line per run plus a trailing summary line, and reads metadata
and position arrays only — never waveforms. Run it in the background: a first
pass over a large campaign can exceed the default timeout.

Read the summary line FIRST. `field_coverage` says how many runs produced each
field; a field present in some runs and not others is either a real campaign
change or a gap in the extractor, and you need to know which before you trust
anything downstream. `runs_with_unknown_layout` and `runs_with_read_error` are
runs whose data you have NOT seen — never describe them as if you had.

**If a field you need is missing, extend `dump_campaign.py` and re-run phase 1.**
Do not open HDF5 files ad hoc to patch around a gap — that is how the
intermediate stops being authoritative and the log starts disagreeing with
itself. Bump `SCHEMA_VERSION` when a field's meaning changes.

The extractor observes; it never interprets. It gives you shot ranges, pairwise
range relations, null-row counts, parsed ports and unmatched sets — the raw
material for decisions that are yours to make:

  - **Probe motion.** `motion_pairs` gives `disjoint` / `overlap` / `identical`
    per pair of motion groups. Disjoint tiling ranges mean the probes ran
    sequentially; identical ranges mean they moved together. Both happen.
  - **Padding.** `null_rows` counts rows whose `shot_num` is 0. Before treating
    that as padding, check `null_pos_collides`: if a real measurement sits at
    the null coordinate, the convention is unsafe for that run and you need
    another discriminator.
  - **Dtype drift.** The same attribute may be a string in early runs and an
    integer later. Cast before comparing, and record the drift as a discrepancy.
  - **Acquisition mode.** `probe_voltage_channels` lists the channels carrying an
    applied probe voltage. Non-empty = swept, empty = fixed-bias; the description
    text does not override it (see "Per-run state that varies" below).
  - **Hardware state.** `description_markers.not_used` names hardware blocks
    marked `(NOT USED)`; `description_markers.obstacle` carries the `- Obstacle:`
    line where one is present. Both match one spelling: confirm the campaign
    uses these markers before reading a null as "hardware was out", since a
    campaign writing it differently returns null for every run.

Phase 2 — reason. Do all summarizing, grouping, and cross-run comparison against
the intermediate, not by re-opening HDF5 files. Go back to the raw files only to
resolve specific discrepancies (see below), where you need actual signals rather
than metadata.

## What to extract

Per run: filename, creation time, size, run description, channel descriptions
per scope, declared vs actually-acquired channels, motion groups and their
position grids, shot counts, time base, and whatever diagnostics this campaign
recorded. Do not assume which diagnostics exist — enumerate what is actually in
the files and let the campaign tell you what it was measuring.

Combine descriptions across runs rather than repeating them: state a setting once
where it holds, and note where it CHANGES. The changes are the useful signal.

### Per-run state that varies: give it a COLUMN, not a section

Anything that changes run to run and changes how a run is analysed belongs as a
column in the per-run table — not a prose section, and not repeated in the daily
summary. Two that campaigns get wrong:

  - **Acquisition mode** (fixed-bias Isat vs swept I–V). Decide from the CHANNEL
    LIST: only a sweep digitizes the applied probe voltage, so a swept run pairs
    each current channel with a `V, …` one. Do NOT classify from the description
    — it quotes "IV sweep 200us" even for probes marked `(NOT USED)`, so it says
    "sweep" in nearly every run. Beware plate/electrode voltage monitors, which
    are voltage channels but not probe sweeps.
  - **Hardware in/out** (obstacle, plate, probe swapped out). Recorded only in
    the description prose. Scan every run for negation markers like `(NOT USED)`
    — a run can list a full plate configuration and still have had it withdrawn,
    and a header line may announce the state for only some runs of a span.

Define each column's rule ONCE, where §1 names conventions, in a sentence or
two. The table carries the per-run values; nothing else restates them.

## Structure

1. How to read this document — where each field comes from (which HDF5 path),
   and which fields are ground truth vs operator intent. Storage layout facts an
   analyst needs (shot ordering, raw dtype and scaling, padding conventions).
   Naming conventions used below.
2. Campaign at a glance — the configuration(s) the campaign splits into,
   probe/diagnostic inventory, digitizer channel map, and how the controlled
   parameters evolved run to run.
3. Daily summary — what was done each day and why, in narrative form.
4. Per-run reference table — the spine of the document (see below).
5. Probe-drive to channel map (see below).
6. Per-campaign diagnostics that ran on every shot.
7. Known-good reference runs worth trusting.
8. Analysis cautions.
9. Discrepancies (see below).

Adapt these sections to what the campaign actually is. Drop what does not apply;
add what does. Keep section numbering stable so it can be cross-referenced.

## Probe-drive to channel map

For every run, give a table of which digitizer channels belong to which moving
probe. This is the single most useful thing in the log for later analysis: it is
what lets an analyst pair a position array with the signals recorded at those
positions.

Nothing in the file links them structurally. Both the motion-group name and the
channel description are free text, and the only shared key is the PORT NUMBER:

    motion group  "<Apollo>    p28_Bdot_C11"   -> port 28
    channel       "Bdot_scope_C1 = P28-Bx--X10" -> port 28

The extractor's `join` block has already parsed both sides. Use it rather than
re-deriving the ports.

Two things break a naive port match, and both need judgement rather than code:

  - **A typo on either side.** A channel description carrying a truncated or
    wrong port will not join to its probe. The signature is a non-empty
    `unmatched_motion_groups` AND `unmatched_channels` in the same run.
  - **A port collision.** Two probes can share one port (e.g. east and west
    faces), listed in `contested_ports`. Port alone maps every channel of that
    port to both probes. Disambiguate with the diagnostic word in each name — a
    group named `Bdot` takes the B-field channels, one named `LP` takes the
    Langmuir channels.

Where the join fails for either reason, **leave the mapping cell empty** and
record the cause in the discrepancies section. Do not guess, and do not
substitute a probable answer. A human will read the discrepancies and fill the
cell in later; an empty cell is a correct statement that the file does not say,
whereas a guessed one is indistinguishable from a verified mapping.

Also report, per run, channels whose port matches NO motion group. These are
usually stationary diagnostics — a fixed reference probe that is digitized but
never moved — and are worth naming as such rather than leaving as orphans.

## Discrepancies — the most important section

Flag every disagreement between run description, channel description, and actual
data. Then:

  - Index them BY RUN NUMBER, not by issue type: "Run 07 — scope group misnamed
    X, read Y instead". One table, one row per run-issue, with a "what to do
    instead" column that is directly actionable.
  - Split campaign-wide issues into their own short table above the per-run
    index.
  - Account for every run exactly once: either it has a listed discrepancy or it
    appears in an explicit "no run-specific discrepancies" list.
  - No severity column. These are things to be resolved, not judged.
  - INTENDED changes — configuration switches, parameter scans, retunes — are
    not discrepancies. Document them in sections 2/3 and say so at the top of
    section 9.

Then go back into the raw data and try to RESOLVE them. Measure the signals
rather than reasoning from the descriptions — the data outranks the text. Mark
each as *resolved* / *confirmed* / *supported* / *documentary* and put the
supporting numbers in an evidence subsection.

Be honest about strength of evidence. If a measurement corroborates but does not
prove — within-group scatter comparable to between-group separation, say — mark
it *supported*, not *resolved*, and state why. If the data cannot settle
something, say so explicitly.

**Expect the data to REVERSE your initial reading.** Where a description looks
wrong, first check whether the description was right and your interpretation was
wrong. This is the single most likely way to get the log wrong, and it bites
hardest exactly when a mismatch looks obvious. A description saying a probe did
not move may mean it was parked, not that data is missing; verify by measuring
whether anything actually varies with position before writing it up as an error.

## Style

Written for an AI agent to extract facts accurately, and usable by a reader who
does NOT have the data-analysis repo — describe operations (what to compute,
from which HDF5 path), not the library functions that perform them. No function
names, module paths, or venv paths in the document.

Terse and factual. Tables where the content is tabular. Where an array layout,
unit, or sign convention matters, name it — those are the facts a reader cannot
recover from the file alone.

**State each fact once, at the layer that owns it.** This is the rule that decides
length, and the one most easily lost while editing. Each fact has exactly one
home: a per-run value in the run table, a convention in §1, a discrepancy in §9,
a per-run parameter that varies in the controlled-parameter table. Everything
else references it.

  - Do not re-assert a run's mode, hardware state, grid, or shot count in the
    daily summary, the channel map, the reference runs, or the cautions. The
    daily summary carries WHY a day was run that way; the numbers are in §4.
  - **Delete any section whose content is derivable from the run table.** A
    "groupings by intent" list that just re-sorts the Intent column earns
    nothing. Keep a grouping only where it states something the table cannot —
    a span, an ordering, a shared derived artifact.
  - A caution that repeats a §9 row or a §1 storage fact is noise; §8 is for
    traps that have no other home.
  - Prefer widening the run table to adding a section.

A reader looking for one run's facts should find them in one row, not assembled
from five places.

## Environment notes

  - Write scratch scripts with the Write tool, not shell heredocs. Backticks and
    apostrophes in markdown content get mangled by shell quoting.
  - Start scripts that print document text with
    `sys.stdout.reconfigure(encoding="utf-8")` — section signs and arrows raise
    UnicodeEncodeError on the default Windows codepage otherwise.

## Before you finish

Verify programmatically, not by reading:

  - every run appears exactly once in section 9 (flagged or explicitly clean,
    never both, none missing) — beware range notation like `11`–`14` in a table
    cell, which silently sweeps runs into a group;
  - every section cross-reference resolves to a real heading;
  - no orphaned references to sections you deleted while editing, and no pointer
    to a section for content you have since moved elsewhere;
  - no paragraph appears twice. Rewriting a preamble in place leaves the old
    copy directly above the new one, and it reads as intentional.

Then read the document once, start to finish, asking of each sentence: is this
the one place that owns this fact? Cut what answers no. This catches what the
programmatic checks cannot.

Ask me about anything the files genuinely cannot settle. Do not guess at
physical intent that is not recorded.
```
