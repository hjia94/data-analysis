"""Single resolver for where generated artifacts go.

Generated artifacts (figures, processed ``.npz``, etc.) live **outside the repo**
by default. One function owns that decision; every writer routes through it, so
the output location is one setting rather than scattered literals.
"""

from __future__ import annotations

import os
from pathlib import Path


def output_root(explicit: str | os.PathLike | None = None) -> Path:
    """Base directory for ALL generated artifacts. Never inside the repo by default.

    Resolution order (first that is set wins):
      1. ``explicit`` argument (caller override, e.g. per-run)
      2. ``$DATA_ANALYSIS_OUTPUT`` env var      -- the main knob
      3. fallback: ``~/data-analysis-output``   (outside the repo tree)

    The resolved directory is created if it does not exist.
    """
    base = explicit or os.environ.get("DATA_ANALYSIS_OUTPUT") or (Path.home() / "data-analysis-output")
    p = Path(base).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def output_path(*parts, explicit: str | os.PathLike | None = None) -> Path:
    """Join ``parts`` under the output root and ensure the parent dir exists.

    Example::

        fig.savefig(output_path("figures", "xray", f"{run}.png"))
        np.savez(output_path("processed", f"{run}-mach.npz"), ...)
    """
    p = output_root(explicit).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
