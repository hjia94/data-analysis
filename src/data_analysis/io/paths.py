"""Single resolver for where generated artifacts go.

Generated artifacts (figures, processed ``.npz``, etc.) live **outside the repo**
by default. One function owns that decision; every writer routes through it, so
the output location is one setting rather than scattered literals.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Output location -- the one knob.  Edit this to send all generated artifacts
# (figures, processed ``.npz``, etc.) somewhere else.  Must be outside the repo
# tree.  ``~`` is expanded; a per-call ``explicit=`` argument still overrides it.
# ---------------------------------------------------------------------------
OUTPUT_ROOT = Path(r"C:\Users\hjia9\Documents\lapd\data-analysis-output")


def output_root(explicit: str | os.PathLike | None = None) -> Path:
    """Base directory for ALL generated artifacts. Never inside the repo by default.

    Resolution order (first that is set wins):
      1. ``explicit`` argument (caller override, e.g. per-run)
      2. the module-level :data:`OUTPUT_ROOT` constant (edit it at the top of
         this file to change where everything goes)

    The resolved directory is created if it does not exist.
    """
    base = explicit or OUTPUT_ROOT
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
