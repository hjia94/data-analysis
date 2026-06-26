# -*- coding: utf-8 -*-
"""Back-compat shim (Step 2 of the reorg).

Moved to ``data_analysis.plasma.sheffield_thomson`` (and the misspelled
"spetral" corrected). Re-exported here unchanged so existing imports keep
working during the transition. New code should use
``from data_analysis.plasma.sheffield_thomson import SheffieldThomson``.
"""

from data_analysis.plasma.sheffield_thomson import *  # noqa: F401,F403
