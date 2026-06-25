# -*- coding: utf-8 -*-
"""Back-compat shim (Step 2 of the reorg).

Moved to ``data_analysis.io.network_analyzer``. Re-exported here unchanged so
existing imports keep working during the transition. New code should use
``from data_analysis.io.network_analyzer import read_NA_data``.
"""

from data_analysis.io.network_analyzer import *  # noqa: F401,F403
