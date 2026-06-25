# -*- coding: utf-8 -*-
"""Back-compat shim (Step 2 of the reorg).

Moved to ``data_analysis.tracking.evaluate_freefall_accuracy``. Re-exported here
unchanged so existing imports keep working during the transition. New code should
use ``from data_analysis.tracking.evaluate_freefall_accuracy import ...``.
"""

from data_analysis.tracking.evaluate_freefall_accuracy import *  # noqa: F401,F403
