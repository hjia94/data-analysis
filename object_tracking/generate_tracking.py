# -*- coding: utf-8 -*-
"""Back-compat shim (Step 2 of the reorg).

Moved to ``data_analysis.tracking.generate_tracking``. Re-exported here unchanged
so existing imports keep working during the transition -- both the package form
``from object_tracking.generate_tracking import count_y_passes`` and the flat
``from generate_tracking import ...`` (via the ``object_tracking/``-on-``sys.path``
append). New code should use ``from data_analysis.tracking.generate_tracking import ...``.
"""

from data_analysis.tracking.generate_tracking import *  # noqa: F401,F403
