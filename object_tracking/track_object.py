# -*- coding: utf-8 -*-
"""Back-compat shim (Step 2 of the reorg).

Moved to ``data_analysis.tracking.track_object``. Re-exported here unchanged so
existing ``from track_object import ...`` code (resolved via the
``object_tracking/``-on-``sys.path`` append) keeps working during the transition.
New code should use ``from data_analysis.tracking.track_object import ...``.
"""

from data_analysis.tracking.track_object import *  # noqa: F401,F403
