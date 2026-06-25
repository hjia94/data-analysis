# -*- coding: utf-8 -*-
"""Back-compat shim (Step 2 of the reorg).

The Phantom ``.cine`` reader moved to ``data_analysis.io.cine``. This module
re-exports it unchanged so existing ``from read_cine import ...`` code (resolved
via the ``object_tracking/``-on-``sys.path`` append) keeps working during the
transition. New code should use ``from data_analysis.io.cine import ...``.

Retired once experiment scripts move to the canonical import.
"""

from data_analysis.io.cine import *  # noqa: F401,F403
