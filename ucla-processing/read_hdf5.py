"""Back-compat shim (Step 3 of the reorg).

The legacy 2018-2020 process-plasma reader moved to
``data_analysis.io._backends.legacy_2018``. This module re-exports it unchanged
so existing ``import read_hdf5`` code keeps working during the transition. New
code should use the unified ``data_analysis.io.lapd_hdf5.open_lapd`` or import
the backend directly.

Retired in Step 4 once experiment scripts move to the canonical import.
"""

from data_analysis.io._backends.legacy_2018 import *  # noqa: F401,F403
