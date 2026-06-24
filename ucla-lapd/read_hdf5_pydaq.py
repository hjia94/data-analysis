"""Back-compat shim (Step 3 of the reorg).

The pydaq (Python LAPD_DAQ) reader moved to ``data_analysis.io._backends.pydaq``.
This module re-exports it unchanged so existing ``import read_hdf5_pydaq`` code
keeps working during the transition. New code should use the unified
``data_analysis.io.lapd_hdf5.open_lapd`` or import the backend directly.

Retired in Step 4 once experiment scripts move to the canonical import.
"""

from data_analysis.io._backends.pydaq import *  # noqa: F401,F403
from data_analysis.io._backends.pydaq import __all__  # noqa: F401
