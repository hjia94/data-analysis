# -*- coding: utf-8 -*-
"""Object tracking from high-speed-camera (.cine) video (was object_tracking/).

Analysis half of the former top-level ``object_tracking/`` folder, moved into the
package in Step 2 of the reorg. The ``.cine`` file *reader* lives separately under
``data_analysis.io.cine`` (grouped by data shape); this subpackage holds the
trajectory tracking / calibration analysis that consumes it.
"""
