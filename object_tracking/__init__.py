# -*- coding: utf-8 -*-
"""Back-compat shim package (Step 2 of the reorg).

The ``object_tracking/`` analysis modules moved into ``data_analysis.tracking``
and the ``.cine`` reader into ``data_analysis.io.cine``. This package and the
sibling shim modules re-export them unchanged so existing imports keep working
during the transition:

    from object_tracking.generate_tracking import count_y_passes   # package form
    from track_object import track_object                          # flat form
    from read_cine import read_cine                                # flat form

New code should import from ``data_analysis.tracking`` / ``data_analysis.io.cine``.
Retired once experiment scripts move to the canonical imports.
"""
