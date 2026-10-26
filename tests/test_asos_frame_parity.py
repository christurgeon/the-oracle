"""Parity test: the LST climate-day frame must be identical between the runtime
client (``python/data/asos.py``) and the climatology builder
(``scripts/build_asos_climatology.py``).

A divergence between the fit frame (builder) and the serve frame (runtime
``max_so_far``) silently corrupts the running extremum ``M`` relative to the
climatology it is looked up against (train/serve skew). This test pins them.

Run with: uv run pytest tests/test_asos_frame_parity.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python.data.asos import CITY_STD_OFFSET_HOURS as RUNTIME_OFFSETS  # noqa: E402
from scripts.build_asos_climatology import (  # noqa: E402
    CITY_STD_OFFSET_HOURS as BUILDER_OFFSETS,
)


def test_std_offset_maps_match_between_runtime_and_builder():
    """The city→standard-offset maps in both files must be identical.

    Same keys (same cities) and same offset values; otherwise the fit frame and
    the serve frame disagree and M is computed against a climatology fit on a
    different climate-day boundary.
    """
    assert RUNTIME_OFFSETS == BUILDER_OFFSETS, (
        "asos.py CITY_STD_OFFSET_HOURS != build_asos_climatology.py "
        f"CITY_STD_OFFSET_HOURS: runtime={RUNTIME_OFFSETS} builder={BUILDER_OFFSETS}"
    )
