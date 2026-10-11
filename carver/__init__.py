"""Robert Carver-style CTA / systematic-trend system, applied to a single ETF.

The package implements the staged pipeline described in
``notes/carver_cta_framework.md``:

    vol  ->  rules (EWMAC)  ->  scale & cap  ->  combine (+FDM)  ->  position sizing

See ``carver.system.CarverSystem`` for the entry point.
"""

BUSINESS_DAYS_IN_YEAR = 256          # Carver's convention
ROOT_BDAYS_INYEAR = BUSINESS_DAYS_IN_YEAR ** 0.5   # ~16

from .system import CarverSystem, SystemConfig   # noqa: E402  (needs constants above)

__all__ = ["CarverSystem", "SystemConfig", "BUSINESS_DAYS_IN_YEAR", "ROOT_BDAYS_INYEAR"]
