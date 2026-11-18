"""Patent analysis (Phase 9) — the WoS/Scopus pipeline pointed at a patent register.

The corpus here is a Ukrainian patent register slice (22 225 grants, priority years
2012–2022) plus two international satellites: the WO publications the same families
reached, and the IP5 filings among them.

**Nothing in this package re-implements analysis.** Clustering, trend fitting, the
taxonomy tabulation, the lenses, the report and the commentary all run unchanged — the
work here is to hand them records in the shape they already read, which is why
:func:`~horizon.patents.workbooks.normalize_row` emits the same canonical keys
``horizon.ingest.collection`` does (``year``, ``subject_areas``, ``venue``, ``authors``,
``script``, ``citation_percentile``) alongside the patent-native ones. The analysis layer
never learns it is looking at patents; only the labels in the report say so.

Two things genuinely differ from the bibliographic corpus, and both live here:

* **Three time axes, not one.** A publication has a year. A patent has a priority date
  (when the invention was claimed), a filing date and a grant date, and they sit a median
  2.5 years apart. Which one is "the year" is a measurement decision, not a detail —
  :mod:`horizon.patents.completeness` makes it explicit and defaults to priority.
* **Grant lag truncates a run of cohorts, not just the last one.** The register was cut
  on 2026-08-12, so recent priority years hold only the fraction of their eventual grants
  that have issued so far — 2022 shows 340 against a median of ~2 400, and it is not a
  small year, it is an unfinished one. The bibliographic guard drops one partial final
  cohort; that is not enough here, so completeness is estimated from the register's own
  lag distribution.
"""

from __future__ import annotations

from horizon.patents.completeness import (
    YEAR_AXES,
    complete_years,
    lag_quantiles,
)
from horizon.patents.source import PatentSource
from horizon.patents.workbooks import (
    build_dataset,
    normalize_row,
    read_main,
    read_satellite,
)

__all__ = [
    "PatentSource",
    "YEAR_AXES",
    "build_dataset",
    "complete_years",
    "lag_quantiles",
    "normalize_row",
    "read_main",
    "read_satellite",
]
