"""Cloud filtering for Sentinel-2 L1C imagery.

See SPEC.md for the plan this package implements. The foundations (F1-F10) are
independent of any cloud detector: they read the product, cut the 20x20 grid of
549-pixel tiles, apply the 30 % rule, and write the deliverables. Detectors are
registered separately in ``pipeline.detectors``.
"""

__version__ = "0.1.0"
