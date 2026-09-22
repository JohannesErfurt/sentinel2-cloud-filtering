"""Pixel-level agreement metrics between two boolean masks.

Used by the B2.2 threshold sweep and, later, by the cross-group comparison
module (E1, ``pipeline/compare.py``) -- kept here once rather than duplicated,
since both need exactly the same four numbers against the same kind of input.

Nothing here treats either mask as ground truth. Whichever mask is passed as
``reference`` is only that: the thing precision and recall are measured
against, almost always ESA's mask, which SPEC.md is emphatic is a baseline,
not the truth (§3.2, §5.1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Agreement:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def iou(self) -> float:
        denom = self.true_positive + self.false_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def pixel_agreement(self) -> float:
        total = self.true_positive + self.false_positive + self.false_negative + self.true_negative
        return (self.true_positive + self.true_negative) / total if total else 0.0

    def as_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "iou": self.iou,
            "pixel_agreement": self.pixel_agreement,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
        }


def agreement(predicted: np.ndarray, reference: np.ndarray) -> Agreement:
    """Precision, recall, F1 and IoU of ``predicted`` against ``reference``.

    Both must be boolean and the same shape. Precision, recall and F1 are
    reported separately (SPEC.md 5.1): F1 alone hides the trade-off a
    threshold sweep is all about.
    """
    if predicted.dtype != np.bool_ or reference.dtype != np.bool_:
        raise ValueError(
            "expected boolean masks, got %s and %s" % (predicted.dtype, reference.dtype)
        )
    if predicted.shape != reference.shape:
        raise ValueError("shape mismatch: %s vs %s" % (predicted.shape, reference.shape))

    tp = int(np.count_nonzero(predicted & reference))
    fp = int(np.count_nonzero(predicted & ~reference))
    fn = int(np.count_nonzero(~predicted & reference))
    tn = int(np.count_nonzero(~predicted & ~reference))
    return Agreement(tp, fp, fn, tn)


__all__ = ["Agreement", "agreement"]
