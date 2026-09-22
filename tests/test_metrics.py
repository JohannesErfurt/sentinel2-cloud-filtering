"""precision / recall / F1 / IoU on tiny hand-made masks with known answers."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.metrics import agreement


def _mask(*flags):
    return np.array(flags, dtype=bool)


def test_perfect_agreement():
    m = _mask(True, False, True, False)
    result = agreement(m, m)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.iou == 1.0
    assert result.pixel_agreement == 1.0


def test_known_confusion_matrix():
    # predicted: T T F F  |  reference: T F T F
    predicted = _mask(True, True, False, False)
    reference = _mask(True, False, True, False)
    result = agreement(predicted, reference)
    assert (result.true_positive, result.false_positive, result.false_negative, result.true_negative) == (1, 1, 1, 1)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)
    assert result.iou == pytest.approx(1 / 3)
    assert result.pixel_agreement == pytest.approx(0.5)


def test_no_predicted_positives_gives_zero_precision_not_nan():
    predicted = _mask(False, False, False)
    reference = _mask(True, False, True)
    result = agreement(predicted, reference)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.iou == 0.0


def test_no_reference_positives_gives_zero_recall_not_nan():
    predicted = _mask(True, False)
    reference = _mask(False, False)
    result = agreement(predicted, reference)
    assert result.recall == 0.0
    assert result.precision == 0.0  # 0 true positives / 1 predicted positive


def test_empty_masks_agree_perfectly_by_convention():
    predicted = _mask(False, False)
    reference = _mask(False, False)
    result = agreement(predicted, reference)
    assert result.pixel_agreement == 1.0


def test_rejects_non_boolean_input():
    with pytest.raises(ValueError, match="boolean"):
        agreement(np.array([1, 0]), np.array([True, False]))


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        agreement(np.zeros((2, 2), bool), np.zeros((3, 3), bool))


def test_as_dict_has_every_field():
    result = agreement(_mask(True), _mask(True))
    d = result.as_dict()
    for key in ("precision", "recall", "f1", "iou", "pixel_agreement",
                "true_positive", "false_positive", "false_negative", "true_negative"):
        assert key in d
