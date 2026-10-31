import numpy as np
import pytest

from heatsplat.cost import CostModel, estimate_cost
from heatsplat.detect import Leak


def _leak(area: float, delta: float) -> Leak:
    return Leak("L1", 1, (0, 0, 0), (0, 0, 0), (1, 1, 0), 10, delta, delta, area, delta * area, "x", np.arange(10))


def test_cost_scales_with_area_and_delta():
    est = estimate_cost(_leak(1.0, 4.0), CostModel(12.0, 2500.0, 0.15))
    assert est.watts == pytest.approx(48.0)
    assert est.kwh_per_year == pytest.approx(120.0)
    assert est.cost_per_year == pytest.approx(18.0)
    assert est.currency == "USD"


def test_zero_area_costs_nothing():
    assert estimate_cost(_leak(0.0, 5.0)).cost_per_year == 0.0
