"""Rough annual energy-cost estimate for a detected leak.

A patch of envelope that reads dT warmer than its surroundings is shedding roughly
h * A * dT extra watts to the outside air, where h is the combined convective + radiative
exterior surface coefficient. Multiplying by heating hours gives kWh per year.
This is an order-of-magnitude estimate for prioritisation, not an engineering audit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from heatsplat.detect import Leak


@dataclass(frozen=True)
class CostModel:
    surface_coefficient_w_m2k: float = 12.0
    heating_hours_per_year: float = 2500.0
    price_per_kwh: float = 0.15
    currency: str = "USD"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CostEstimate:
    watts: float
    kwh_per_year: float
    cost_per_year: float
    currency: str

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_cost(leak: Leak, model: CostModel = CostModel()) -> CostEstimate:
    watts = model.surface_coefficient_w_m2k * leak.area_m2 * leak.mean_delta_c
    kwh = watts * model.heating_hours_per_year / 1000.0
    return CostEstimate(
        watts=round(watts, 1),
        kwh_per_year=round(kwh, 1),
        cost_per_year=round(kwh * model.price_per_kwh, 2),
        currency=model.currency,
    )
