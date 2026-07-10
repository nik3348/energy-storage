"""Uniform-price merit-order clearing."""

import numpy as np

from energy_storage.market.generators import Offer


def clear_hour(
    offers: list[Offer], demand_mw: float, price_cap: float, price_floor: float
) -> tuple[float, list[tuple[Offer, float]]]:
    """Clear one hour. Returns (price, [(offer, dispatched_mw)]).

    Offers are stacked cheapest-first against inelastic demand; the clearing
    price is the marginal offer's bid. If supply falls short, price is the
    cap (value of lost load).
    """
    offers = sorted(offers, key=lambda o: o.price)
    dispatch = []
    remaining = demand_mw
    price = price_cap
    for offer in offers:
        if remaining <= 1e-9:
            break
        taken = min(offer.quantity_mw, remaining)
        dispatch.append((offer, taken))
        remaining -= taken
        price = offer.price
    if remaining > 1e-9:
        price = price_cap
    return float(np.clip(price, price_floor, price_cap)), dispatch
