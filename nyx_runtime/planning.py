"""Dates de livraison et origine commune des prévisions J+1 et J+2."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

ZONES = ("FR", "DE", "BE", "NL")


def make_plan(first_day: str | None = None, as_of: str | None = None) -> dict:
    """Construire deux jours civils depuis la même coupure de 08 h à Paris."""
    paris = ZoneInfo("Europe/Paris")
    now = datetime.now(timezone.utc)
    origin = datetime.fromisoformat(as_of.replace("Z", "+00:00")) if as_of else None
    if origin is not None and (origin.tzinfo is None or origin.utcoffset() is None):
        raise ValueError("--as-of exige un fuseau ou un décalage UTC explicite.")
    if first_day:
        day = date.fromisoformat(first_day)
        if day.isoformat() != first_day:
            raise ValueError("--first-day exige YYYY-MM-DD.")
    else:
        day = (origin or now).astimezone(paris).date() + timedelta(days=1)
    expected = datetime.combine(day - timedelta(days=1), time(8), paris).astimezone(timezone.utc)
    if origin is not None and origin.astimezone(timezone.utc) != expected:
        raise ValueError("La recette exige exactement J 08:00 Europe/Paris pour les deux horizons.")
    if expected > now:
        raise ValueError("La coupure demandée n'est pas encore atteinte (08 h Europe/Paris).")
    deliveries = []
    for horizon in (1, 2):
        delivery = day + timedelta(days=horizon - 1)
        start = datetime.combine(delivery, time(), paris).astimezone(timezone.utc)
        end = datetime.combine(delivery + timedelta(days=1), time(), paris).astimezone(timezone.utc)
        deliveries.append({
            "horizon_day": horizon,
            "delivery_day": delivery.isoformat(),
            "hours_per_country": int((end - start).total_seconds() / 3600),
        })
    return {
        "protocol": "nyx_aws_daily_refit_v1", "first_day": day.isoformat(),
        "forecast_origin_utc": expected.isoformat(), "countries": list(ZONES),
        "deliveries": deliveries, "qualified_for_production": False,
        "model": "amazon/chronos-2", "device": "cpu",
    }
