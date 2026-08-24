"""Static network definition: 20 charging stations across Egypt's New Administrative Capital.

Coordinates are approximate real-world positions of the districts they sit in
(the NAC is centred around 30.00 N, 31.72 E).  ``demand_scale`` is the relative
footfall of the site and ``profile`` selects the hourly arrival shape used by the
simulator and, later, as a categorical model feature.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd


@dataclass(frozen=True)
class Station:
    station_id: str
    name: str
    district: str
    profile: str          # government | business | residential | retail | transit | leisure | hotel | education
    lat: float
    lon: float
    n_chargers: int
    charger_kw: float     # rated power per charger
    demand_scale: float   # relative popularity multiplier

    @property
    def capacity_kw(self) -> float:
        return self.n_chargers * self.charger_kw


STATIONS: tuple[Station, ...] = (
    Station("ST01", "Government District Hub",      "Government District",  "government",  30.0086, 31.7395, 12, 60.0, 1.55),
    Station("ST02", "Ministries Plaza",             "Government District",  "government",  30.0121, 31.7462,  8, 50.0, 1.20),
    Station("ST03", "Council of Ministers Garage",  "Government District",  "government",  30.0044, 31.7331,  6, 22.0, 0.85),
    Station("ST04", "Iconic Tower CBD",             "Business District",    "business",    29.9931, 31.7188, 14, 120.0, 1.70),
    Station("ST05", "Financial District North",     "Business District",    "business",    29.9974, 31.7241, 10, 60.0, 1.30),
    Station("ST06", "Central Bank Parking",         "Business District",    "business",    29.9902, 31.7139,  6, 50.0, 0.95),
    Station("ST07", "R2 Residential Loop",          "R2 District",          "residential", 30.0298, 31.6957,  8, 22.0, 1.10),
    Station("ST08", "R3 Community Centre",          "R3 District",          "residential", 30.0412, 31.6822,  6, 22.0, 0.90),
    Station("ST09", "R5 Family Housing",            "R5 District",          "residential", 29.9705, 31.6931,  8, 22.0, 1.05),
    Station("ST10", "R7 Neighbourhood Point",       "R7 District",          "residential", 29.9612, 31.7488,  4, 22.0, 0.65),
    Station("ST11", "Green River Park",             "Green River",          "leisure",     30.0032, 31.7104,  6, 50.0, 0.95),
    Station("ST12", "Downtown Mall Level B1",       "Downtown",             "retail",      29.9985, 31.7318, 16, 60.0, 1.75),
    Station("ST13", "Expo City Gate 3",             "Expo City",            "retail",      29.9761, 31.7602, 10, 60.0, 1.15),
    Station("ST14", "Capital Airport Terminal",     "Capital Airport",      "transit",     30.0879, 31.8590, 12, 120.0, 1.40),
    Station("ST15", "Monorail Interchange",         "Bin Zayed Axis",       "transit",     30.0203, 31.6740, 10, 120.0, 1.35),
    Station("ST16", "Regional Ring Road Rest Stop", "Ring Road",            "transit",     29.9548, 31.6612,  8, 150.0, 1.25),
    Station("ST17", "Al Massa Hotel",               "Diplomatic District",  "hotel",       30.0157, 31.7256,  6, 50.0, 0.80),
    Station("ST18", "Diplomatic Quarter Annex",     "Diplomatic District",  "hotel",       30.0189, 31.7175,  4, 50.0, 0.60),
    Station("ST19", "Knowledge City Campus",        "Knowledge City",       "education",   29.9836, 31.6885,  8, 22.0, 1.00),
    Station("ST20", "Sports City Arena",            "Sports City",          "leisure",     29.9664, 31.7024,  6, 60.0, 0.85),
)

STATION_BY_ID: dict[str, Station] = {s.station_id: s for s in STATIONS}


def stations_frame() -> pd.DataFrame:
    """Station master table (one row per station)."""
    df = pd.DataFrame([asdict(s) for s in STATIONS])
    df["capacity_kw"] = df["n_chargers"] * df["charger_kw"]
    return df


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two WGS84 points."""
    import math

    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearby_stations(lat: float, lon: float, radius_km: float = 8.0, limit: int = 6) -> list[tuple[Station, float]]:
    """Stations within ``radius_km`` of a point, nearest first."""
    scored = [(s, haversine_km(lat, lon, s.lat, s.lon)) for s in STATIONS]
    scored.sort(key=lambda t: t[1])
    hits = [t for t in scored if t[1] <= radius_km][:limit]
    return hits or scored[:limit]


if __name__ == "__main__":
    df = stations_frame()
    print(df.to_string(index=False))
    print(f"\nTotal chargers: {df.n_chargers.sum()}   Installed power: {df.capacity_kw.sum():.0f} kW")
