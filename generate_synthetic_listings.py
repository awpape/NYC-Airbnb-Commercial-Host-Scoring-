"""
NYC Airbnb Commercial Activity Score — Synthetic Listings Generator

Generates a realistic synthetic batch of Airbnb listings for dashboard
demonstration purposes. No real listing data is exposed in the output.

Method: Rule-based profile generation across four host archetypes:
  - Commercial operator   — high availability, large portfolio, short minimums
  - Occasional host       — moderate availability, single listing
  - Long-term rental      — high minimum nights, moderate availability
  - Dormant listing       — zero availability and zero reviews

Fake host names, listing names, and IDs are attached for dashboard realism.
No real host or listing identifiers carry over into the output.

Author: Alain William Pape
"""

import numpy as np
import pandas as pd

NEIGHBOURHOODS = [
    "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"
]

ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room"]

HOST_FIRST = [
    "James", "Maria", "Wei", "Fatima", "Carlos", "Aisha", "Robert",
    "Sofia", "Marcus", "Elena", "David", "Jennifer", "Raj", "Grace",
    "Thomas", "Linda", "Kevin", "Patricia", "Omar", "Claire"
]

HOST_LAST = [
    "Smith", "Johnson", "Garcia", "Williams", "Martinez", "Chen",
    "Brown", "Lopez", "Wilson", "Anderson", "Taylor", "Thomas",
    "Moore", "Jackson", "Lee", "Harris", "Clark", "Lewis", "Walker", "Hall"
]

LISTING_ADJECTIVES = [
    "Cozy", "Sunny", "Spacious", "Modern", "Charming", "Quiet",
    "Stylish", "Bright", "Comfortable", "Beautiful"
]

LISTING_NOUNS = [
    "Studio", "Apartment", "Loft", "Suite", "Room", "Place",
    "Hideaway", "Retreat", "Flat", "Space"
]


def _rng_state(seed):
    return np.random.default_rng(seed)


def _fake_name(rng):
    return f"{rng.choice(HOST_FIRST)} {rng.choice(HOST_LAST)}"


def _fake_listing_name(rng, neighbourhood):
    adj = rng.choice(LISTING_ADJECTIVES)
    noun = rng.choice(LISTING_NOUNS)
    return f"{adj} {noun} in {neighbourhood}"


def generate_synthetic_listings(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic batch of Airbnb listings across four host archetypes.

    Parameters
    ----------
    n : int
        Total number of listings to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    DataFrame
        Synthetic listings with all standard Inside Airbnb columns,
        plus an archetype label for reference.
    """
    rng = _rng_state(seed)

    # Archetype proportions
    n_commercial = max(1, int(n * 0.20))
    n_dormant = max(1, int(n * 0.10))
    n_longterm = max(1, int(n * 0.15))
    n_casual = n - n_commercial - n_dormant - n_longterm

    records = []

    # Commercial operators — high score expected
    for i in range(n_commercial):
        neighbourhood = rng.choice(NEIGHBOURHOODS)
        host_name = _fake_name(rng)
        records.append({
            "id": 10000 + i,
            "host_id": 90000 + i,
            "host_name": host_name,
            "name": f"{host_name} Properties — {neighbourhood} Unit {i+1}",
            "neighbourhood_group": neighbourhood,
            "room_type": "Entire home/apt",
            "price": int(rng.integers(120, 300)),
            "minimum_nights": int(rng.integers(1, 4)),
            "number_of_reviews": int(rng.integers(40, 200)),
            "reviews_per_month": round(float(rng.uniform(3.0, 8.0)), 2),
            "calculated_host_listings_count": int(rng.integers(8, 25)),
            "availability_365": int(rng.integers(280, 365)),
            "archetype": "Commercial Operator",
        })

    # Casual hosts — low score expected
    for i in range(n_casual):
        neighbourhood = rng.choice(NEIGHBOURHOODS)
        records.append({
            "id": 20000 + i,
            "host_id": 80000 + i,
            "host_name": _fake_name(rng),
            "name": _fake_listing_name(rng, neighbourhood),
            "neighbourhood_group": neighbourhood,
            "room_type": rng.choice(["Entire home/apt", "Private room"], p=[0.5, 0.5]),
            "price": int(rng.integers(60, 140)),
            "minimum_nights": int(rng.integers(1, 7)),
            "number_of_reviews": int(rng.integers(2, 40)),
            "reviews_per_month": round(float(rng.uniform(0.2, 1.5)), 2),
            "calculated_host_listings_count": 1,
            "availability_365": int(rng.integers(30, 150)),
            "archetype": "Casual Host",
        })

    # Long-term rentals — moderate score expected
    for i in range(n_longterm):
        neighbourhood = rng.choice(NEIGHBOURHOODS)
        records.append({
            "id": 30000 + i,
            "host_id": 70000 + i,
            "host_name": _fake_name(rng),
            "name": _fake_listing_name(rng, neighbourhood),
            "neighbourhood_group": neighbourhood,
            "room_type": "Entire home/apt",
            "price": int(rng.integers(80, 180)),
            "minimum_nights": int(rng.integers(30, 90)),
            "number_of_reviews": int(rng.integers(1, 15)),
            "reviews_per_month": round(float(rng.uniform(0.1, 0.5)), 2),
            "calculated_host_listings_count": int(rng.integers(1, 4)),
            "availability_365": int(rng.integers(180, 300)),
            "archetype": "Long-Term Rental",
        })

    # Dormant listings
    for i in range(n_dormant):
        neighbourhood = rng.choice(NEIGHBOURHOODS)
        records.append({
            "id": 40000 + i,
            "host_id": 60000 + i,
            "host_name": _fake_name(rng),
            "name": _fake_listing_name(rng, neighbourhood),
            "neighbourhood_group": neighbourhood,
            "room_type": rng.choice(ROOM_TYPES),
            "price": int(rng.integers(50, 200)),
            "minimum_nights": int(rng.integers(1, 30)),
            "number_of_reviews": 0,
            "reviews_per_month": 0.0,
            "calculated_host_listings_count": int(rng.integers(1, 5)),
            "availability_365": 0,
            "archetype": "Dormant",
        })

    df = pd.DataFrame(records)
    return df.sample(frac=1, random_state=int(rng.integers(0, 999999))).reset_index(drop=True)


if __name__ == "__main__":
    listings = generate_synthetic_listings(n=50)
    print(f"Generated {len(listings)} synthetic listings")
    print(listings["archetype"].value_counts())
    print(listings.head(10).to_string())
