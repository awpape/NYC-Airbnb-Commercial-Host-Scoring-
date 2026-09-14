"""
NYC Airbnb Commercial Activity Score — Scoring Module

Applies the five-signal behavioral scoring model to a batch of Airbnb
listings and returns a ranked priority list of suspected commercial operators.

The score (0-100) combines five behavioral signals:
  - Portfolio size (30%)  — hosts with many listings score higher
  - Availability (30%)    — 271+ days/year signals hotel-like operation
  - Review cadence (20%)  — high review frequency indicates commercial turnover
  - Minimum nights (10%)  — short minimums signal guest-facing commercial use
  - Room type (10%)       — entire home/apt listings score higher

Tiers:
  High     >= 60  — suspected commercial operator, priority for enforcement
  Moderate >= 40  — worth monitoring
  Low      < 40   — likely casual host
  Dormant         — zero availability and zero reviews

Author: Alain William Pape
"""

import pandas as pd
import numpy as np

# Default weights from the validated model
WEIGHTS = dict(portfolio=30, availability=30, reviews=20, min_nights=10, room=10)
CUTOFF = 60.0

REQUIRED_COLUMNS = [
    "id", "host_id", "host_name", "name", "neighbourhood_group",
    "room_type", "price", "minimum_nights", "number_of_reviews",
    "reviews_per_month", "calculated_host_listings_count", "availability_365"
]


def _sp(c):
    """Portfolio signal: partial credit at 2+, 4+, 11+ listings."""
    return 1.0 if c >= 11 else 0.75 if c >= 4 else 0.45 if c >= 2 else 0.0


def _sa(a):
    """Availability signal: partial credit at 90+, 180+, 271+ days."""
    return 1.0 if a >= 271 else 0.7 if a > 180 else 0.35 if a > 90 else 0.0


def _sm(m):
    """Minimum nights signal: full credit <= 3 nights, partial credit < 30."""
    return 1.0 if m <= 3 else 0.4 if m < 30 else 0.0


def compute_signals(df: pd.DataFrame, q50: float, q75: float) -> pd.DataFrame:
    """
    Compute the five behavioral signals for each listing.

    Parameters
    ----------
    df : DataFrame
        Listings with required Inside Airbnb columns.
    q50 : float
        Median reviews_per_month among active listings (used as lower threshold).
    q75 : float
        75th percentile reviews_per_month (used as upper threshold).

    Returns
    -------
    DataFrame
        Five signal columns, each valued 0.0 / partial / 1.0.
    """
    def _sr(r):
        return 1.0 if r >= q75 else 0.5 if r >= q50 else 0.0

    return pd.DataFrame({
        "portfolio": df["calculated_host_listings_count"].map(_sp),
        "availability": df["availability_365"].map(_sa),
        "reviews": df["reviews_per_month"].fillna(0).map(_sr),
        "min_nights": df["minimum_nights"].map(_sm),
        "room": (df["room_type"] == "Entire home/apt").astype(float),
    }, index=df.index)


def score_listings(
    df: pd.DataFrame,
    weights: dict = WEIGHTS,
    cutoff: float = CUTOFF,
    q50: float = None,
    q75: float = None,
) -> pd.DataFrame:
    """
    Score a batch of Airbnb listings and return a ranked priority queue.

    Parameters
    ----------
    df : DataFrame
        Listings with standard Inside Airbnb columns.
    weights : dict
        Signal weights summing to 100. Default: validated model weights.
    cutoff : float
        Score threshold for High tier. Default: 60.0.
    q50 : float, optional
        Median reviews_per_month. If None, computed from df.
    q75 : float, optional
        75th percentile reviews_per_month. If None, computed from df.

    Returns
    -------
    DataFrame
        Input listings with score, tier, signal columns, and commercial_rank.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    # Compute review cadence thresholds from active listings
    if q50 is None or q75 is None:
        active = df.loc[df["reviews_per_month"] > 0, "reviews_per_month"]
        q50 = float(active.quantile(0.50)) if q50 is None else q50
        q75 = float(active.quantile(0.75)) if q75 is None else q75

    signals = compute_signals(df, q50, q75)
    score = sum(weights[k] * signals[k] for k in weights)

    dormant = (df["availability_365"] == 0) & (df["number_of_reviews"] == 0)
    tier = pd.Series("Low", index=df.index)
    tier[score >= 40] = "Moderate"
    tier[score >= cutoff] = "High"
    tier[dormant] = "Dormant"

    out = df.copy()
    out["score"] = score.round(1)
    out["tier"] = tier
    for col in signals.columns:
        out[f"signal_{col}"] = signals[col]

    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out["commercial_rank"] = out.index + 1
    return out


def get_tier_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table of tier counts and percentages."""
    vc = scored_df["tier"].value_counts()
    total = len(scored_df)
    rows = []
    for tier in ["High", "Moderate", "Low", "Dormant"]:
        n = vc.get(tier, 0)
        rows.append({"Tier": tier, "Count": n, "Pct": round(100 * n / total, 1)})
    return pd.DataFrame(rows)
