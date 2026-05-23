"""Generate synthetic SaaS event data for local development and testing.

Produces three CSV files in the same directory:
  - users.csv    : one row per user with acquisition metadata
  - events.csv   : behavioural events with power-law frequency distribution
  - orders.csv   : subscription / payment events

All outputs are deterministic given random.seed(42).

Usage:
    python data/synthetic/generate_synthetic_data.py
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent
N_USERS = 5_000
SIM_START = pd.Timestamp("2024-01-01")
SIM_END = pd.Timestamp("2024-12-31")

ACQUISITION_CHANNELS = {
    "organic_search": {"weight": 0.35, "conversion_rate": 0.18},
    "paid_search":    {"weight": 0.20, "conversion_rate": 0.28},
    "direct":         {"weight": 0.15, "conversion_rate": 0.22},
    "referral":       {"weight": 0.15, "conversion_rate": 0.35},
    "social":         {"weight": 0.10, "conversion_rate": 0.12},
    "email":          {"weight": 0.05, "conversion_rate": 0.40},
}

EVENT_TYPES = [
    "page_view",
    "signup",
    "login",
    "feature_used",
    "dashboard_viewed",
    "report_created",
    "invite_sent",
    "settings_changed",
    "export_downloaded",
    "upgrade_clicked",
]

PLANS = ["free", "starter", "growth", "enterprise"]
PLAN_WEIGHTS = [0.50, 0.25, 0.18, 0.07]
MONTHLY_CHURN_RATE = 0.05  # ~5% monthly churn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_timestamp(start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
    delta = int((end - start).total_seconds())
    return start + pd.Timedelta(seconds=random.randint(0, delta))


def _signup_date_distribution(n: int) -> list[pd.Timestamp]:
    """Return n signup dates with a slight growth curve across SIM period."""
    days = (SIM_END - SIM_START).days
    # Exponential growth bias: more signups later in the year
    weights = np.exp(np.linspace(0, 1.5, days))
    weights /= weights.sum()
    chosen_days = np.random.choice(days, size=n, p=weights)
    return [SIM_START + pd.Timedelta(days=int(d)) for d in chosen_days]


def _did_churn(signup: pd.Timestamp, monthly_rate: float = MONTHLY_CHURN_RATE) -> pd.Timestamp | None:
    """Return churn date or None if user is still active at SIM_END."""
    current = signup
    while current < SIM_END:
        if random.random() < monthly_rate:
            churn_day = random.randint(1, 30)
            churn_date = current + pd.Timedelta(days=churn_day)
            return min(churn_date, SIM_END)
        current += pd.Timedelta(days=30)
    return None


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def generate_users() -> pd.DataFrame:
    """Generate the users table."""
    logger.info("Generating %d users ...", N_USERS)

    channels = list(ACQUISITION_CHANNELS.keys())
    channel_weights = [ACQUISITION_CHANNELS[c]["weight"] for c in channels]

    signup_dates = _signup_date_distribution(N_USERS)

    rows = []
    for user_id in range(1, N_USERS + 1):
        signup = signup_dates[user_id - 1]
        channel = random.choices(channels, weights=channel_weights)[0]
        conv_rate = ACQUISITION_CHANNELS[channel]["conversion_rate"]
        converted = random.random() < conv_rate
        plan = random.choices(PLANS, weights=PLAN_WEIGHTS)[0] if converted else "free"
        churn_date = _did_churn(signup) if converted else None

        rows.append(
            {
                "user_id": f"user_{user_id:05d}",
                "signup_at": signup,
                "acquisition_channel": channel,
                "converted": converted,
                "plan": plan,
                "churn_at": churn_date,
                "country": random.choices(
                    ["US", "GB", "DE", "FR", "CA", "AU", "IN", "BR"],
                    weights=[40, 12, 8, 7, 7, 5, 12, 9],
                )[0],
            }
        )

    df = pd.DataFrame(rows)
    logger.info("  active users: %d  churned: %d", df["churn_at"].isna().sum(), df["churn_at"].notna().sum())
    return df


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def generate_events(users: pd.DataFrame) -> pd.DataFrame:
    """Generate behavioural events with power-law frequency per user."""
    logger.info("Generating events ...")

    rows = []
    # Power-law: most users generate few events, a few generate many
    event_counts = np.random.zipf(1.8, size=len(users)).clip(1, 500)

    for (_, user), n_events in zip(users.iterrows(), event_counts):
        user_end = user["churn_at"] if pd.notna(user["churn_at"]) else SIM_END
        if user_end <= user["signup_at"]:
            continue

        # Always emit a signup event
        rows.append(
            {
                "event_id": None,  # assigned later
                "user_id": user["user_id"],
                "event_type": "signup",
                "occurred_at": user["signup_at"],
                "properties": "{}",
            }
        )

        for _ in range(int(n_events)):
            ts = _random_timestamp(user["signup_at"], user_end)
            event_type = random.choices(
                EVENT_TYPES,
                weights=[30, 1, 20, 15, 10, 5, 3, 3, 5, 8],
            )[0]
            rows.append(
                {
                    "event_id": None,
                    "user_id": user["user_id"],
                    "event_type": event_type,
                    "occurred_at": ts,
                    "properties": "{}",
                }
            )

    df = pd.DataFrame(rows)
    df["event_id"] = [f"evt_{i:07d}" for i in range(1, len(df) + 1)]
    df = df.sort_values("occurred_at").reset_index(drop=True)
    logger.info("  total events: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


def generate_orders(users: pd.DataFrame) -> pd.DataFrame:
    """Generate subscription / payment rows for converted users."""
    logger.info("Generating orders ...")

    PLAN_PRICES = {"free": 0, "starter": 29, "growth": 99, "enterprise": 499}

    rows = []
    order_id = 1
    converted = users[users["converted"] & (users["plan"] != "free")]

    for _, user in converted.iterrows():
        price = PLAN_PRICES[user["plan"]]
        current = user["signup_at"]
        end = user["churn_at"] if pd.notna(user["churn_at"]) else SIM_END

        while current < end:
            rows.append(
                {
                    "order_id": f"ord_{order_id:06d}",
                    "user_id": user["user_id"],
                    "order_type": "new_mrr" if current == user["signup_at"] else "renewal",
                    "plan": user["plan"],
                    "amount_usd": price,
                    "billed_at": current,
                    "period_start": current,
                    "period_end": current + pd.Timedelta(days=30),
                }
            )
            order_id += 1
            current += pd.Timedelta(days=30)

        # Churn record
        if pd.notna(user["churn_at"]):
            rows.append(
                {
                    "order_id": f"ord_{order_id:06d}",
                    "user_id": user["user_id"],
                    "order_type": "churn",
                    "plan": user["plan"],
                    "amount_usd": 0,
                    "billed_at": user["churn_at"],
                    "period_start": user["churn_at"],
                    "period_end": user["churn_at"],
                }
            )
            order_id += 1

    df = pd.DataFrame(rows)
    logger.info("  total orders: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    users = generate_users()
    events = generate_events(users)
    orders = generate_orders(users)

    users_path = OUTPUT_DIR / "users.csv"
    events_path = OUTPUT_DIR / "events.csv"
    orders_path = OUTPUT_DIR / "orders.csv"

    users.to_csv(users_path, index=False)
    events.to_csv(events_path, index=False)
    orders.to_csv(orders_path, index=False)

    logger.info("Wrote %s (%d rows)", users_path, len(users))
    logger.info("Wrote %s (%d rows)", events_path, len(events))
    logger.info("Wrote %s (%d rows)", orders_path, len(orders))


if __name__ == "__main__":
    main()
