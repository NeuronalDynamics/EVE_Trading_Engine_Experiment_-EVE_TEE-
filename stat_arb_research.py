# stat_arb_research.py

import glob
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import coint   # optional: cointegration test


HISTORY_DIR = "output"   # where main.py writes history_*_6m.csv


# -------------------------------------------------------------------
# Load all your history CSVs into a single "panel" DataFrame
#   index: date
#   columns: type_id (int)
#   values: region average price (The Forge) for that day
# -------------------------------------------------------------------

def load_price_panel(history_dir: str = HISTORY_DIR) -> pd.DataFrame:
    """
    Reads all history_*_6m.csv files and combines them into a price panel.
    Each column is one type_id, each row is a date.
    """
    paths = glob.glob(os.path.join(history_dir, "history_*_6m.csv"))
    series_list = []

    for path in paths:
        base = os.path.basename(path)  # e.g. 'history_27459_legion_mjolnir_auto..._6m.csv'
        parts = base.split("_")
        if len(parts) < 3:
            continue

        try:
            type_id = int(parts[1])
        except ValueError:
            continue

        df = pd.read_csv(path, parse_dates=["date"])
        if "average" not in df.columns:
            continue

        df = df[["date", "average"]].dropna()
        df.sort_values("date", inplace=True)

        s = df.set_index("date")["average"].rename(type_id)
        series_list.append(s)

    if not series_list:
        raise RuntimeError("No history_*_6m.csv files found in output/")

    panel = pd.concat(series_list, axis=1).sort_index()
    return panel


# -------------------------------------------------------------------
# 2. Simple single‑asset stat‑arb:
#    "Which items are far from their own 30‑day mean?"
# -------------------------------------------------------------------

def mean_reversion_screen(
    prices: pd.DataFrame,
    window: int = 30,
    min_obs: int = 20,
) -> Tuple[pd.Timestamp, pd.Series]:
    """
    For each item (column), compute a rolling mean/std and get
    the z-score of the last day in the sample.

    Returns:
        last_date, z_scores_series  (sorted: low -> high)
    """
    # Handle zeros (log(0) is bad) by replacing 0 with NaN
    prices = prices.replace(0, np.nan)

    roll_mean = prices.rolling(window=window, min_periods=min_obs).mean()
    roll_std = prices.rolling(window=window, min_periods=min_obs).std()

    z = (prices - roll_mean) / roll_std

    # Last date with any data
    last_date = z.dropna(how="all").index.max()
    last_row = z.loc[last_date]

    # Drop items where std was zero or NaN
    valid = roll_std.loc[last_date].fillna(0) > 0
    last_row = last_row[valid].dropna()

    # Sort: low -> high. Negative = "cheap vs its own history", positive = "expensive".
    last_row = last_row.sort_values()

    return last_date, last_row


def plot_item_with_band(prices: pd.DataFrame, type_id: int, window: int = 30) -> None:
    """
    Plot one item with its rolling mean and +/- 2 std bands.
    """
    s = prices[type_id].dropna()

    roll_mean = s.rolling(window).mean()
    roll_std = s.rolling(window).std()

    upper = roll_mean + 2 * roll_std
    lower = roll_mean - 2 * roll_std

    plt.figure(figsize=(10, 5))
    plt.plot(s.index, s.values, label=f"Price {type_id}")
    plt.plot(roll_mean.index, roll_mean.values, label="Rolling mean", linestyle="--")
    plt.fill_between(
        s.index,
        lower.values,
        upper.values,
        alpha=0.2,
        label="±2σ band",
        step="mid",
    )
    plt.title(f"Type {type_id} – price & ±2σ band")
    plt.xlabel("Date")
    plt.ylabel("Price (ISK)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 3. Pairs / stat‑arb between items:
#    "Pick two related items, check if their price spread mean‑reverts"
# -------------------------------------------------------------------

def compute_pair_spread(prices: pd.DataFrame, type_a: int, type_b: int) -> pd.DataFrame:
    """
    Simple pairs model:
      y = log(price_a), x = log(price_b)
      fit y ~ alpha + beta * x (via OLS)
      spread_t = y_t - beta * x_t

    Returns DataFrame with columns: ['spread', 'z']
    """
    pair = prices[[type_a, type_b]].dropna()
    if len(pair) < 50:
        raise ValueError("Not enough overlapping observations for this pair")

    log_a = np.log(pair[type_a])
    log_b = np.log(pair[type_b])

    # Simple OLS slope: beta = cov / var
    beta = np.cov(log_b, log_a)[0, 1] / np.var(log_b)

    spread = log_a - beta * log_b
    spread_mean = spread.mean()
    spread_std = spread.std()

    z = (spread - spread_mean) / spread_std

    out = pd.DataFrame({"spread": spread, "z": z})
    return out


def plot_pair_spread(spread_df: pd.DataFrame, type_a: int, type_b: int) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(spread_df.index, spread_df["z"], label="Spread z-score")
    plt.axhline(0, color="black", linewidth=1)
    plt.axhline( 2, linestyle="--")
    plt.axhline(-2, linestyle="--")
    plt.title(f"Pair spread z-score: {type_a} vs {type_b}")
    plt.xlabel("Date")
    plt.ylabel("Z-score")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 4. Example “research flow”
# -------------------------------------------------------------------

def main():
    print("Loading panel from output/history_*_6m.csv ...")
    prices = load_price_panel()
    print(f"Panel shape: {prices.shape[0]} days × {prices.shape[1]} items")

    # 4.1 Single-asset mean reversion
    last_date, z = mean_reversion_screen(prices, window=30, min_obs=20)
    print(f"\nMean-reversion z-scores on {last_date.date()}:")
    print("Most underpriced (z << 0):")
    print(z.head(10))
    print("\nMost overpriced (z >> 0):")
    print(z.tail(10))

    # Example: plot one “extreme” item
    extreme_cheap = z.index[0]
    extreme_expensive = z.index[-1]
    print(f"\nPlotting extreme cheap item {extreme_cheap} ...")
    plot_item_with_band(prices, extreme_cheap, window=30)
    print(f"Plotting extreme expensive item {extreme_expensive} ...")
    plot_item_with_band(prices, extreme_expensive, window=30)

    # 4.2 Pairs example: find a highly correlated pair with enough overlap
    corr = prices.corr()
    np.fill_diagonal(corr.values, np.nan)

    # Flatten to (type_a, type_b) -> corr, sorted descending
    pairs = corr.unstack().dropna().sort_values(ascending=False)

    chosen_pair = None
    chosen_spread_df = None

    for (a, b), cval in pairs.items():
        # skip duplicates (corr matrix is symmetric)
        if a >= b:
            continue
        try:
            spread_df = compute_pair_spread(prices, a, b)
        except ValueError:
            # Not enough overlapping observations; try next pair
            continue

        # We found a pair with enough data
        chosen_pair = (a, b)
        chosen_spread_df = spread_df
        print(
            f"\nChosen correlated pair: {a} & {b} "
            f"(corr={cval:.3f}, n={len(spread_df)})"
        )
        break

    if chosen_spread_df is None:
        print("Could not find any pair with enough overlapping observations.")
        return

    print(chosen_spread_df.tail())
    plot_pair_spread(chosen_spread_df, chosen_pair[0], chosen_pair[1])


if __name__ == "__main__":
    main()
