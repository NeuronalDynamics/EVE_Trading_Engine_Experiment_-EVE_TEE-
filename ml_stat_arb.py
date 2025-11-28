# ml_stat_arb.py

import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


HISTORY_DIR = "output"   # where main.py writes history_*_6m.csv


# -------------------------------------------------------------------
# 1) Load all price histories into a panel: date × type_id
# -------------------------------------------------------------------

def load_price_panel(history_dir: str = HISTORY_DIR) -> pd.DataFrame:
    """
    Reads all history_*_6m.csv files and combines them into a price panel.
    Each column is one type_id, each row is a date, values = average price.
    """
    paths = glob.glob(os.path.join(history_dir, "history_*_6m.csv"))
    series_list: List[pd.Series] = []

    for path in paths:
        base = os.path.basename(path)
        parts = base.split("_")
        if len(parts) < 3:
            continue

        # filename pattern: history_<type_id>_<slug>_6m.csv
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
# 2) Build ML features & target
# -------------------------------------------------------------------

def build_feature_table(
    prices: pd.DataFrame,
    lookback_short: int = 5,
    lookback_long: int = 20,
) -> pd.DataFrame:
    """
    Create a long-format table with one row per (date, type_id), containing:
      - price
      - 1d, 5d, 20d returns
      - 20d volatility
      - z-score of price vs 20d mean
    Target = next-day return (1d forward).
    """
    # Daily simple returns
    returns = prices.pct_change()

    frames: List[pd.DataFrame] = []

    for type_id in prices.columns:
        s_price = prices[type_id]
        s_ret = returns[type_id]

        df = pd.DataFrame(index=prices.index)
        df["type_id"] = type_id
        df["price"] = s_price

        # Past returns
        df["ret_1d"] = s_ret
        df["ret_5d"] = s_price.pct_change(lookback_short)
        df["ret_20d"] = s_price.pct_change(lookback_long)

        # Volatility
        df["vol_20d"] = s_ret.rolling(lookback_long).std()

        # Rolling mean & std for price
        roll_mean = s_price.rolling(lookback_long).mean()
        roll_std = s_price.rolling(lookback_long).std()

        df["z_price_20d"] = (s_price - roll_mean) / roll_std

        # Target: next-day return (shift -1 so that today's features predict tomorrow)
        df["target_ret_1d"] = df["ret_1d"].shift(-1)

        frames.append(df)

    data = pd.concat(frames)
    data = data.reset_index().rename(columns={"index": "date"})

    # Drop rows with any NaNs in features or target
    feature_cols = [
        "price",
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "vol_20d",
        "z_price_20d",
    ]
    data = data.dropna(subset=feature_cols + ["target_ret_1d"])

    # Optional: clip extreme returns to reduce outlier influence
    data["target_ret_1d"] = data["target_ret_1d"].clip(-0.5, 0.5)

    return data


# -------------------------------------------------------------------
# 3) Train/test split by time and fit a ML model
# -------------------------------------------------------------------

def train_test_split_time(data: pd.DataFrame, train_frac: float = 0.7):
    """
    Split by date to avoid look-ahead bias.
    """
    dates = sorted(data["date"].unique())
    split_index = int(len(dates) * train_frac)
    split_date = dates[split_index]

    train_mask = data["date"] < split_date
    test_mask = data["date"] >= split_date

    feature_cols = [
        "price",
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "vol_20d",
        "z_price_20d",
    ]
    target_col = "target_ret_1d"

    X_train = data.loc[train_mask, feature_cols]
    y_train = data.loc[train_mask, target_col]

    X_test = data.loc[test_mask, feature_cols]
    y_test = data.loc[test_mask, target_col]

    test_meta = data.loc[test_mask, ["date", "type_id"]].copy()

    return (X_train, y_train, X_test, y_test, test_meta)


def fit_random_forest(X_train, y_train) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


# -------------------------------------------------------------------
# 4) Evaluate and build a simple long/short strategy
# -------------------------------------------------------------------

def evaluate_predictions(
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> None:
    r2 = r2_score(y_test, y_pred)
    print(f"Test R^2: {r2:.4f}")

    # Also show basic correlation
    corr = np.corrcoef(y_test, y_pred)[0, 1]
    print(f"Test Pearson corr(y_true, y_pred): {corr:.4f}")


def build_test_frame(
    test_meta: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    df = test_meta.copy()
    df["true_ret"] = y_test.values
    df["pred_ret"] = y_pred
    return df


def backtest_long_short(
    df: pd.DataFrame,
    top_q: float = 0.1,
    bottom_q: float = 0.1,
) -> pd.DataFrame:
    """
    Each day:
      - long top_q quantile of predicted returns
      - short bottom_q quantile
    PnL = avg(long true_ret) - avg(short true_ret)
    (No fees/slippage here; add those later.)
    """
    results = []

    for date, g in df.groupby("date"):
        if len(g) < 50:
            continue

        q_low = g["pred_ret"].quantile(bottom_q)
        q_high = g["pred_ret"].quantile(1 - top_q)

        longs = g[g["pred_ret"] >= q_high]
        shorts = g[g["pred_ret"] <= q_low]

        if longs.empty or shorts.empty:
            continue

        long_ret = longs["true_ret"].mean()
        short_ret = shorts["true_ret"].mean()

        daily_pnl = long_ret - short_ret
        results.append({"date": date, "daily_pnl": daily_pnl})

    if not results:
        raise RuntimeError("No days with enough data for backtest.")

    pnl = pd.DataFrame(results).sort_values("date").set_index("date")
    pnl["cum_return"] = (1 + pnl["daily_pnl"]).cumprod()

    return pnl


def plot_equity_curve(pnl: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(pnl.index, pnl["cum_return"])
    plt.title("ML long-short strategy – cumulative return (no fees)")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalised)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 5) Main research flow
# -------------------------------------------------------------------

def main():
    print("Loading panel...")
    prices = load_price_panel()
    print(f"Panel shape: {prices.shape[0]} days × {prices.shape[1]} items")

    print("Building feature table...")
    data = build_feature_table(prices)
    print(f"Total samples (date × item): {len(data)}")

    X_train, y_train, X_test, y_test, test_meta = train_test_split_time(data)

    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    print("Fitting Random Forest...")
    model = fit_random_forest(X_train, y_train)

    print("Predicting on test set...")
    y_pred = model.predict(X_test)

    print("Evaluating predictions...")
    evaluate_predictions(y_test, y_pred)

    print("Building test frame and running long/short backtest...")
    test_df = build_test_frame(test_meta, y_test, y_pred)
    pnl = backtest_long_short(test_df, top_q=0.1, bottom_q=0.1)
    print(pnl.tail())

    plot_equity_curve(pnl)


if __name__ == "__main__":
    main()
