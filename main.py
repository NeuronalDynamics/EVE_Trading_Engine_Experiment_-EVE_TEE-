# main.py

'''
from datetime import datetime
import csv

import matplotlib.pyplot as plt

from market_api import InfernoHeavyMissileSinqAPI


def print_full_order_book(book) -> None:
    buy = book["buy"]
    sell = book["sell"]

    print("=== FULL REGION ORDER BOOK: Sinq Laison – Inferno Heavy Missile ===\n")

    print(f"Total BUY orders:  {len(buy)}")
    print(f"Total SELL orders: {len(sell)}\n")

    print("---- BUY ORDERS (highest price first) ----")
    for o in buy:
        print(
            f"BUY  {o['volume_remain']:>10} @ {o['price']:>12,.2f} ISK "
            f"(system_id={o['system_id']}, location_id={o['location_id']})"
        )

    print("\n---- SELL ORDERS (lowest price first) ----")
    for o in sell:
        print(
            f"SELL {o['volume_remain']:>10} @ {o['price']:>12,.2f} ISK "
            f"(system_id={o['system_id']}, location_id={o['location_id']})"
        )

def export_order_book_to_csv(book, filename: str) -> None:
    buy = book["buy"]
    sell = book["sell"]

    fieldnames = [
        "side",          # "buy" or "sell"
        "price",
        "volume_remain",
        "volume_total",
        "system_id",
        "location_id",
        "duration",
        "issued",
        "order_id",
        "min_volume",
        "range",
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for o in buy:
            writer.writerow(
                {
                    "side": "buy",
                    "price": o["price"],
                    "volume_remain": o["volume_remain"],
                    "volume_total": o["volume_total"],
                    "system_id": o["system_id"],
                    "location_id": o["location_id"],
                    "duration": o["duration"],
                    "issued": o["issued"],
                    "order_id": o["order_id"],
                    "min_volume": o["min_volume"],
                    "range": o["range"],
                }
            )

        for o in sell:
            writer.writerow(
                {
                    "side": "sell",
                    "price": o["price"],
                    "volume_remain": o["volume_remain"],
                    "volume_total": o["volume_total"],
                    "system_id": o["system_id"],
                    "location_id": o["location_id"],
                    "duration": o["duration"],
                    "issued": o["issued"],
                    "order_id": o["order_id"],
                    "min_volume": o["min_volume"],
                    "range": o["range"],
                }
            )

    print(f"Order book exported to {filename}")

def export_history_to_csv(history, filename: str) -> None:
    if not history:
        print("No history data to export.")
        return

    fieldnames = ["date", "average", "highest", "lowest", "order_count", "volume"]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "date": row["date"],
                    "average": row["average"],
                    "highest": row["highest"],
                    "lowest": row["lowest"],
                    "order_count": row["order_count"],
                    "volume": row["volume"],
                }
            )

    print(f"CSV exported to {filename}")

def plot_history_from_csv(csv_filename: str, png_filename: str) -> None:
    dates = []
    avg_prices = []
    high_prices = []
    low_prices = []
    volumes = []

    with open(csv_filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # date as datetime
            dates.append(datetime.fromisoformat(row["date"]))
            # convert numeric fields
            avg_prices.append(float(row["average"]))
            high_prices.append(float(row["highest"]))
            low_prices.append(float(row["lowest"]))
            volumes.append(float(row["volume"]))

    if not dates:
        print("No rows in CSV; nothing to plot.")
        return

    fig, ax_price = plt.subplots(figsize=(10, 5))

    # Price lines (left axis)
    ax_price.plot(dates, avg_prices, label="Average", linewidth=1)
    ax_price.plot(dates, high_prices, label="High", linestyle="--", linewidth=1)
    ax_price.plot(dates, low_prices, label="Low", linestyle="--", linewidth=1)

    ax_price.set_xlabel("Date")
    ax_price.set_ylabel("Price (ISK)")
    ax_price.set_title("Inferno Heavy Missile – Sinq Laison – last ~6 months")

    # Volume (right axis)
    ax_vol = ax_price.twinx()
    ax_vol.bar(dates, volumes, alpha=0.2, label="Volume")
    ax_vol.set_ylabel("Volume (units)")
    ax_vol.set_yscale("log")

    # Combine legends from both axes
    lines_1, labels_1 = ax_price.get_legend_handles_labels()
    lines_2, labels_2 = ax_vol.get_legend_handles_labels()
    ax_price.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    fig.tight_layout()
    fig.savefig(png_filename, dpi=150)
    print(f"Figure saved to {png_filename}")

    # Optional: show window
    # plt.show()

def main():
    api = InfernoHeavyMissileSinqAPI(
        user_agent="inferno-hm-sinq-market-tool/1.0 (you@example.com)"
    )

    # ----- REGION ORDER BOOK -----
    book = api.get_sinq_order_book()

    # Show the full order book in the terminal
    print_full_order_book(book)

    # Optionally also export the order book to CSV
    export_order_book_to_csv(book, "inferno_orderbook_sinq.csv")

    # ----- 6-MONTH HISTORY -----
    history_6m = api.get_sinq_history_6m()
    print(f"\n6‑month history rows: {len(history_6m)}")

    if history_6m:
        first = history_6m[0]
        last = history_6m[-1]
        print(
            f"From {first['date']} to {last['date']} "
            f"(latest avg={last['average']:.2f} ISK, "
            f"low={last['lowest']:.2f}, high={last['highest']:.2f}, "
            f"volume={last['volume']})"
        )

    # Export history to CSV
    history_csv = "inferno_history_6m.csv"
    export_history_to_csv(history_6m, history_csv)

    # Plot using the CSV (not the in-memory list)
    history_png = "inferno_history_6m.png"
    plot_history_from_csv(history_csv, history_png)


if __name__ == "__main__":
    main()
'''

# main.py

import csv
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List

import matplotlib.pyplot as plt

from market_api import EveMarketAPI, THE_FORGE_REGION_ID

# ESI base info (must match what you use in market_api.py)
ESI_BASE_URL = "https://esi.evetech.net/latest"
DATASOURCE = "tranquility"

# --- simple caches so we don't hammer /universe/ endpoints more than needed ---

_type_cache: Dict[int, Dict[str, Any]] = {}
_group_cache: Dict[int, Dict[str, Any]] = {}


def slugify(name: str) -> str:
    """Make a safe file name from a type name."""
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


# ---------------------------------------------------------------------------
# 1) Discover all type_ids traded in the region
# ---------------------------------------------------------------------------

def get_region_type_ids(api: EveMarketAPI, region_id: int) -> List[int]:
    """
    Use /markets/{region_id}/types/ to list type IDs with active orders
    in this region. :contentReference[oaicite:6]{index=6}
    """
    type_ids: List[int] = []
    page = 1

    while True:
        resp = api.session.get(
            f"{ESI_BASE_URL}/markets/{region_id}/types/",
            params={"datasource": DATASOURCE, "page": page},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break

        type_ids.extend(data)

        x_pages = resp.headers.get("X-Pages")
        if not x_pages or page >= int(x_pages):
            break

        page += 1

    return type_ids


# ---------------------------------------------------------------------------
# 2) Universe helpers: type -> group -> detect missiles
# ---------------------------------------------------------------------------

def get_type_info(api: EveMarketAPI, type_id: int) -> Dict[str, Any]:
    """
    /universe/types/{type_id}/ – get type name & group_id, cached. :contentReference[oaicite:7]{index=7}
    """
    if type_id in _type_cache:
        return _type_cache[type_id]

    resp = api.session.get(
        f"{ESI_BASE_URL}/universe/types/{type_id}/",
        params={"datasource": DATASOURCE},
        timeout=30,
    )
    resp.raise_for_status()
    info = resp.json()
    _type_cache[type_id] = info
    return info


def get_group_info(api: EveMarketAPI, group_id: int) -> Dict[str, Any]:
    """
    /universe/groups/{group_id}/ – get group name & category_id, cached. :contentReference[oaicite:8]{index=8}
    """
    if group_id in _group_cache:
        return _group_cache[group_id]

    resp = api.session.get(
        f"{ESI_BASE_URL}/universe/groups/{group_id}/",
        params={"datasource": DATASOURCE, "language": "en"},
        timeout=30,
    )
    resp.raise_for_status()
    info = resp.json()
    _group_cache[group_id] = info
    return info


def is_missile_type(api: EveMarketAPI, type_id: int) -> bool:
    """
    Heuristic: treat a type as a missile if:
      - its group belongs to category_id == 8 ("Charge"), and
      - the group name contains "missile", "rocket", or "torpedo". :contentReference[oaicite:9]{index=9}
    """
    type_info = get_type_info(api, type_id)
    group_info = get_group_info(api, type_info["group_id"])

    # Charges category
    if group_info.get("category_id") != 8:
        return False

    gname = group_info.get("name", "").lower()
    return any(word in gname for word in ("missile", "rocket", "torpedo"))


# ---------------------------------------------------------------------------
# 3) CSV + plotting helpers (same idea as before)
# ---------------------------------------------------------------------------

def export_order_book_to_csv(book: Dict[str, List[Dict[str, Any]]], filename: str) -> None:
    buy = book["buy"]
    sell = book["sell"]

    fieldnames = [
        "side",          # "buy" or "sell"
        "price",
        "volume_remain",
        "volume_total",
        "system_id",
        "location_id",
        "duration",
        "issued",
        "order_id",
        "min_volume",
        "range",
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for o in buy:
            writer.writerow(
                {
                    "side": "buy",
                    "price": o["price"],
                    "volume_remain": o["volume_remain"],
                    "volume_total": o["volume_total"],
                    "system_id": o["system_id"],
                    "location_id": o["location_id"],
                    "duration": o["duration"],
                    "issued": o["issued"],
                    "order_id": o["order_id"],
                    "min_volume": o["min_volume"],
                    "range": o["range"],
                }
            )

        for o in sell:
            writer.writerow(
                {
                    "side": "sell",
                    "price": o["price"],
                    "volume_remain": o["volume_remain"],
                    "volume_total": o["volume_total"],
                    "system_id": o["system_id"],
                    "location_id": o["location_id"],
                    "duration": o["duration"],
                    "issued": o["issued"],
                    "order_id": o["order_id"],
                    "min_volume": o["min_volume"],
                    "range": o["range"],
                }
            )

    print(f"  - order book -> {filename}")


def export_history_to_csv(history: List[Dict[str, Any]], filename: str) -> None:
    if not history:
        print("  - no history data to export")
        return

    fieldnames = ["date", "average", "highest", "lowest", "order_count", "volume"]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "date": row["date"],
                    "average": row["average"],
                    "highest": row["highest"],
                    "lowest": row["lowest"],
                    "order_count": row["order_count"],
                    "volume": row["volume"],
                }
            )

    print(f"  - history -> {filename}")


def plot_history_from_csv(csv_filename: str, png_filename: str) -> None:
    dates: List[datetime] = []
    averages: List[float] = []
    highs: List[float] = []
    lows: List[float] = []
    volumes: List[float] = []

    with open(csv_filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.append(datetime.fromisoformat(row["date"]))
            averages.append(float(row["average"]))
            highs.append(float(row["highest"]))
            lows.append(float(row["lowest"]))
            volumes.append(float(row["volume"]))

    if not dates:
        print("  - no rows in CSV; nothing to plot")
        return

    fig, ax_price = plt.subplots(figsize=(10, 5))

    # Price lines (linear scale)
    ax_price.plot(dates, averages, label="Average", linewidth=1)
    ax_price.plot(dates, highs, label="High", linestyle="--", linewidth=1)
    ax_price.plot(dates, lows, label="Low", linestyle="--", linewidth=1)
    ax_price.set_xlabel("Date")
    ax_price.set_ylabel("Price (ISK)")
    ax_price.set_title("Missile – Sinq Laison – last ~6 months")
    ax_price.grid(True, which="both", linestyle="--", alpha=0.3)

    # Volume on secondary axis, log scale (so big spikes are visible)
    ax_vol = ax_price.twinx()
    ax_vol.bar(dates, volumes, alpha=0.2, label="Volume")
    ax_vol.set_ylabel("Volume (units)")
    ax_vol.set_yscale("log")
    ax_vol.set_ylim(bottom=1)

    # Combined legend
    lines_1, labels_1 = ax_price.get_legend_handles_labels()
    lines_2, labels_2 = ax_vol.get_legend_handles_labels()
    ax_price.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    fig.tight_layout()
    fig.savefig(png_filename, dpi=150)
    plt.close(fig)

    print(f"  - figure -> {png_filename}")


# ---------------------------------------------------------------------------
# 4) main(): loop over ALL missiles in the region
# ---------------------------------------------------------------------------

def main():
    api = EveMarketAPI(
        user_agent="all-missiles-the-forge-tool/1.0 (you@example.com)"#"all-missiles-sinq-tool/1.0 (you@example.com)"
    )

    region_id = THE_FORGE_REGION_ID#SINQ_LAISON_REGION_ID  # “regime”

    print("Discovering type IDs traded in region...")
    region_type_ids = get_region_type_ids(api, region_id)
    print(f"Total types with orders in region: {len(region_type_ids)}")

    # Filter down to missiles
    missile_ids: List[int] = []
    for tid in region_type_ids:
        if is_missile_type(api, tid):
            missile_ids.append(tid)

    print(f"Missile types with orders in region: {len(missile_ids)}")
    if not missile_ids:
        return

    os.makedirs("output", exist_ok=True)
    cutoff = datetime.utcnow().date() - timedelta(days=180)  # ~6 months

    for type_id in missile_ids:
        type_info = get_type_info(api, type_id)
        type_name = type_info["name"]
        slug = slugify(type_name)

        print(f"\n=== {type_name} (type_id={type_id}) ===")

        # --- order book (region-wide) ---
        book = api.get_region_order_book(region_id=region_id, type_id=type_id)
        ob_csv = os.path.join("output", f"orderbook_{type_id}_{slug}.csv")
        export_order_book_to_csv(book, ob_csv)

        # --- history (region-wide, last ~6 months) ---
        full_hist = api.get_region_history(region_id=region_id, type_id=type_id)
        hist_6m = [
            row
            for row in full_hist
            if datetime.fromisoformat(row["date"]).date() >= cutoff
        ]

        if not hist_6m:
            print("  - no history in last 6 months; skipping CSV/plot for this missile")
            continue  # go to next missile

        hist_csv = os.path.join("output", f"history_{type_id}_{slug}_6m.csv")
        export_history_to_csv(hist_6m, hist_csv)

        # --- plot using the CSV (so "everything in the CSV" goes on the figure) ---
        png_path = os.path.join("output", f"history_{type_id}_{slug}_6m.png")
        plot_history_from_csv(hist_csv, png_path)


if __name__ == "__main__":
    main()
