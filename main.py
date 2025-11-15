# main.py

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
