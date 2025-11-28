# market_api.py

import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any

ESI_BASE_URL = "https://esi.evetech.net/latest"
DATASOURCE = "tranquility"

# Inferno Heavy Missile
INFERNO_HEAVY_MISSILE_TYPE_ID = 208      # type_id for Inferno Heavy Missile

# Region
SINQ_LAISON_REGION_ID = 10000032         # region_id for Sinq Laison
THE_FORGE_REGION_ID = 10000002           # region_id for Forge

# System
JITA_SYSTEM_ID = 30000142             # solar system Jita

# Station
JITA_44_STATION_ID = 60003760         # Jita IV - Moon 4 - Caldari Navy Assembly Plant

class EveMarketAPI:
    def __init__(
        self,
        user_agent: str = "inferno-hm-sinq-market-tool/1.0 (you@example.com)",
        base_url: str = ESI_BASE_URL,
        datasource: str = DATASOURCE,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.datasource = datasource
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "application/json",
            }
        )

    def _get(self, path: str, **params) -> requests.Response:
        params.setdefault("datasource", self.datasource)
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp

    # -------- region order book (buy + sell) --------

    def get_region_order_book(
        self,
        region_id: int,
        type_id: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Full order book for a type in a region.

        Returns:
            {
              "buy":  [highest price first],
              "sell": [lowest price first],
            }
        """
        orders: List[Dict[str, Any]] = []
        page = 1

        while True:
            resp = self._get(
                f"/markets/{region_id}/orders/",
                type_id=type_id,
                order_type="all",
                page=page,
            )
            data = resp.json()
            if not data:
                break

            orders.extend(data)

            x_pages = resp.headers.get("X-Pages")
            if not x_pages or page >= int(x_pages):
                break
            page += 1

        buy_orders = [o for o in orders if o.get("is_buy_order")]
        sell_orders = [o for o in orders if not o.get("is_buy_order")]

        buy_orders.sort(key=lambda o: o["price"], reverse=True)
        sell_orders.sort(key=lambda o: o["price"])

        return {"buy": buy_orders, "sell": sell_orders}

    # -------- region price history --------
    '''
    def get_region_history(
        self,
        region_id: int,
        type_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Daily price history for an item in a region.
        """
        resp = self._get(
            f"/markets/{region_id}/history/",
            type_id=type_id,
        )
        return resp.json()
    '''
    def get_region_history(
        self,
        region_id: int,
        type_id: int,
    ):
        """
        Get daily price/volume history for a type in a region.

        Returns [] if ESI says 404 (no history for this type/region).
        """
        try:
            # Adjust to your _get signature; this version matches the one you showed
            resp = self._get(
                f"/markets/{region_id}/history/",
                type_id=type_id,
            )
        except requests.exceptions.HTTPError as exc:
            # ESI docs: 404 = type not found / no history for that type in region :contentReference[oaicite:1]{index=1}
            if exc.response is not None and exc.response.status_code == 404:
                return []
            # anything else (500, 420, etc.) still bubbles up
            raise

        return resp.json()

    def get_region_history_last_days(
        self,
        region_id: int,
        type_id: int,
        days: int,
    ) -> List[Dict[str, Any]]:
        """
        Filter history down to the last `days` days.
        """
        all_rows = self.get_region_history(region_id=region_id, type_id=type_id)
        cutoff = datetime.utcnow().date() - timedelta(days=days)
        filtered: List[Dict[str, Any]] = []

        for row in all_rows:
            row_date = datetime.fromisoformat(row["date"]).date()
            if row_date >= cutoff:
                filtered.append(row)

        return filtered


class InfernoHeavyMissileSinqAPI(EveMarketAPI):
    """
    Convenience wrapper for:
      - Inferno Heavy Missile (type 208)
      - Sinq Laison (region 10000032)
    """

    def get_sinq_order_book(self) -> Dict[str, List[Dict[str, Any]]]:
        return self.get_region_order_book(
            region_id=SINQ_LAISON_REGION_ID,
            type_id=INFERNO_HEAVY_MISSILE_TYPE_ID,
        )

    def get_sinq_history_6m(self) -> List[Dict[str, Any]]:
        # ~6 months ≈ 180 days
        return self.get_region_history_last_days(
            region_id=SINQ_LAISON_REGION_ID,
            type_id=INFERNO_HEAVY_MISSILE_TYPE_ID,
            days=180,
        )
