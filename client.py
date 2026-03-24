import math
import random
import time
import hmac
import hashlib
import requests
import logging
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from config import BASE_URL, API_KEY, API_SECRET

# Set up structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("RoostooClient")

class RoostooClient:
    def __init__(self):
        self.base_url = BASE_URL.rstrip('/')
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.session = requests.Session()
        self._exchange_cache: dict = {}   # populated once on first call
        logger.info("RoostooClient initialized with base URL: %s", self.base_url)
        logger.info("Trading client initialized and ready.")

    def _get_timestamp(self) -> str:
        """Returns 13-digit millisecond timestamp."""
        return str(int(time.time() * 1000))

    def _generate_signature(self, payload: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Any], str]:
        """
        Calculates HMAC SHA256 signature and returns headers + updated payload.
        Ensures parameters are sorted for deterministic signing.
        """
        payload = dict(payload)
        # Inject timestamp into payload if not present
        if "timestamp" not in payload:
            payload["timestamp"] = self._get_timestamp()

        # Sort keys to ensure the signature matches the server-side generation
        sorted_keys = sorted(payload.keys())
        total_params = "&".join(f"{k}={str(payload[k])}" for k in sorted_keys)

        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            total_params.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        headers = {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": signature,
        }

        return headers, payload, total_params

    def _request(self, method: str, endpoint: str, signed: bool = False, params: Optional[Dict] = None) -> Dict:
        """Centralized request handler with error catching and logging."""
        url = f"{self.base_url}{endpoint}"
        logger.debug("Full URL being called: %s", url)
        payload = params or {}
        headers = {}
        total_params: Optional[str] = None

        if signed:
            headers, payload, total_params = self._generate_signature(payload)
            # Roostoo requires x-www-form-urlencoded for POST signed endpoints.
            if method.upper() == "POST":
                headers["Content-Type"] = "application/x-www-form-urlencoded"

        try:
            method_upper = method.upper()
            request_kwargs: Dict[str, Any] = {"headers": headers}

            if signed:
                # Signed GET endpoints: send as query string.
                if method_upper == "GET":
                    request_kwargs["params"] = payload
                # Signed POST endpoints: send the exact signed string as body.
                elif method_upper == "POST":
                    request_kwargs["data"] = total_params or ""
                else:
                    raise ValueError(f"Unsupported signed HTTP method: {method}")
            else:
                request_kwargs["params"] = payload

            response = self.session.request(method_upper, url, **request_kwargs)

            # Log the outgoing intent
            logger.debug(f"Request: {method} {endpoint} | Payload: {payload}")

            response.raise_for_status()
            data = response.json()

            return data

        except requests.exceptions.RequestException as e:
            status = getattr(e.response, 'status_code', 'N/A')
            response_text = None
            try:
                response_text = e.response.text if e.response is not None else None
            except Exception:
                response_text = None
            logger.error(f"API Error: {endpoint} | Status: {status} | Msg: {e}")
            if response_text:
                logger.error("API Error Response Body: %s", response_text)
            return {"error": str(e), "status_code": status, "response_text": response_text, "Success": False}

    def _call_with_retry(self, method, *args, **kwargs) -> Dict:
        """
        Call any RoostooClient method and retry on transient network failures.
        Network errors are detected from the error message in the returned dict
        (the underlying _request never raises — it converts exceptions to dicts).
        """
        from config import MAX_RETRIES, RETRY_BACKOFF, RETRY_JITTER
        last: dict = {}
        for attempt in range(MAX_RETRIES + 1):
            result = method(*args, **kwargs)
            last = result
            if isinstance(result, dict) and result.get("Success") is False:
                err = str(result.get("error", "")).lower()
                is_network = any(
                    kw in err for kw in
                    ("connection", "timeout", "network", "max retries", "refused", "reset by peer")
                )
                if is_network and attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF * (2 ** attempt) * (1 + random.uniform(0, RETRY_JITTER))
                    logger.warning(
                        "Network error (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, MAX_RETRIES + 1, err, wait,
                    )
                    time.sleep(wait)
                    continue
            return result
        logger.error("All retry attempts exhausted.")
        return last

    ## --- Precision helpers (used by bot order methods) ---

    def _fmt_quantity(self, pair: str, quantity: float) -> str:
        """
        Format a quantity string to the exchange's AmountPrecision for this pair.
        AmountPrecision=0 means integer quantities (e.g. DOGE, SHIB, BONK).
        Truncates (floors) rather than rounds to avoid over-ordering.
        """
        meta = self.get_pair_meta(pair)
        amt_prec = meta.get("AmountPrecision", 8)
        if amt_prec == 0:
            return str(int(math.floor(quantity)))
        factor = 10 ** amt_prec
        truncated = math.floor(quantity * factor) / factor
        return f"{truncated:.{amt_prec}f}"

    def _fmt_price(self, pair: str, price: float) -> str:
        """Format a price string to the exchange's PricePrecision for this pair."""
        meta = self.get_pair_meta(pair)
        price_prec = meta.get("PricePrecision", 8)
        return f"{price:.{price_prec}f}"

    ## --- Public Endpoints ---

    def get_exchange_info(self) -> Dict:
        """Fetches (and caches) market metadata."""
        if not self._exchange_cache:
            logger.info("Fetching exchange info...")
            result = self._request("GET", "/v3/exchangeInfo")
            if result.get("IsRunning") is not None:
                self._exchange_cache = result
            return result
        return self._exchange_cache

    def get_ticker(self, pair: str) -> Dict:
        """Fetches real-time price for a pair (e.g., 'BTC/USD')."""
        params = {"pair": pair, "timestamp": self._get_timestamp()}
        return self._request("GET", "/v3/ticker", params=params)

    ## --- Private Endpoints ---

    def get_balance(self) -> Dict:
        """Fetches account holdings (Requires Signature)."""
        logger.info("Requesting account balance...")
        return self._request("GET", "/v3/balance", signed=True)

    def pending_count(self) -> Dict:
        """Get pending order count (SIGNED): GET /v3/pending_count."""
        logger.info("Requesting pending order count...")
        return self._request("GET", "/v3/pending_count", signed=True)

    def query_order(
        self,
        order_id: Optional[str] = None,
        pair: Optional[str] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        pending_only: Optional[bool] = None,
    ) -> Dict:
        """Query order(s) (SIGNED): POST /v3/query_order.

        Doc rules:
        - timestamp is required (injected automatically by signer)
        - If `order_id` is sent, none of other optional parameters are allowed.
        - `pending_only` accepts TRUE/FALSE.
        """

        if order_id is not None:
            if any(v is not None for v in (pair, offset, limit, pending_only)):
                raise ValueError("When order_id is provided, pair/offset/limit/pending_only must not be set")
            params: Dict[str, Any] = {"order_id": str(order_id)}
            return self._request("POST", "/v3/query_order", signed=True, params=params)

        params = {}
        if pair is not None:
            normalized_pair = str(pair).strip().upper().replace("-", "/")
            params["pair"] = normalized_pair

        if offset is not None:
            params["offset"] = str(int(offset))

        if limit is not None:
            params["limit"] = str(int(limit))

        if pending_only is not None:
            params["pending_only"] = "TRUE" if pending_only else "FALSE"

        logger.info(
            "Querying orders: order_id=%s pair=%s pending_only=%s offset=%s limit=%s",
            order_id,
            params.get("pair"),
            params.get("pending_only"),
            params.get("offset"),
            params.get("limit"),
        )
        return self._request("POST", "/v3/query_order", signed=True, params=params)

    def place_order(
        self,
        pair: str,
        side: str,
        type: str,
        quantity: str,
        price: Optional[float] = None,
    ) -> Dict:
        """Place an order (SIGNED): POST /v3/place_order.
        """

        normalized_pair = str(pair).strip().upper()
        if "/" not in normalized_pair:
            raise ValueError("pair must look like 'BTC/USD'")

        normalized_side = str(side).strip().upper()
        if normalized_side not in {"BUY", "SELL"}:
            raise ValueError("side must be 'BUY' or 'SELL'")

        normalized_type = str(type).strip().upper()
        if normalized_type not in {"LIMIT", "MARKET"}:
            raise ValueError("type must be 'LIMIT' or 'MARKET'")

        if quantity is None or str(quantity).strip() == "":
            raise ValueError("quantity is required")

        if normalized_type == "LIMIT" and price is None:
            raise ValueError("LIMIT orders require price")

        params: Dict[str, Any] = {
            "pair": normalized_pair,
            "side": normalized_side,
            "type": normalized_type,
            "quantity": str(quantity),
        }
        if normalized_type == "LIMIT":
            params["price"] = str(price)

        logger.info(
            "ORDER_INIT: %s %s %s (%s)",
            params["side"],
            params["quantity"],
            params["pair"],
            params["type"],
        )
        result = self._request("POST", "/v3/place_order", signed=True, params=params)

        if result.get("Success") == True:
            order_detail = result.get("OrderDetail") or {}
            logger.info(
                "ORDER_ACCEPTED: ID %s | %s",
                order_detail.get("OrderID"),
                order_detail.get("Status"),
            )
        else:
            logger.warning("ORDER_REJECTED: %s", result.get("ErrMsg") or result.get("error") or "Unknown Error")

        return result

    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel an open order by ID (SIGNED): POST /v3/cancel_order.
        NOTE: Adjust the endpoint path if the live API differs.
        """
        logger.info("Cancelling order %s", order_id)
        return self._call_with_retry(
            self._request,
            "POST", "/v3/cancel_order",
            signed=True,
            params={"order_id": str(order_id)},
        )

    ## --- Bot convenience methods ---

    def get_pair_meta(self, pair: str) -> Dict:
        """Return the metadata dict for a single pair (PricePrecision, AmountPrecision, etc.)."""
        return self.get_exchange_info().get("TradePairs", {}).get(pair, {})

    def validate_pair(self, pair: str) -> bool:
        """Return True if the pair exists on the exchange and CanTrade=True."""
        meta = self.get_pair_meta(pair)
        return bool(meta) and bool(meta.get("CanTrade", False))

    def get_price(self, pair: str) -> Optional[float]:
        """Return the last trade price for a pair, or None on failure."""
        try:
            ticker = self._call_with_retry(self.get_ticker, pair)
            if not ticker.get("Success"):
                logger.debug("Ticker Success=False for %s: %s", pair, ticker.get("ErrMsg"))
                return None
            pair_data = ticker.get("Data", {}).get(pair, {})
            raw = pair_data.get("LastPrice")
            return float(raw) if raw is not None else None
        except Exception as exc:
            logger.error("get_price(%s) raised: %s", pair, exc)
            return None

    def get_usd_balance(self) -> float:
        """Return the free USD balance."""
        try:
            df = self.get_balance_df()
            if df.empty:
                return 0.0
            row = df[df["asset"] == "USD"]
            return float(row["free"].iloc[0]) if not row.empty else 0.0
        except Exception as exc:
            logger.error("get_usd_balance failed: %s", exc)
            return 0.0

    def get_asset_balance(self, asset: str) -> float:
        """Return the free balance for a given asset symbol (e.g. 'BTC')."""
        try:
            df = self.get_balance_df()
            if df.empty:
                return 0.0
            row = df[df["asset"] == asset]
            return float(row["free"].iloc[0]) if not row.empty else 0.0
        except Exception as exc:
            logger.error("get_asset_balance(%s) failed: %s", asset, exc)
            return 0.0

    def get_portfolio_value(self) -> Dict:
        """
        Return a snapshot of portfolio value.
        Keys: usd_cash, asset_values ({asset: {amount, price, value_usd}}), total_usd.
        """
        try:
            df = self.get_balance_df()
            if df.empty:
                return {"usd_cash": 0.0, "asset_values": {}, "total_usd": 0.0}

            usd_row  = df[df["asset"] == "USD"]
            usd_cash = float(usd_row["free"].iloc[0]) if not usd_row.empty else 0.0

            asset_values: Dict = {}
            total_usd = usd_cash

            for _, row in df[df["asset"] != "USD"].iterrows():
                asset        = str(row["asset"])
                total_amount = float(row["total"])
                if total_amount <= 0:
                    continue
                price = self.get_price(f"{asset}/USD")
                if price:
                    value = total_amount * price
                    asset_values[asset] = {"amount": total_amount, "price": price, "value_usd": value}
                    total_usd += value

            return {"usd_cash": usd_cash, "asset_values": asset_values, "total_usd": total_usd}
        except Exception as exc:
            logger.error("get_portfolio_value failed: %s", exc)
            return {"usd_cash": 0.0, "asset_values": {}, "total_usd": 0.0}

    def get_open_orders(self, pair: Optional[str] = None) -> list:
        """Return a list of currently pending/open orders."""
        try:
            result = self._call_with_retry(self.query_order, pair=pair, pending_only=True)
            if result.get("Success") is False:
                err = (result.get("ErrMsg") or "").lower()
                if "no order" in err:
                    return []
                logger.debug("get_open_orders: %s", result.get("ErrMsg"))
                return []
            return result.get("OrderMatched") or []
        except Exception as exc:
            logger.error("get_open_orders failed: %s", exc)
            return []

    def get_order_status(self, order_id: str) -> Dict:
        """Return the status dict for a specific order, or {} if not found."""
        try:
            result = self._call_with_retry(self.query_order, order_id=order_id)
            orders = result.get("OrderMatched") or []
            return orders[0] if orders else {}
        except Exception as exc:
            logger.error("get_order_status(%s) failed: %s", order_id, exc)
            return {}

    def place_limit_buy(self, pair: str, quantity: float, price: float) -> Dict:
        """Place a limit BUY with exchange-correct precision."""
        qty_str   = self._fmt_quantity(pair, quantity)
        price_str = self._fmt_price(pair, price)
        logger.info("Placing LIMIT BUY  %s  qty=%s  price=%s", pair, qty_str, price_str)
        return self._call_with_retry(
            self.place_order,
            pair=pair, side="BUY", type="LIMIT",
            quantity=qty_str, price=float(price_str),
        )

    def place_limit_sell(self, pair: str, quantity: float, price: float) -> Dict:
        """Place a limit SELL with exchange-correct precision."""
        qty_str   = self._fmt_quantity(pair, quantity)
        price_str = self._fmt_price(pair, price)
        logger.info("Placing LIMIT SELL %s  qty=%s  price=%s", pair, qty_str, price_str)
        return self._call_with_retry(
            self.place_order,
            pair=pair, side="SELL", type="LIMIT",
            quantity=qty_str, price=float(price_str),
        )

    def place_market_buy(self, pair: str, quantity: float) -> Dict:
        """Place a market BUY with exchange-correct precision."""
        qty_str = self._fmt_quantity(pair, quantity)
        logger.info("Placing MARKET BUY  %s  qty=%s", pair, qty_str)
        return self._call_with_retry(
            self.place_order, pair=pair, side="BUY", type="MARKET", quantity=qty_str,
        )

    def place_market_sell(self, pair: str, quantity: float) -> Dict:
        """Place a market SELL with exchange-correct precision."""
        qty_str = self._fmt_quantity(pair, quantity)
        logger.info("Placing MARKET SELL %s  qty=%s", pair, qty_str)
        return self._call_with_retry(
            self.place_order, pair=pair, side="SELL", type="MARKET", quantity=qty_str,
        )

    ## --- Analytics helpers (unchanged) ---

    def get_balance_df(self) -> pd.DataFrame:
        """
        Fetches balance and returns a cleaned DataFrame.
        Roostoo balance responses may include one of:
        - 'Wallet' (single wallet object keyed by asset symbol)
        - 'SpotWallet' and 'MarginWallet' (separate wallet objects)
        - 'Balances' (list format, legacy/alt)
        """
        raw_data = self.get_balance()

        if raw_data.get('Success') == False:
            logger.warning("Balance fetch failed: %s", raw_data.get('ErrMsg') or raw_data.get('error') or raw_data)
            return pd.DataFrame()

        def _wallet_rows(wallet_obj: Any) -> list[dict]:
            if not isinstance(wallet_obj, dict) or not wallet_obj:
                return []
            parsed_rows = []
            for asset, amounts in wallet_obj.items():
                amounts = amounts or {}
                try:
                    free_val = float(amounts.get("Free", 0) or 0)
                except (TypeError, ValueError):
                    free_val = 0.0
                try:
                    lock_val = float(amounts.get("Lock", 0) or 0)
                except (TypeError, ValueError):
                    lock_val = 0.0
                parsed_rows.append({
                    "asset": str(asset),
                    "free": free_val,
                    "locked": lock_val,
                    "total": free_val + lock_val,
                })
            return parsed_rows

        rows: list[dict] = []
        wallet = raw_data.get('Wallet')
        spot_wallet = raw_data.get('SpotWallet')
        margin_wallet = raw_data.get('MarginWallet')

        if isinstance(wallet, dict) and wallet:
            rows = _wallet_rows(wallet)
        elif isinstance(spot_wallet, dict) or isinstance(margin_wallet, dict):
            # Combine spot + margin by asset symbol.
            combined: Dict[str, Dict[str, float]] = {}
            for wallet_obj in (spot_wallet, margin_wallet):
                for row in _wallet_rows(wallet_obj):
                    asset = row["asset"]
                    if asset not in combined:
                        combined[asset] = {"free": 0.0, "locked": 0.0, "total": 0.0}
                    combined[asset]["free"] += float(row["free"])
                    combined[asset]["locked"] += float(row["locked"])
                    combined[asset]["total"] += float(row["total"])
            rows = [
                {"asset": asset, **vals}
                for asset, vals in combined.items()
            ]
        else:
            balances = raw_data.get('Balances')
            if isinstance(balances, list) and balances:
                for item in balances:
                    if not isinstance(item, dict):
                        continue
                    asset = item.get("Asset") or item.get("asset")
                    if not asset:
                        continue
                    free_val = item.get("Free", item.get("free", 0))
                    locked_val = item.get("Locked", item.get("locked", item.get("Lock", item.get("lock", 0))))
                    total_val = item.get("Total", item.get("total", None))
                    try:
                        free_num = float(free_val or 0)
                    except (TypeError, ValueError):
                        free_num = 0.0
                    try:
                        locked_num = float(locked_val or 0)
                    except (TypeError, ValueError):
                        locked_num = 0.0
                    if total_val is None:
                        total_num = free_num + locked_num
                    else:
                        try:
                            total_num = float(total_val or 0)
                        except (TypeError, ValueError):
                            total_num = free_num + locked_num

                    rows.append({
                        "asset": str(asset),
                        "free": free_num,
                        "locked": locked_num,
                        "total": total_num,
                    })
            else:
                logger.warning("Balance response missing Wallet/SpotWallet/MarginWallet/Balances fields: %s", raw_data)
                return pd.DataFrame()

        df = pd.DataFrame(rows)
        for col in ["free", "locked", "total"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        return df

    def get_order_history_df(self, pair: Optional[str] = None, pending_only: Optional[bool] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetches order history and returns a cleaned DataFrame.
        """
        result = self.query_order(pair=pair, pending_only=pending_only, limit=limit)

        if result.get('Success') == False:
            err_msg = (result.get('ErrMsg') or "").strip()
            if err_msg.lower() == "no order matched":
                return pd.DataFrame()
            logger.warning("Order history fetch failed: %s", err_msg or result.get('error') or result)
            return pd.DataFrame()

        orders = result.get("OrderMatched") or []
        if not isinstance(orders, list):
            logger.warning("Unexpected format for OrderMatched: %s", orders)
            return pd.DataFrame()

        df = pd.DataFrame(orders)
        # Additional cleaning/normalization can be done here based on actual order fields.
        return df

    def get_binance_history(self, pair: str, interval: str = "4h", limit: int = 300) -> pd.Series:
        """
        Fetch historical closing prices from Binance's free public REST API (no auth required).
        Used at startup to seed price_history so momentum signals are immediately meaningful.

        Pair mapping: 'BTC/USD' → 'BTCUSDT', 'ETH/USD' → 'ETHUSDT', etc.
        Returns a pd.Series of closing prices indexed by UTC datetime, or an empty Series on failure.
        """
        import requests as _req
        symbol = pair.replace("/USD", "USDT").replace("/", "")
        url    = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        try:
            resp = _req.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            # Each element: [open_time, open, high, low, close, volume, ...]
            closes = pd.Series(
                [float(bar[4]) for bar in data],
                index=pd.to_datetime([bar[0] for bar in data], unit="ms"),
                name=pair,
            )
            logger.info(
                "[BINANCE SEED] %s  bars=%d  latest_close=%.4f",
                pair, len(closes), closes.iloc[-1] if len(closes) else 0,
            )
            return closes
        except Exception as exc:
            logger.warning("[BINANCE SEED] Failed for %s: %s — will fill from live prices.", pair, exc)
            return pd.Series(name=pair, dtype=float)
