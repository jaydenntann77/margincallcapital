import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from client import RoostooClient


def run_live_tests():
    # 1. Initialize the client (Reads from your config.py)
    client = RoostooClient()
    print("=== LIVE API TEST START ===")

    # --- TEST 1: Public GET (Connectivity Check) ---
    print("\n[1/6] Testing Public GET: Exchange Info...")
    info = client.get_exchange_info()

    # Roostoo uses 'IsRunning' and 'TradePairs'
    if info.get("IsRunning") == True:
        pairs = info.get("TradePairs", {})
        # Get the first 5 symbols from the dictionary keys
        sample_symbols = list(pairs.keys())[:5]
        print(f"SUCCESS: Connected. Sample assets available: {sample_symbols}")
        print(f"Initial Wallet Setting: {info.get('InitialWallet')}")
    else:
        print(f"FAILED: Exchange is not running or returned: {info}")

    # --- TEST 2: Private SIGNED (Authentication Check) ---
    print("\n[2/6] Testing Private SIGNED: Account Balance...")
    balance_df = client.get_balance_df()

    if not balance_df.empty:
        print("SUCCESS: Authentication verified.")
        print("Current Holdings:")
        print(balance_df.to_string(index=False))
    else:
        # Print raw payload for debugging (typically contains Success/ErrMsg)
        raw_balance = client.get_balance()
        print("FAILED: Received empty balance or Auth error.")
        print(f"Raw balance response: {raw_balance}")

    # --- TEST 3: Market Data GET (Ticker) ---
    pair = "BTC/USD"
    print(f"\n[3/6] Testing Ticker GET: {pair}...")
    ticker = client.get_ticker(pair)
    print(f"Raw Ticker Response: {ticker}")

    # 1. Roostoo uses 'Success' (Capital S) not 'status'
    # 2. 'Data' is also capitalized
    is_success = ticker.get("Success")
    data = ticker.get("Data")

    print(f"Status: {is_success}, Data: {data}")

    if is_success:
        # The price is nested inside the pair name key
        # e.g., ticker['Data']['BTC/USD']['LastPrice']
        pair_data = data.get(pair, {})
        price = pair_data.get("LastPrice")

        if price:
            print(f"SUCCESS: {pair} Last Price: {price}")
        else:
            print(f"FAILED: Found data for {pair} but LastPrice was missing.")
    else:
        print(f"FAILED: API returned Success=False. Error: {ticker.get('ErrMsg')}")

    # --- TEST 4: Private SIGNED (Pending Count) ---
    print("\n[4/6] Testing Private SIGNED: Pending Order Count...")
    pending = client.pending_count()
    if pending.get("Success") is True:
        print(
            f"SUCCESS: TotalPending={pending.get('TotalPending')} | OrderPairs={pending.get('OrderPairs')}"
        )
    else:
        # Roostoo returns Success=false when no pending orders
        print(
            f"INFO: Pending count returned Success=False. ErrMsg={pending.get('ErrMsg')} | TotalPending={pending.get('TotalPending')}"
        )

    # --- TEST 5: Private SIGNED (Order History DF) ---
    print("\n[5/6] Testing Private SIGNED: Order History (DataFrame)...")
    query_pair = None
    if isinstance(pending.get("OrderPairs"), dict) and pending.get("OrderPairs"):
        query_pair = list(pending.get("OrderPairs").keys())[0]

    if query_pair:
        orders_df = client.get_order_history_df(pair=query_pair, limit=5)
        print(f"Fetched pending order history for {query_pair}")
    else:
        orders_df = client.get_order_history_df(limit=5)
        print("Fetched recent order history")

    if not orders_df.empty:
        print(f"SUCCESS: Matched {len(orders_df)} orders")
        print(orders_df.head(3).to_string(index=False))
    else:
        print("INFO: No orders matched (empty history)")

    # --- TEST 6: Private POST (Execution Check) ---
    # WARNING: This places a MOCK trade on the Roostoo platform.
    print("\n[6/6] Testing Order Placement: 0.001 BTC Market BUY...")
    confirmation = client.place_order(
        pair="BTC/USD",
        side="BUY",
        type="MARKET",
        quantity=str(0.001),
    )

    if confirmation.get("Success") is True:
        order_detail = confirmation.get("OrderDetail") or {}
        print(f"SUCCESS: Order accepted. ID: {order_detail.get('OrderID')}")
    else:
        print(
            f"FAILED: Order rejected. Reason: {confirmation.get('ErrMsg') or confirmation.get('error')}"
        )

    print("\n=== LIVE API TEST COMPLETE ===")


if __name__ == "__main__":
    run_live_tests()
