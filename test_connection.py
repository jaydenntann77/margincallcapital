from config import API_KEY, API_SECRET, BASE_URL

print("API key loaded:", API_KEY is not None)
print("API secret loaded:", API_SECRET is not None)
print("Base URL:", BASE_URL)

from client import get_balance, get_exchange_info, get_ticker

#get_exchange_info()
get_balance()
get_ticker("BTC/USD")