import time
import hmac
import hashlib
import requests

from config import BASE_URL, API_KEY, API_SECRET

# to get public info
def get_exchange_info():
    url = f"{BASE_URL}/v3/exchangeInfo"

    response = requests.get(url)

    print("Status code:", response.status_code)

    try:
        print("Response JSON:", response.json())
    except Exception:
        print("Raw response:", response.text)



# to get signed info
def get_timestamp():
    return str(int(time.time() * 1000))

def get_signed_headers(payload=None):
    if payload is None:
        payload = {}

    payload = dict(payload)
    payload["timestamp"] = get_timestamp()

    sorted_keys = sorted(payload.keys())
    total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)

    signature = hmac.new(
        API_SECRET.encode("utf-8"),
        total_params.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    headers = {
        "RST-API-KEY": API_KEY,
        "MSG-SIGNATURE": signature
    }

    return headers, payload


def get_balance():
    url = f"{BASE_URL}/v3/balance"
    headers, payload = get_signed_headers()

    response = requests.get(url, headers=headers, params=payload)

    print("Status Code:", response.status_code)
    print("Raw response:", response.text)

def get_ticker(pair):
    url = f"{BASE_URL}/v3/ticker"
    params = {
        "pair": pair,
        "timestamp": str(int(time.time() * 1000))
    }

    response = requests.get(url, params=params)
    data = response.json()

    print("Status Code:", response.status_code)
    print("Response JSON:", data)

