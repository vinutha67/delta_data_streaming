import requests
import pandas as pd
from datetime import datetime, timedelta

# Endpoint base URL
BASE_URL = "https://api.india.delta.exchange"  # Or https://api.delta.exchange

# Define parameters (e.g., last 24 hours of 1-hour candles)
end_time = int(datetime.now().timestamp())
start_time = int((datetime.now() - timedelta(days=1)).timestamp())

params = {
    'symbol': 'BTCUSD',
    'resolution': '1h',
    'start': str(start_time),
    'end': str(end_time)
}

# Send GET Request
response = requests.get(f"{BASE_URL}/v2/history/candles", params=params)

if response.status_code == 200:
    data = response.json()
    candles = data.get('result', [])

    # Convert to Pandas DataFrame
    df = pd.DataFrame(candles)
    if not df.empty:
        # Convert Unix epoch timestamp to readable format
        df['time'] = pd.to_datetime(df['time'], unit='s')
        print(df[['time', 'open', 'high', 'low', 'close', 'volume']])
    else:
        print("No candle data returned.")
else:
    print(f"Error {response.status_code}: {response.text}")