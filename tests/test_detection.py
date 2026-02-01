import sys
import os
sys.path.append(os.getcwd())
from scripts.fetch_binance import detect_listing_date

symbol = "WLDUSDT"
print(f"Testing detection for {symbol}...")
date = detect_listing_date(symbol)
print(f"Detected: {date}")
