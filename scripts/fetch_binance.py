"""
Binance OHLCVデータ取得スクリプト

Binance APIから複数時間軸（1m, 15m, 1h, 4h, 1d）のデータを取得し、
日別ファイルに分割保存。
動的な上位通貨取得（24h Quote Volume基準）とクリーンアップ機能に対応。
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import time
import json
import shutil
from typing import Dict, List, Optional
import pandas as pd
import requests


# 対応時間軸
TIMEFRAMES = ["1m", "15m", "1h", "4h", "1d"]

# Binanceの各シンボル開始日（約）- 主要通貨のみハードコード、他はデフォルト
SYMBOL_START_DATES = {
    "BTCUSDT": "2017-08-17",
    "ETHUSDT": "2017-08-17",
    "BNBUSDT": "2017-11-06",
    "SOLUSDT": "2020-08-11",
    "XRPUSDT": "2018-05-04",
    "DOGEUSDT": "2019-07-05",
    "ADAUSDT": "2018-04-17",
    "TRXUSDT": "2018-06-13",
    "AVAXUSDT": "2020-09-22",
    "LINKUSDT": "2019-01-16",
}

# デフォルト開始日（不明なシンボル用）
DEFAULT_START_DATE = "2019-01-01"

# 上場日キャッシュファイルのパス
LISTING_DATES_CACHE_FILE = Path("data/listing_dates.json")


def detect_listing_date(symbol: str, verbose: bool = True) -> Optional[str]:
    """
    二分探索でシンボルの最初の取引日を検出
    
    Args:
        symbol: 対象シンボル
        verbose: ログ出力
        
    Returns:
        YYYY-MM-DD形式の開始日、検出失敗時はNone
    """
    # Binance開設（2017-07-14）から今日まで
    min_ts = int(datetime(2017, 7, 14).timestamp() * 1000)
    max_ts = int(datetime.now().timestamp() * 1000)
    
    # まず現在時刻でデータがあるか確認（廃止された通貨などのチェック）
    latest = fetch_klines(symbol, "1M", limit=1) # 月足でチェック
    if not latest:
        if verbose:
            print(f"⚠️ {symbol}: 現在データが存在しません（廃止または無効なシンボル）")
        return None

    # 二分探索
    start_ts = min_ts
    end_ts = max_ts
    first_found_ts = None
    
    if verbose:
        print(f"🔍 {symbol}: 上場日を検索中...", end="", flush=True)

    # 1ヶ月単位くらいで大まかに探す
    while start_ts <= end_ts:
        mid_ts = (start_ts + end_ts) // 2
        
        # mid_ts以降の最初のデータを取得
        klines = fetch_klines(symbol, "1m", start_time=mid_ts, limit=1)
        
        if klines:
            # データが見つかった -> もっと過去にあるかもしれない
            kline_open_time = klines[0][0]
            first_found_ts = kline_open_time
            end_ts = mid_ts - 1
            # print(f".", end="", flush=True)
        else:
            # データが見つからない -> もっと未来にある
            start_ts = mid_ts + 1
            # print(f".", end="", flush=True)
            
    if first_found_ts:
        dt = datetime.fromtimestamp(first_found_ts / 1000)
        date_str = dt.strftime("%Y-%m-%d")
        if verbose:
            print(f" 発見! -> {date_str}")
        return date_str
    
    if verbose:
        print(" 見つかりませんでした")
    return None


def load_listing_dates_cache() -> Dict[str, str]:
    if LISTING_DATES_CACHE_FILE.exists():
        try:
            with open(LISTING_DATES_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_listing_dates_cache(cache: Dict[str, str]):
    LISTING_DATES_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LISTING_DATES_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def get_symbol_start_date(symbol: str, verbose: bool = True) -> str:
    """
    シンボルの開始日を決定する。
    1. ハードコードされた設定
    2. キャッシュ
    3. APIで検出
    4. デフォルト
    の順で決定。
    """
    # 1. ハードコード
    if symbol in SYMBOL_START_DATES:
        return SYMBOL_START_DATES[symbol]
        
    # 2. キャッシュ
    cache = load_listing_dates_cache()
    if symbol in cache:
        return cache[symbol]
        
    # 3. 自動検出
    detected_date = detect_listing_date(symbol, verbose)
    if detected_date:
        cache[symbol] = detected_date
        save_listing_dates_cache(cache)
        return detected_date
        
    # 4. デフォルト
    return DEFAULT_START_DATE


# 除外するステーブルコイン等（完全一致）
EXCLUDE_SYMBOLS = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "USDD", "GUSD", 
    "LUSD", "SUSD", "FRAX", "MIM", "EURI", "PAXG", "WBTC", "FDUSD"
}

def get_binance_exchange_info() -> set:
    """Binanceで取引可能なシンボルのセットを取得"""
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return {s["symbol"] for s in data["symbols"] if s["status"] == "TRADING"}
    except Exception as e:
        print(f"⚠️ Error fetching exchange info: {e}")
        return set()


CoinGecko
    """
    CoinGeckoから時価総額上位を取得し、BinanceのUSDTペアに変換
    """
    print(f"\n🌍 Fetching top {limit} coins by Market Cap from CoinGecko...")
    
    # Binanceの有効シンボルを取得して存在確認に使用
    binance_symbols = get_binance_exchange_info()
    if not binance_symbols:
        print("⚠️ Failed to verify Binance symbols. Falling back to simple conversion.")
    
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 200,  # 除外分を見越して多めに取得
        "page": 1,
        "sparkline": "false"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        valid_symbols = []
        rank = 1
        
        for coin in data:
            base_symbol = coin["symbol"].upper()
            symbol_usdt = f"{base_symbol}USDT"
            name = coin["name"]
            cap = coin.get("market_cap", 0)
            
            # --- フィルタリング ---
            
            # 1. 除外リスト（完全一致）
            if base_symbol in EXCLUDE_SYMBOLS:
                continue
                
            # 2. "USD" を含むものを除外（ステーブルコイン対策）
            if "USD" in base_symbol:
                continue
                
            # 3. リステーキング/ラップド系などの簡易除外
            if base_symbol.startswith("W") and base_symbol != "WLD": # WBTC, WETHなど。WLDは除外しない
                 # WBTCは上のリストでも弾いているが念のため。
                 # まぁWだけだと誤爆するかもしれないので、主要なものだけはEXCLUDE_SYMBOLSで。
                 pass
            
            # 4. Binance存在チェック
            if binance_symbols and symbol_usdt not in binance_symbols:
                continue

            # 採用
            valid_symbols.append(symbol_usdt)
            print(f"  {rank}. {base_symbol:<5} ({name[:15]:<15}) - Cap: ${cap:,.0f} -> {symbol_usdt}")
            rank += 1
            
            if len(valid_symbols) >= limit:
                break
                
        return valid_symbols
        
    except Exception as e:
        print(f"❌ Error fetching from CoinGecko: {e}")
        print("Falling back to Volume based selection.")
        return get_top_symbols(limit)


def get_top_symbols(limit: int = 30, sort_by: str = "volume") -> List[str]:
    """
    上位シンボルを取得
    
    Args:
        limit: 取得数
        sort_by: 'volume' or 'marketcap'
    """
    if sort_by == "marketcap":
        return fetch_top_by_market_cap(limit)

    # 以下、既存のVolumeベース処理
    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        tickers = response.json()
        
        # USDTペアのみ抽出 & レバレッジトークン(UP/DOWN)等を除外
        filtered = []
        for t in tickers:
            symbol = t["symbol"]
            if not symbol.endswith("USDT"):
                continue
            
            # 除外キーワード
            if any(k in symbol for k in ["UPUSDT", "DOWNUSDT", "BEARUSDT", "BULLUSDT", "BUSD", "DAI", "TUSD", "USDC"]):
                continue
                
            filtered.append(t)
            
        # 出来高（quoteVolume = USDT Volume）順にソート
        filtered.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
        
        # 上位N件のシンボル名を抽出
        top_symbols = [t["symbol"] for t in filtered[:limit]]
        
        print(f"\n📊 Top {limit} Symbols by 24h Volume (USDT):")
        for i, s in enumerate(top_symbols, 1):
            # ちょっとした情報を表示
            t = next(filter(lambda x: x["symbol"] == s, filtered))
            vol = float(t["quoteVolume"])
            print(f"  {i}. {s:<10} (Vol: ${vol:,.0f})")
            
        return top_symbols
        
    except Exception as e:
        print(f"Error fetching top symbols: {e}")
        # フォールバック: 主要通貨を返す
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    start_time: int = None,
    end_time: int = None,
    limit: int = 1000,
) -> list:
    """
    Binance APIからKline（ローソク足）データを取得
    """
    url = "https://api.binance.com/api/v3/klines"
    
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    
    return response.json()


def klines_to_dataframe(klines: list) -> pd.DataFrame:
    """KlineデータをDataFrameに変換"""
    if not klines:
        return pd.DataFrame()
    
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", 
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    df["timestamp"] = df["open_time"].astype(int)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    
    return df


def fetch_one_day_data(
    symbol: str,
    interval: str,
    date: datetime,
    verbose: bool = False,
) -> pd.DataFrame:
    """1日分のデータを取得"""
    start_time = int(date.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    end_time = int(date.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp() * 1000)
    
    all_data = []
    current_start = start_time
    
    while current_start < end_time:
        try:
            klines = fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=1000,
            )
            
            if not klines:
                break
            
            all_data.extend(klines)
            last_close_time = klines[-1][6]
            current_start = last_close_time + 1
            
            time.sleep(0.05)  # レート制限対策
            
        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"    エラー: {e}")
            time.sleep(1)
            continue
    
    return klines_to_dataframe(all_data)


def fetch_and_save_daily(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    start_date: str = None,
    days: int = None,
    output_dir: Path = None,
    verbose: bool = True,
) -> Dict[str, str]:
    """日別ファイルに分割して保存"""
    symbol_dir = output_dir / symbol / interval
    symbol_dir.mkdir(parents=True, exist_ok=True)
    
    # 期間の決定
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        start_dt = datetime.now() - timedelta(days=days or 30)
    
    if days:
        end_dt = start_dt + timedelta(days=days)
    else:
        end_dt = datetime.now()
    
    # 取得済みファイルをスキップ
    existing_dates = set()
    for f in symbol_dir.glob("*.csv"):
        try:
            existing_dates.add(f.stem)  # YYYY-MM-DD
        except:
            pass
    
    saved_files = {}
    current_dt = start_dt
    total_days = (end_dt - start_dt).days
    day_count = 0
    
    while current_dt < end_dt:
        date_str = current_dt.strftime("%Y-%m-%d")
        filepath = symbol_dir / f"{date_str}.csv"
        
        day_count += 1
        
        # 今日（まだ終わっていない日）はスキップせず最新を取得しても良いが、
        # ここでは「過去の確定データ」という意味で、既存ならスキップする方針は維持
        if date_str in existing_dates:
            if verbose:
                print(f"  [{day_count}/{total_days}] {date_str} スキップ（既存）")
            saved_files[date_str] = str(filepath)
            current_dt += timedelta(days=1)
            continue
        
        if verbose:
            print(f"  [{day_count}/{total_days}] {date_str} 取得中...", end="")
        
        df = fetch_one_day_data(symbol, interval, current_dt, verbose=False)
        
        if len(df) > 0:
            df.to_csv(filepath, index=False)
            saved_files[date_str] = str(filepath)
            if verbose:
                print(f" {len(df):,}バー")
        else:
            if verbose:
                print(" データなし")
        
        current_dt += timedelta(days=1)
        time.sleep(0.1)  # レート制限対策
    
    return saved_files


def fetch_multi_symbol_daily(
    symbols: List[str] = None,
    timeframes: List[str] = None,
    start_date: str = None,
    days: int = None,
    output_dir: Path = None,
    verbose: bool = True,
) -> Dict:
    """複数シンボル・複数時間軸のデータを日別ファイルで保存"""
    if not symbols:
        return {}
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    total_symbols = len(symbols)
    
    for sym_idx, symbol in enumerate(symbols, 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"[{sym_idx}/{total_symbols}] {symbol}")
            print("="*60)
        
        results[symbol] = {}
        
        for tf in timeframes:
            if verbose:
                print(f"\n  {tf}:")
            
            # シンボル固有の開始日を使用
            effective_start = start_date
            if not effective_start and not days:
                effective_start = get_symbol_start_date(symbol, verbose=(tf == timeframes[0]))
            
            saved = fetch_and_save_daily(
                symbol=symbol,
                interval=tf,
                start_date=effective_start,
                days=days,
                output_dir=output_dir,
                verbose=verbose,
            )
            results[symbol][tf] = saved
            
    return results


def perform_cleanup(target_symbols: List[str], output_dir: Path):
    """ターゲットリストにないシンボルのデータを削除"""
    print(f"\n🧹 Cleaning up data not in target list...")
    print(f"Target list ({len(target_symbols)}): {target_symbols[:5]}...")
    
    if not output_dir.exists():
        print("Output directory does not exist.")
        return

    # output_dir直下のディレクトリをチェック
    removed_count = 0
    for item in output_dir.iterdir():
        if item.is_dir():
            symbol_name = item.name
            # ディレクトリ名がターゲットリストになければ削除
            if symbol_name not in target_symbols:
                print(f"  - Removing: {symbol_name}")
                try:
                    shutil.rmtree(item)
                    removed_count += 1
                except Exception as e:
                    print(f"    Error removing {symbol_name}: {e}")
    
    print(f"Done. Removed {removed_count} directories.")


def main():
    parser = argparse.ArgumentParser(description="Binance OHLCVデータ取得（日別保存・複数通貨対応）")
    parser.add_argument("--symbol", type=str, default=None, help="通貨ペア（単一指定）")
    parser.add_argument("--symbols", type=str, nargs="+", default=None, help="複数通貨ペア指定")
    parser.add_argument("--top", type=int, default=None, help="上位N通貨を自動取得")
    parser.add_argument("--sort", type=str, default="volume", choices=["volume", "marketcap"], help="ソート基準 (volume/marketcap)")
    parser.add_argument("--interval", type=str, default=None, help="単一時間足")
    parser.add_argument("--timeframes", type=str, nargs="+", default=None, help="複数時間軸")
    parser.add_argument("--days", type=int, default=None, help="取得日数")
    parser.add_argument("--start-date", type=str, default=None, help="開始日 YYYY-MM-DD")
    parser.add_argument("--all", action="store_true", dest="fetch_all", help="全期間取得")
    parser.add_argument("--output", type=str, default="./data", help="出力ディレクトリ")
    parser.add_argument("--multi", action="store_true", help="多時間軸モード")
    parser.add_argument("--clean", action="store_true", help="リスト外のデータを削除する")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    # シンボルリストの決定
    if args.symbols:
        symbols = args.symbols
    elif args.top:
        symbols = get_top_symbols(limit=args.top, sort_by=args.sort)
    elif args.symbol:
        symbols = [args.symbol]
    else:
        # デフォルトなし（--topなどを強制したいが、互換性のためBTCのみ）
        print("⚠️ No symbols specified. Using BTCUSDT.")
        symbols = ["BTCUSDT"]
    
    # ユニーク化
    symbols = list(set(symbols))
    
    # 時間軸の決定
    if args.timeframes:
        timeframes = args.timeframes
    elif args.interval:
        timeframes = [args.interval]
    elif args.multi:
        timeframes = TIMEFRAMES
    else:
        timeframes = ["1m"]
    
    # 全期間モード
    start_date = args.start_date
    days = args.days
    if args.fetch_all:
        start_date = None  # fetch_and_save_dailyでシンボル固有の開始日を使用
        days = None
        print("\n✨ 全期間モード: 各シンボルの取引開始日から取得します")
    elif not start_date and not days:
        days = 30  # デフォルト30日
    
    print(f"\n対象シンボル: {len(symbols)}通貨")
    print(f"対象時間軸: {timeframes}")
    print(f"出力先: {output_dir}")
    
    # クリーンアップ（ダウンロード前に実行して、不要なものを消す）
    # あるいはダウンロード後に実行するか？
    # -> 引数で指定されたリストが「正」なので、ここに含まれないものは消す。
    if args.clean:
        perform_cleanup(symbols, output_dir)
    
    # 取得実行
    fetch_multi_symbol_daily(
        symbols=symbols,
        timeframes=timeframes,
        start_date=start_date,
        days=days,
        output_dir=output_dir,
        verbose=True,
    )
    
    # メタデータ保存
    metadata = {
        "symbols": symbols,
        "timeframes": timeframes,
        "fetch_time": datetime.now().isoformat(),
        "structure": "daily_files",
        "path_pattern": "{symbol}/{interval}/YYYY-MM-DD.csv",
    }
    
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("完了!")
    print("="*60)


if __name__ == "__main__":
    main()
