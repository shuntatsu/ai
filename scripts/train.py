"""
学習スクリプト

MarS Lite環境でPPOエージェントを学習
多時間軸・ランダムサンプリング・train/val分割・複数通貨対応
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, List

from mars_lite.data.preprocessing import preprocess_ohlcv
from mars_lite.data.multi_timeframe_loader import MultiTimeframeLoader, MultiSymbolLoader
from mars_lite.data.data_split import split_temporal, split_temporal_multi_tf, get_split_info
from mars_lite.env.mars_lite_env import MarsLiteEnv
# from mars_lite.env.multi_tf_env import MarsLiteMultiTFEnv
from mars_lite.env.cross_symbol_env import CrossSymbolEnv, SequentialSymbolEnv
from mars_lite.learning.agent import create_ppo_agent, train_agent, evaluate_agent
from mars_lite.learning.random_sampler import RandomEpisodeSampler, MultiModeEpisodeSampler
from mars_lite.learning.training_callback import (
    TrainingMetricsCallback,
    CheckpointCallback,
    get_metrics_history,
)
from mars_lite.utils.config import MarsLiteConfig, create_ppo_kwargs, create_env_kwargs

# 上位30通貨（fetch_binance.pyと同じリスト）

def load_available_symbols(data_dir: Path) -> List[str]:
    """
    データディレクトリから利用可能なシンボルリストを取得
    metadata.jsonがあればそれを優先、なければディレクトリをスキャン
    """
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                symbols = metadata.get("symbols", [])
                if symbols:
                    return symbols
        except Exception as e:
            print(f"Warning: Failed to load metadata.json: {e}")

    # メタデータがない場合はディレクトリをスキャン
    symbols = []
    if data_dir.exists():
        for item in data_dir.iterdir():
            if item.is_dir() and item.name.endswith("USDT"):
                symbols.append(item.name)
    return sorted(symbols)



def load_single_tf_data(data_path: str) -> pd.DataFrame:
    """
    単一時間軸OHLCVデータを読み込み・前処理
    
    Args:
        data_path: CSVファイルパス
        
    Returns:
        前処理済みDataFrame
    """
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()
    
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"])
        df = df.drop(columns=["date"])
    
    return preprocess_ohlcv(df)


def load_multi_tf_data(
    data_dir: str,
    config: MarsLiteConfig,
    start_date: str = None,
    end_date: str = None,
    limit_days: int = None,
) -> tuple:
    """
    多時間軸データを読み込み（日別ファイル形式対応）
    
    Args:
        data_dir: データディレクトリ
        config: 設定オブジェクト
        start_date: 開始日 YYYY-MM-DD
        end_date: 終了日 YYYY-MM-DD
        limit_days: 最大日数
        
    Returns:
        (base_data, higher_tf_data)
    """
    loader = MultiTimeframeLoader(
        data_dir=Path(data_dir),
        timeframes=list(config.timeframes),
        symbol=config.symbol,
        days=limit_days or config.data_days,
        preprocess=True,
    )
    
    loader.load_all(
        start_date=start_date,
        end_date=end_date,
        limit_days=limit_days,
    )
    base_data = loader.get_base_timeframe()
    higher_tf_data = loader.get_higher_timeframes()
    
    print(f"  データ形式: {'日別ファイル' if loader.is_daily_format else '単一ファイル'}")
    
    return base_data, higher_tf_data


def load_multi_symbol_data(
    data_dir: str,
    symbols: List[str],
    config: MarsLiteConfig,
    start_date: str = None,
    end_date: str = None,
    limit_days: int = None,
) -> Dict[str, tuple]:
    """
    複数通貨の多時間軸データを読み込み
    
    Args:
        data_dir: データディレクトリ
        symbols: 通貨リスト
        config: 設定オブジェクト
        start_date: 開始日
        end_date: 終了日
        limit_days: 最大日数
       
    Returns:
        {symbol: (base_data, higher_tf_data)}の辞書
    """
    loader = MultiSymbolLoader(
        data_dir=Path(data_dir),
        symbols=symbols,
        timeframes=list(config.timeframes),
        days=limit_days or config.data_days,
        preprocess=True,
    )
    
    all_data = loader.load_all(
        start_date=start_date,
        end_date=end_date,
        limit_days=limit_days,
    )
    
    result = {}
    for symbol in loader.loaded_symbols:
        loader_obj = loader.get_loader(symbol)
        result[symbol] = (
            loader_obj.get_base_timeframe(),
            loader_obj.get_higher_timeframes(),
        )
    
    print(f"  読み込み成功: {len(result)}/{len(symbols)}通貨")
    
    return result


def create_sample_data(n_bars: int = 10000) -> pd.DataFrame:
    """
    サンプルデータを生成（実データがない場合）
    
    Args:
        n_bars: バー数
        
    Returns:
        前処理済みDataFrame
    """
    np.random.seed(42)
    
    # ランダムウォークで価格生成
    returns = np.random.randn(n_bars) * 0.002
    close = 100 * np.exp(np.cumsum(returns))
    
    # OHLC生成
    high = close * (1 + np.abs(np.random.randn(n_bars)) * 0.003)
    low = close * (1 - np.abs(np.random.randn(n_bars)) * 0.003)
    open_ = low + (high - low) * np.random.rand(n_bars)
    
    # 出来高（時刻依存のU字型パターン）
    tod = np.arange(n_bars) % 1440
    base_volume = 1000 * (1 + 0.5 * np.cos(2 * np.pi * tod / 1440))
    volume = base_volume * np.random.exponential(1, n_bars)
    
    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    
    return preprocess_ohlcv(df)


def create_env_with_sampler(
    data: pd.DataFrame,
    higher_tf_data: dict,
    config: MarsLiteConfig,
    sampler: RandomEpisodeSampler = None,
) -> MarsLiteEnv:
    """
    環境を作成（MarsLiteEnvに統一）
    
    Args:
        data: ベース時間軸データ
        higher_tf_data: 上位時間軸データ
        config: 設定
        sampler: エピソードサンプラー（オプション）
        
    Returns:
        環境インスタンス
    """
    env_kwargs = create_env_kwargs(config)
    
    # data_dictを再構築
    # MarsLiteEnvは {symbol: {tf: df}} を期待するが、
    # ここでは単一通貨のデータを渡されている前提（CrossSymbolEnvでラップされる前）
    # しかし、MarsLiteEnvは内部で `data_dict[current_symbol]` にアクセスする。
    # ここでは便宜上、config.symbolをキーとして data_dict を作成する。
    
    symbol = config.symbol or "UNKNOWN"
    
    # higher_tf_data には '15m', '1h' などが入っている。
    # base_timeframe ('1m') も含める必要がある。
    tf_data = {config.base_timeframe: data}
    if higher_tf_data:
        tf_data.update(higher_tf_data)
        
    data_dict = {symbol: tf_data}
    
    # タイムフレームリストの構築
    # config.timeframes があればそれを使うが、data_dictにあるものを優先すべきか？
    # MarsLiteEnvは config.timeframes の順序を期待する。
    # 欠損チェックはEnv内で行われる。
    
    # higher_tf_lookback パラメータは MarsLiteEnv にはない（obs構成が違うため無視されるか、統合済み）
    # MarsLiteEnvの __init__ を確認すると:
    # n_lookback, y_impact, lambda_risk などを受け取る。
    # higher_tf_lookback は受け取らない（固定長か、内部ロジック）。
    # なので不要なkwargsを除去する。
    
    safe_kwargs = {k: v for k, v in env_kwargs.items() if k not in ["higher_tf_lookback"]}
    
    env = MarsLiteEnv(
        data_dict=data_dict,
        timeframes=list(config.timeframes),
        **safe_kwargs
    )
    
    return env


def main():
    parser = argparse.ArgumentParser(description="MarS Lite 学習スクリプト（複数通貨・多時間軸）")
    parser.add_argument("--data", type=str, default=None, help="データパス（CSVまたはディレクトリ）")
    parser.add_argument("--symbol", type=str, default=None, help="通貨ペア（単一通貨時）")
    parser.add_argument("--symbols", type=str, nargs="+", default=None, help="複数通貨ペア")
    parser.add_argument("--top", type=int, default=None, help="上位N通貨")
    parser.add_argument("--start-date", type=str, default=None, help="開始日 YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="終了日 YYYY-MM-DD")
    parser.add_argument("--limit-days", type=int, default=None, help="最大日数")
    parser.add_argument("--all", action="store_true", dest="use_all_data", help="全期間データを使用")
    parser.add_argument("--config", type=str, default=None, help="設定ファイル（JSON）")
    parser.add_argument("--output", type=str, default="./output", help="出力ディレクトリ")
    parser.add_argument("--timesteps", type=int, default=100000, help="総学習ステップ数")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--verbose", type=int, default=1, help="ログ出力レベル")
    parser.add_argument("--multi-tf", action="store_true", help="多時間軸モード")
    parser.add_argument("--no-split", action="store_true", help="データ分割を無効化")
    parser.add_argument("--serve", action="store_true", help="UIサーバーを起動")
    parser.add_argument("--port", type=int, default=8000, help="UIサーバーポート")
    parser.add_argument("--checkpoint-freq", type=int, default=10000, help="チェックポイント保存間隔")
    args = parser.parse_args()
    
    # 複数通貨リストの決定
    # まずデータディレクトリから利用可能なシンボルを取得
    available_symbols = []
    if args.data:
        data_path = Path(args.data)
        if data_path.is_dir():
            available_symbols = load_available_symbols(data_path)
            
    if args.symbols:
        symbols = args.symbols
        multi_symbol_mode = True
    elif args.top:
        if available_symbols:
             symbols = available_symbols[:args.top]
        else:
             print("Warning: No data found to select top symbols from. Using fallback list.")
             # Fallback list (Historical Top 10)
             symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT"][:args.top]
        multi_symbol_mode = True
    elif args.symbol:
        symbols = [args.symbol]
        multi_symbol_mode = False
    else:
        # デフォルト: 利用可能な全シンボル、またはBTC
        if available_symbols:
            symbols = available_symbols
            multi_symbol_mode = True
            print(f"Using all available symbols from data directory: {len(symbols)} found.")
        else:
            symbols = ["BTCUSDT"]
            multi_symbol_mode = False
    
    # --all フラグ処理
    if args.use_all_data:
        args.start_date = None
        args.limit_days = None
        print("\n✨ 全期間モード: 利用可能な全データを使用します")
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 設定読み込み
    if args.config:
        config = MarsLiteConfig.load(args.config)
    else:
        config = MarsLiteConfig()
    
    config.seed = args.seed
    config.total_timesteps = args.timesteps
    config.save_dir = str(output_dir)
    config.symbol = symbols[0]  # メイン通貨（最初の通貨）
    
    if args.multi_tf:
        config.use_multi_tf = True
    
    # 設定保存
    config.save(str(output_dir / "config.json"))
    
    print("=" * 60)
    print("MarS Lite 学習開始（複数通貨・多時間軸）")
    print("=" * 60)
    if multi_symbol_mode:
        print(f"通貨: {len(symbols)}通貨 {symbols[:5]}{'...' if len(symbols) > 5 else ''}")
    else:
        print(f"通貨: {symbols[0]}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"総ステップ数: {config.total_timesteps:,}")
    print(f"多時間軸モード: {config.use_multi_tf}")
    print(f"時間軸: {config.timeframes}")
    print("=" * 60)
    
    # データ読み込み
    multi_symbol_data = {}  # {symbol: (base_data, higher_tf_data)}
    
    if args.data:
        data_path = Path(args.data)
        
        if data_path.is_dir():
            # ディレクトリ指定: 多時間軸/日別データ
            print(f"\\nデータ読み込み中: {data_path}")
            config.data_dir = str(data_path)
            
            if multi_symbol_mode:
                # 複数通貨読み込み
                multi_symbol_data = load_multi_symbol_data(
                    str(data_path),
                    symbols,
                    config,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    limit_days=args.limit_days,
                )
                # 最初の通貨をメイン表示
                if multi_symbol_data:
                    first_symbol = list(multi_symbol_data.keys())[0]
                    base_data, higher_tf_data = multi_symbol_data[first_symbol]
                    print(f"  {first_symbol} ベースデータ: {len(base_data):,}バー")
            else:
                # 単一通貨読み込み
                base_data, higher_tf_data = load_multi_tf_data(
                    str(data_path), 
                    config,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    limit_days=args.limit_days,
                )
                print(f"ベースデータ件数: {len(base_data):,}バー")
                if higher_tf_data:
                    print(f"上位TF: {list(higher_tf_data.keys())}")
        else:
            # ファイル指定: 単一時間軸
            print(f"単一時間軸データ読み込み中: {data_path}")
            base_data = load_single_tf_data(str(data_path))
            config.use_multi_tf = False
            higher_tf_data = {}
    else:
        print("サンプルデータを生成中...")
        base_data = create_sample_data(n_bars=20000)
        config.use_multi_tf = False
        higher_tf_data = {}
    
    # データ分割
    if not args.no_split:
        print("\nデータ分割中...")
        
        if config.use_multi_tf and higher_tf_data:
            # 多時間軸データも分割
            all_data = {config.base_timeframe: base_data, **higher_tf_data}
            train_data, val_data, test_data = split_temporal_multi_tf(
                all_data,
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
                test_ratio=config.test_ratio,
            )
            
            train_base = train_data[config.base_timeframe]
            val_base = val_data[config.base_timeframe]
            train_higher = {k: v for k, v in train_data.items() if k != config.base_timeframe}
            val_higher = {k: v for k, v in val_data.items() if k != config.base_timeframe}
        else:
            # 単一時間軸
            train_base, val_base, test_base = split_temporal(
                base_data,
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
                test_ratio=config.test_ratio,
            )
            train_higher = {}
            val_higher = {}
        
        # 分割情報表示
        split_info = get_split_info(train_base, val_base, 
                                    test_data.get(config.base_timeframe, pd.DataFrame()) 
                                    if config.use_multi_tf else test_base)
        print(f"  Train: {split_info['train']['bars']:,}バー ({split_info['train']['ratio']*100:.1f}%)")
        print(f"  Val: {split_info['val']['bars']:,}バー ({split_info['val']['ratio']*100:.1f}%)")
        print(f"  Test: {split_info['test']['bars']:,}バー ({split_info['test']['ratio']*100:.1f}%)")
        
        # 分割情報保存
        with open(output_dir / "split_info.json", "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
    else:
        train_base = base_data
        val_base = base_data
        train_higher = higher_tf_data
        val_higher = higher_tf_data
    
    # 環境作成
    print("\n環境を作成中...")
    
    if multi_symbol_mode and multi_symbol_data:
        # 複数通貨：各通貨ごとに環境を作成してCrossSymbolEnvでラップ
        train_envs = {}
        eval_envs = {}
        
        for symbol, (base, higher) in multi_symbol_data.items():
            # データ分割
            if not args.no_split:
                if config.use_multi_tf and higher:
                    all_d = {config.base_timeframe: base, **higher}
                    train_d, val_d, _ = split_temporal_multi_tf(
                        all_d,
                        train_ratio=config.train_ratio,
                        val_ratio=config.val_ratio,
                        test_ratio=config.test_ratio,
                    )
                    train_b = train_d[config.base_timeframe]
                    val_b = val_d[config.base_timeframe]
                    train_h = {k: v for k, v in train_d.items() if k != config.base_timeframe}
                    val_h = {k: v for k, v in val_d.items() if k != config.base_timeframe}
                else:
                    train_b, val_b, _ = split_temporal(base, config.train_ratio, config.val_ratio, config.test_ratio)
                    train_h, val_h = {}, {}
            else:
                train_b, val_b = base, base
                train_h, val_h = higher, higher
            
            # 環境作成
            train_envs[symbol] = create_env_with_sampler(train_b, train_h, config)
            eval_envs[symbol] = create_env_with_sampler(val_b, val_h, config)
        
        # CrossSymbolEnvでラップ
        train_env = CrossSymbolEnv(train_envs, seed=config.seed)
        eval_env = SequentialSymbolEnv(eval_envs)  # 評価は順番に
        
        print(f"  通貨ごとの環境作成: {len(train_envs)}環境")
    else:
        # 単一通貨：従来通り
        train_env = create_env_with_sampler(train_base, train_higher, config)
        eval_env = create_env_with_sampler(val_base, val_higher, config)
    
    print(f"観測空間: {train_env.observation_space.shape}")
    print(f"行動空間: {train_env.action_space.shape}")
    
    # エージェント作成
    print("\nPPOエージェントを作成中...")
    ppo_kwargs = create_ppo_kwargs(config)
    agent = create_ppo_agent(train_env, verbose=args.verbose, **ppo_kwargs)
    
    # コールバック準備
    callbacks = []
    
    # メトリクスコールバック（常に有効）
    metrics_callback = TrainingMetricsCallback(
        total_timesteps=config.total_timesteps,
        log_freq=1,
        verbose=args.verbose,
    )
    callbacks.append(metrics_callback)
    
    # チェックポイントコールバック
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="model",
        verbose=args.verbose,
    )
    callbacks.append(checkpoint_callback)
    
    # UIサーバー起動（--serveオプション時）
    server_thread = None
    if args.serve:
        import threading
        from mars_lite.server.metrics_server import run_server
        
        def start_server():
            run_server(
                host="0.0.0.0",
                port=args.port,
                output_dir=str(output_dir),
            )
        
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        print(f"\n📡 UIサーバー起動: http://localhost:{args.port}")
        print(f"   WebSocket: ws://localhost:{args.port}/ws/metrics")
    
    # 学習
    print("\n学習開始...")
    agent = train_agent(
        agent=agent,
        total_timesteps=config.total_timesteps,
        callbacks=callbacks,
        eval_env=eval_env,
        eval_freq=10000,
        n_eval_episodes=5,
        save_path=str(output_dir),
    )
    
    # 最終評価
    print("\n最終評価中...")
    eval_results = evaluate_agent(agent, eval_env, n_episodes=10)
    
    print("=" * 60)
    print("学習完了")
    print("=" * 60)
    print(f"平均報酬: {eval_results['mean_reward']:.4f} ± {eval_results['std_reward']:.4f}")
    print(f"平均エピソード長: {eval_results['mean_length']:.1f}")
    
    if eval_results["execution_stats"]:
        mean_trades = np.mean([s["n_trades"] for s in eval_results["execution_stats"]])
        mean_pov = np.mean([s["mean_pov"] for s in eval_results["execution_stats"]])
        print(f"平均取引回数: {mean_trades:.1f}")
        print(f"平均POV: {mean_pov:.4f}")
    
    # 結果保存
    with open(output_dir / "eval_results.json", "w", encoding="utf-8") as f:
        serializable = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in eval_results.items()
            if k != "execution_stats"
        }
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果を保存しました: {output_dir}")


if __name__ == "__main__":
    main()
