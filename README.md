# MarS Lite

OHLCVデータから仮想マーケットインパクトをシミュレートし、強化学習エージェントに**注文分割（Iceberg相当）**・**時間リスクとのトレードオフ（Almgren-Chriss）**を学習させる環境。

## 特徴

- **板情報不要**: OHLCVのみからマーケットインパクトを推定
- **Almgren-Chriss報酬**: 執行コスト最小化 + 在庫リスクペナルティ
- **Look-ahead bias防止**: Next Open基準価格、時刻別期待出来高使用
- **PBT-MAP-Elites**: 多様なスペシャリスト群の同時学習
- **環境適応推論**: 市場レジームに応じた個体選択
- **Multi-Timeframe**: 1m, 15m, 1h, 4h, 1dのデータを統合して学習
- **Cross-Symbol Learning**: 複数通貨によるランダム交差学習で汎化性能を向上
- **Smart Data Fetching**: 通貨ごとの上場日を自動検出し、無駄なAPIリクエストを削減
- **Market Cap Selection**: 時価総額（Market Cap）順での上位通貨選択に対応（ステーブルコイン除外機能付き）

## Architecture

```mermaid
graph TD
    subgraph Data Pipeline
        Binance[Binance API] -->|Fetch| RawData[Raw OHLCV]
        RawData -->|Split| DailyFiles[Daily Files (YYYY-MM-DD.csv)]
        DailyFiles -->|Load| MultiTF[MultiTimeframeLoader]
        MultiTF -->|Align| AlignedData[Aligned Multi-TF Data]
        AlignedData -->|Split| SplitData[Train/Val/Test Split]
    end

    subgraph Environment
        SplitData -->|Feed| SimEnv[MarsLiteEnv]
        SimEnv -->|Wrap| TFEnv[MarsLiteMultiTFEnv]
        TFEnv -->|Wrap| CrossEnv[CrossSymbolEnv]
        
        subgraph Market Simulation
            TFEnv -->|Obs| Agent
            Agent -->|Action| TFEnv
            TFEnv -->|Step| Match[Matching Engine]
            Match -->|Exec Price| Reward[Reward Function]
            Reward -->|Scalar| Agent
        end
    end

    subgraph Learning System
        CrossEnv -->|Samples| PPO[PPO Agent]
        PPO -->|Update| Policy[Policy Network]
        
        Sampler[RandomEpisodeSampler] -->|Reset Idx| CrossEnv
    end
```

## インストール

```bash
cd mars_lite
pip install -r requirements.txt
pip install -e .
```

## 使い方

### データ取得
```bash
# 上位30通貨のデータを取得（時価総額順・上場日から全期間）
# --sort marketcap で時価総額順（ステーブルコイン除外）、デフォルトは volume
# デフォルトでは 1m 足のみ取得します（学習時に自動で上位足を作成するため推奨）
# 全ての時間軸ファイルを物理的に保存したい場合は --multi を追加してください
python scripts/fetch_binance.py --top 30 --sort marketcap --all --output ./data --clean --multi
```

### データ整理（クリーンアップ）
指定した通貨リストに含まれない古いデータを削除したい場合は `--clean` オプションを使用します。
例えば、現在の上位30通貨**以外**のデータを削除するには：

```bash
python scripts/fetch_binance.py --top 30 --clean --output ./data --all --sort marketcap
```

全データを削除したい場合は、`data` ディレクトリを直接削除してください。

### 学習
```bash
# 複数通貨・多時間軸で学習
python scripts/train.py --data ./data --top 10 --multi-tf --timesteps 500000

# 単一通貨で学習
python scripts/train.py --data ./data --symbol BTCUSDT --multi-tf
```

### 評価
```bash
python scripts/evaluate.py --model ./output/best_model.zip --episodes 20
```

### Pythonから直接使用
```python
import pandas as pd
from mars_lite.env import MarsLiteEnv, MarsLiteMultiTFEnv, CrossSymbolEnv
from mars_lite.data import load_multi_symbol_data, MarsLiteConfig

# データ読み込み
config = MarsLiteConfig()
data_map = load_multi_symbol_data("./data", symbols=["BTCUSDT", "ETHUSDT"], config=config)

# 環境構築（手動）
envs = {}
for sym, (base, higher) in data_map.items():
    envs[sym] = MarsLiteMultiTFEnv(
        data_1m=base, 
        higher_tf_data=higher, 
        **create_env_kwargs(config)
    )

env = CrossSymbolEnv(envs)

# Gymnasium標準インターフェース
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

## ディレクトリ構成

```
mars_lite/
├── mars_lite/
│   ├── data/          # データ前処理
│   ├── env/           # Gymnasium環境
│   ├── learning/      # PPO/Population管理
│   ├── evolution/     # PBT/MAP-Elites
│   └── utils/         # 設定/評価指標
├── tests/             # ユニットテスト
├── scripts/           # 学習/評価スクリプト
└── requirements.txt
```

## テスト実行

```bash
python -m pytest tests/ -v
```

## 主要パラメータ

| パラメータ | 説明 | デフォルト |
|-----------|------|-----------|
| `y_impact` | インパクト係数 | 0.5 |
| `lambda_risk` | 在庫リスク係数 | 0.001 |
| `initial_inventory` | 初期在庫 | 1000 |
| `max_steps` | 最大ステップ数 | 1440 |

## ライセンス

MIT
