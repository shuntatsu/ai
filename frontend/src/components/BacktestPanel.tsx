import React, { useState } from 'react';
import { Play, BarChart3, Target, TrendingDown, Award } from 'lucide-react';

interface BacktestResult {
    metrics: {
        n_episodes: number;
        mean_reward: number;
        std_reward: number;
        sharpe_ratio?: number;
        max_drawdown?: number;
        win_rate?: number;
        n_trades?: number;
        mean_impact?: number;
    };
    episode_rewards: number[];
}

interface BacktestPanelProps {
    apiUrl?: string;
    selectedModel?: string;
}

/**
 * バックテストパネル
 * 
 * モデル選択、期間設定、バックテスト実行、結果表示を提供。
 */
export const BacktestPanel: React.FC<BacktestPanelProps> = ({
    apiUrl = 'http://localhost:8001',
    selectedModel,
}) => {
    const [symbol, setSymbol] = useState('BTCUSDT');
    const [episodes, setEpisodes] = useState(10);
    const [running, setRunning] = useState(false);
    const [result, setResult] = useState<BacktestResult | null>(null);
    const [error, setError] = useState<string | null>(null);

    // バックテスト実行
    const runBacktest = async () => {
        if (!selectedModel) {
            setError('Please select a model first');
            return;
        }

        setRunning(true);
        setError(null);
        setResult(null);

        try {
            const response = await fetch(`${apiUrl}/api/backtest`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_id: selectedModel,
                    symbol,
                    n_episodes: episodes,
                }),
            });

            if (!response.ok) {
                throw new Error('Backtest failed');
            }

            const data = await response.json();
            setResult(data);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Unknown error');
        } finally {
            setRunning(false);
        }
    };

    // 指標カード
    const MetricCard: React.FC<{
        label: string;
        value: string | number;
        icon: React.ElementType;
        color?: string;
    }> = ({ label, value, icon: Icon, color = 'text-primary' }) => (
        <div className="bg-surface/50 rounded p-3 border border-border">
            <div className="flex items-center gap-2 mb-1">
                <Icon size={12} className={color} />
                <span className="text-xs text-muted">{label}</span>
            </div>
            <div className={`text-lg font-semibold ${color}`}>{value}</div>
        </div>
    );

    return (
        <div className="card h-full flex flex-col">
            <div className="flex items-center gap-2 mb-4">
                <BarChart3 size={16} className="text-secondary" />
                <h3 className="text-sm font-semibold">BACKTEST</h3>
            </div>

            {/* 設定 */}
            <div className="space-y-3 mb-4">
                <div>
                    <label className="text-xs text-muted block mb-1">Model</label>
                    <div className="font-mono text-sm p-2 bg-surface rounded border border-border truncate">
                        {selectedModel || 'No model selected'}
                    </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                    <div>
                        <label className="text-xs text-muted block mb-1">Symbol</label>
                        <select
                            value={symbol}
                            onChange={(e) => setSymbol(e.target.value)}
                            className="w-full p-2 bg-surface rounded border border-border text-sm"
                        >
                            <option value="BTCUSDT">BTCUSDT</option>
                            <option value="ETHUSDT">ETHUSDT</option>
                            <option value="SOLUSDT">SOLUSDT</option>
                            <option value="BNBUSDT">BNBUSDT</option>
                        </select>
                    </div>
                    <div>
                        <label className="text-xs text-muted block mb-1">Episodes</label>
                        <input
                            type="number"
                            value={episodes}
                            onChange={(e) => setEpisodes(parseInt(e.target.value) || 10)}
                            min={1}
                            max={100}
                            className="w-full p-2 bg-surface rounded border border-border text-sm"
                        />
                    </div>
                </div>

                <button
                    onClick={runBacktest}
                    disabled={running || !selectedModel}
                    className={`w-full py-2 px-4 rounded font-medium flex items-center justify-center gap-2 transition-colors ${running || !selectedModel
                        ? 'bg-muted/20 text-muted cursor-not-allowed'
                        : 'bg-primary text-background hover:bg-primary/80'
                        }`}
                >
                    <Play size={16} />
                    {running ? 'Running...' : 'Run Backtest'}
                </button>
            </div>

            {error && (
                <div className="text-red-400 text-xs mb-3 p-2 bg-red-500/10 rounded">
                    {error}
                </div>
            )}

            {/* 結果 */}
            {result && (
                <div className="flex-1 overflow-y-auto">
                    <div className="text-xs text-muted mb-2">Results</div>

                    <div className="grid grid-cols-2 gap-2 mb-4">
                        <MetricCard
                            label="Mean Reward"
                            value={result.metrics.mean_reward.toFixed(3)}
                            icon={Award}
                            color="text-primary"
                        />
                        <MetricCard
                            label="Sharpe Ratio"
                            value={result.metrics.sharpe_ratio?.toFixed(2) ?? 'N/A'}
                            icon={TrendingDown}
                            color={
                                (result.metrics.sharpe_ratio ?? 0) > 1
                                    ? 'text-green-400'
                                    : 'text-yellow-400'
                            }
                        />
                        <MetricCard
                            label="Win Rate"
                            value={`${((result.metrics.win_rate ?? 0) * 100).toFixed(1)}%`}
                            icon={Target}
                            color={
                                (result.metrics.win_rate ?? 0) > 0.5
                                    ? 'text-green-400'
                                    : 'text-red-400'
                            }
                        />
                        <MetricCard
                            label="Max Drawdown"
                            value={result.metrics.max_drawdown?.toFixed(2) ?? 'N/A'}
                            icon={TrendingDown}
                            color="text-red-400"
                        />
                    </div>

                    {/* エピソード報酬リスト */}
                    <div className="text-xs text-muted mb-2">Episode Rewards</div>
                    <div className="space-y-1 max-h-40 overflow-y-auto">
                        {result.episode_rewards.map((reward, i) => (
                            <div
                                key={i}
                                className="flex justify-between text-xs py-1 px-2 bg-surface/50 rounded"
                            >
                                <span className="text-muted">Episode {i + 1}</span>
                                <span className={reward > 0 ? 'text-green-400' : 'text-red-400'}>
                                    {reward.toFixed(4)}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};
