import React, { useMemo } from 'react';
import {
    Zap,
    TrendingUp,
    DollarSign,
    Settings,
    Save,
    Cpu,
    Wifi,
    WifiOff,
    Play,
    Square,
    Loader2,
    Activity
} from 'lucide-react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend
} from 'recharts';
import { StatsCard } from './StatsCard';
import { TrainingChart } from './TrainingChart';
import type { TrainingDataPoint } from './TrainingChart';
import { LogTerminal } from './LogTerminal';
import type { LogEntry } from './LogTerminal';
import { DetailedLossChart } from './DetailedLossChart';
import { TradingVisualizer } from './TradingVisualizer';
import { useWebSocket } from '../hooks/useWebSocket';
import { useTrainingControl } from '../hooks/useTrainingControl';

import { TrainingConfigPanel } from './TrainingConfigPanel';
import type { TrainingConfig } from '../hooks/useTrainingControl';

// WebSocketサーバーURL（環境変数または固定）
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8001/ws/metrics';

export const Dashboard: React.FC = () => {
    // WebSocket接続
    const { isConnected, lastMessage, metricsHistory, error } = useWebSocket({
        url: WS_URL,
        reconnectInterval: 3000,
        maxRetries: 10,
    });

    // 学習制御
    const {
        trainingStatus,
        isLoading: isTrainingLoading,
        progressMessage,
        startTraining,
        stopTraining,
        getDefaultConfig
    } = useTrainingControl();

    // UI State
    const [isSettingsOpen, setIsSettingsOpen] = React.useState(false);
    const [config, setConfig] = React.useState<TrainingConfig | null>(null);
    const [defaultConfig, setDefaultConfig] = React.useState<TrainingConfig | null>(null);

    // 初期設定ロード
    React.useEffect(() => {
        const loadConfig = async () => {
            const defaults = await getDefaultConfig();

            // LocalStorageから読み込み
            const saved = localStorage.getItem('mars_lite_training_config');
            let initial = defaults;

            if (saved && defaults) {
                try {
                    const parsed = JSON.parse(saved);
                    // マージ (新しいパラメータが増えた場合などに備えてデフォルトをベースに)
                    initial = { ...defaults, ...parsed };
                } catch (e) {
                    console.error("Failed to parse saved config", e);
                }
            }

            if (initial) {
                setDefaultConfig(defaults);
                setConfig(initial);
            }
        };
        loadConfig();
    }, [getDefaultConfig]);

    // 設定保存時にLocalStorageにも保存
    const handleSaveConfig = (newConfig: TrainingConfig) => {
        setConfig(newConfig);
        localStorage.setItem('mars_lite_training_config', JSON.stringify(newConfig));
    };

    // メトリクスから報酬チャートデータを生成
    const rewardChartData: TrainingDataPoint[] = useMemo(() => {
        return metricsHistory
            .filter((m) => m.type === 'training_step' && m.mean_reward != null)
            .map((m) => ({
                step: m.step,
                meanReward: m.mean_reward ?? 0,
                valueLoss: m.value_loss ?? 0,
                entropy: m.entropy_loss ? Math.abs(m.entropy_loss) : 0,
            }));
    }, [metricsHistory]);

    // メトリクスからログエントリを生成
    const logs: LogEntry[] = useMemo(() => {
        return metricsHistory
            .filter((m) => m.type === 'training_step' || m.type === 'training_start' || m.type === 'training_end' || m.type === 'log')
            .slice(-50)
            .map((m, i) => {
                const date = new Date(m.timestamp * 1000);
                const timeStr = date.toLocaleTimeString('ja-JP', { hour12: false });

                let level: 'info' | 'success' | 'warning' | 'error' = 'info';
                let message = '';

                if (m.type === 'training_start') {
                    level = 'success';
                    message = `Training started. Total: ${m.total_timesteps?.toLocaleString() ?? 'N/A'} steps`;
                } else if (m.type === 'training_end') {
                    level = 'success';
                    message = `Training completed! Final reward: ${(m as any).final_mean_reward?.toFixed(2) ?? 'N/A'}`;
                } else if (m.type === 'log') {
                    level = 'info';
                    message = (m as any).message || JSON.stringify(m);
                } else {
                    message = `Step ${m.step?.toLocaleString() ?? 'N/A'} | Reward: ${m.mean_reward?.toFixed(2) ?? 'N/A'} | Loss: ${m.policy_loss?.toFixed(4) ?? 'N/A'}`;
                }

                return {
                    id: `${m.timestamp}-${i}`,
                    timestamp: timeStr,
                    level,
                    message,
                };
            });
    }, [metricsHistory]);

    // 現在の統計
    const currentStats = useMemo(() => {
        if (!lastMessage) {
            return {
                meanReward: 'N/A',
                progress: 0,
                step: 0,
                episodes: 0,
                policyLoss: 'N/A',
                winRate: 'N/A',
                maxDrawdown: 'N/A',
                longPct: 0,
                shortPct: 0,
                holdPct: 0,
            };
        }

        return {
            meanReward: typeof lastMessage.mean_reward === 'number' ? lastMessage.mean_reward.toFixed(2) : 'N/A',
            progress: lastMessage.progress ?? 0,
            step: lastMessage.step ?? 0,
            episodes: lastMessage.n_episodes ?? 0,
            policyLoss: typeof lastMessage.policy_loss === 'number' ? lastMessage.policy_loss.toFixed(4) : 'N/A',
            winRate: typeof lastMessage.mean_win_rate === 'number' ? lastMessage.mean_win_rate.toFixed(1) : 'N/A',
            maxDrawdown: typeof lastMessage.mean_max_drawdown === 'number' ? lastMessage.mean_max_drawdown.toFixed(1) : 'N/A',
            longPct: typeof lastMessage.long_pct === 'number' ? lastMessage.long_pct : 0,
            shortPct: typeof lastMessage.short_pct === 'number' ? lastMessage.short_pct : 0,
            holdPct: typeof lastMessage.hold_pct === 'number' ? lastMessage.hold_pct : 0,
            // New Metrics
            apy: typeof lastMessage.apy === 'number' ? lastMessage.apy.toFixed(1) : 'N/A',
            maxTradeProfit: typeof lastMessage.max_trade_profit === 'number' ? lastMessage.max_trade_profit.toFixed(2) : 'N/A',
            maxTradeLoss: typeof lastMessage.max_trade_loss === 'number' ? lastMessage.max_trade_loss.toFixed(2) : 'N/A',
            tradesPerDay: typeof lastMessage.trades_per_day === 'number' ? lastMessage.trades_per_day.toFixed(1) : 'N/A',
        };
    }, [lastMessage]);

    // 学習中かどうか
    const isTraining = isConnected && (trainingStatus.is_running || lastMessage?.type === 'training_step');

    // Start/Stop ボタンのクリックハンドラ
    const handleTrainingToggle = async () => {
        if (trainingStatus.is_running) {
            await stopTraining();
        } else {
            // 現在の設定を使って開始
            await startTraining(config || undefined);
        }
    };

    return (
        <div className="min-h-screen bg-background text-text p-6 flex flex-col gap-6 font-sans">
            {/* ヘッダー */}
            <header className="flex justify-between items-center card py-3">
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <Cpu className="text-primary animate-pulse" size={24} />
                        <h1 className="text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary">
                            MarS Lite <span className="text-muted text-sm font-normal ml-2">v0.5.0</span>
                        </h1>
                    </div>
                    <div className="h-6 w-px bg-border" />

                    {/* 接続状態 */}
                    <div className="flex items-center gap-2 px-3 py-1 rounded bg-surface border border-border">
                        {isConnected ? (
                            <>
                                <Wifi size={14} className="text-primary" />
                                <span className="text-xs font-mono text-primary">CONNECTED</span>
                            </>
                        ) : (
                            <>
                                <WifiOff size={14} className="text-red-400" />
                                <span className="text-xs font-mono text-red-400">DISCONNECTED</span>
                            </>
                        )}
                    </div>

                    {/* 学習状態 */}
                    <div className="flex items-center gap-2 px-3 py-1 rounded bg-surface border border-border">
                        <div className={`w-2 h-2 rounded-full ${isTraining ? 'bg-primary animate-pulse' : 'bg-muted'}`} />
                        <span className="text-xs font-mono text-muted">{isTraining ? 'TRAINING' : 'IDLE'}</span>
                    </div>

                    {/* Start/Stop ボタン */}
                    <div className="flex flex-col items-end gap-1">
                        <button
                            onClick={handleTrainingToggle}
                            disabled={isTrainingLoading || trainingStatus.status === 'stopping'}
                            className={`flex items-center gap-2 px-4 py-2 rounded font-medium text-sm transition-all
                                ${trainingStatus.is_running
                                    ? 'bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/50'
                                    : 'bg-primary/20 hover:bg-primary/30 text-primary border border-primary/50'
                                }
                                disabled:opacity-50 disabled:cursor-not-allowed
                            `}
                        >
                            {isTrainingLoading ? (
                                <Loader2 size={16} className="animate-spin" />
                            ) : trainingStatus.is_running ? (
                                <Square size={16} />
                            ) : (
                                <Play size={16} />
                            )}
                            <span>
                                {trainingStatus.status === 'stopping' ? 'Stopping...'
                                    : trainingStatus.is_running ? 'Stop Training'
                                        : 'Start Training'}
                            </span>
                        </button>
                        {progressMessage && (
                            <span className="text-[10px] text-primary animate-pulse font-mono">
                                {progressMessage}
                            </span>
                        )}
                    </div>
                </div>

                <div className="flex items-center gap-4 w-1/3 justify-end">
                    <div className="flex-1 flex flex-col gap-1 max-w-[200px]">
                        <div className="flex justify-between text-xs text-muted font-mono">
                            <span>PROGRESS</span>
                            <span>{currentStats.step.toLocaleString()} / {config?.max_steps?.toLocaleString() ?? '∞'}</span>
                        </div>
                        <div className="h-1.5 bg-surface border border-border rounded-full overflow-hidden">
                            <div
                                className="h-full bg-gradient-to-r from-primary to-secondary transition-all duration-300"
                                style={{ width: `${Math.min(currentStats.progress, 100)}%` }}
                            />
                        </div>
                    </div>

                    {/* Manual Save Button */}
                    <button
                        onClick={async () => {
                            try {
                                const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';
                                const res = await fetch(`${API_BASE}/api/training/save`, { method: 'POST' });
                                if (res.ok) alert("Checkpoint saved!");
                            } catch (e) {
                                alert("Failed to save checkpoint");
                            }
                        }}
                        disabled={!isTraining}
                        className="p-2 rounded hover:bg-white/5 transition-colors border border-transparent hover:border-border cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Save Checkpoint"
                    >
                        <Save size={18} className="text-muted hover:text-green-400 transition-colors" />
                    </button>

                    <button
                        onClick={() => setIsSettingsOpen(true)}
                        className="p-2 rounded hover:bg-white/5 transition-colors border border-transparent hover:border-border relative z-10 cursor-pointer"
                        title="Configure Training"
                    >
                        <Settings size={18} className="text-muted hover:text-primary transition-colors" />
                    </button>
                </div>
            </header>

            {/* メイングリッド */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 flex-1">

                {/* 左カラム: チャート */}
                <div className="lg:col-span-3 flex flex-col gap-6">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <StatsCard
                            title="Mean Reward"
                            value={currentStats.meanReward}
                            icon={TrendingUp}
                            valueClassName="text-primary"
                        />
                        <StatsCard
                            title="Win Rate"
                            value={currentStats.winRate === 'N/A' ? 'N/A' : `${currentStats.winRate}%`}
                            icon={TrendingUp}
                            valueClassName="text-green-400"
                        />
                        <StatsCard
                            title="Max Drawdown"
                            value={currentStats.maxDrawdown === 'N/A' ? 'N/A' : `${currentStats.maxDrawdown}%`}
                            icon={Zap}
                            valueClassName="text-red-400"
                        />
                    </div>

                    {/* Trading Visualizer */}
                    <div className="mb-8">
                        <TradingVisualizer lastMessage={lastMessage} />
                    </div>

                    {/* Charts Row 1: Main Training & Loss */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
                        {/* Main Chart */}
                        <div className="lg:col-span-2">
                            <TrainingChart data={rewardChartData} />
                        </div>
                        {/* Loss Chart */}
                        <div className="lg:col-span-1">
                            <DetailedLossChart data={metricsHistory} />
                        </div>
                    </div>

                    {/* Charts Row 2: Secondary Metrics (Win Rate & Trade Freq) */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                        {/* Win Rate Chart */}
                        <div className="card h-[300px] flex flex-col p-4">
                            <h3 className="text-sm font-medium text-muted mb-4 flex items-center gap-2">
                                <TrendingUp size={16} /> WIN RATE HISTORY
                            </h3>
                            <div className="flex-1 w-full min-h-0">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={metricsHistory.filter(m => m.mean_win_rate !== null)}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                                        <XAxis
                                            dataKey="step"
                                            stroke="#71717a"
                                            fontSize={10}
                                            tickFormatter={(val) => Number.isFinite(val) ? `${(val / 1000).toFixed(0)}k` : ''}
                                            tickLine={false}
                                            axisLine={false}
                                        />
                                        <YAxis domain={[0, 100]} stroke="#71717a" fontSize={10} tickLine={false} axisLine={false} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px' }}
                                            itemStyle={{ color: '#4ade80' }}
                                        />
                                        <Line type="monotone" dataKey="mean_win_rate" stroke="#4ade80" dot={false} strokeWidth={2} name="Win Rate %" />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Trade Frequency Chart */}
                        <div className="card h-[300px] flex flex-col p-4">
                            <h3 className="text-sm font-medium text-muted mb-4 flex items-center gap-2">
                                <Activity size={16} /> TRADES PER DAY
                            </h3>
                            <div className="flex-1 w-full min-h-0">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={metricsHistory.filter(m => m.trades_per_day !== null)}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                                        <XAxis
                                            dataKey="step"
                                            stroke="#71717a"
                                            fontSize={10}
                                            tickFormatter={(val) => Number.isFinite(val) ? `${(val / 1000).toFixed(0)}k` : ''}
                                            tickLine={false}
                                            axisLine={false}
                                        />
                                        <YAxis stroke="#71717a" fontSize={10} tickLine={false} axisLine={false} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px' }}
                                            itemStyle={{ color: '#60a5fa' }}
                                        />
                                        <Line type="monotone" dataKey="trades_per_day" stroke="#60a5fa" dot={false} strokeWidth={2} name="Trades/Day" />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>

                    {/* Charts Row 3: APY & Avg Trade Return */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                        {/* APY Chart */}
                        <div className="card h-[300px] flex flex-col p-4">
                            <h3 className="text-sm font-medium text-muted mb-4 flex items-center gap-2">
                                <DollarSign size={16} /> APY HISTORY (EST.)
                            </h3>
                            <div className="flex-1 w-full min-h-0">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={metricsHistory.filter(m => m.apy !== null)}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                                        <XAxis
                                            dataKey="step"
                                            stroke="#71717a"
                                            fontSize={10}
                                            tickFormatter={(val) => Number.isFinite(val) ? `${(val / 1000).toFixed(0)}k` : ''}
                                            tickLine={false}
                                            axisLine={false}
                                        />
                                        <YAxis stroke="#71717a" fontSize={10} tickLine={false} axisLine={false} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px' }}
                                            itemStyle={{ color: '#fbbf24' }}
                                        />
                                        <Line type="monotone" dataKey="apy" stroke="#fbbf24" dot={false} strokeWidth={2} name="APY %" />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Avg Trade Return Chart */}
                        <div className="card h-[300px] flex flex-col p-4">
                            <h3 className="text-sm font-medium text-muted mb-4 flex items-center gap-2">
                                <TrendingUp size={16} /> AVG TRADE RETURN
                            </h3>
                            <div className="flex-1 w-full min-h-0">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={metricsHistory.filter(m => m.avg_trade_return !== null)}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                                        <XAxis
                                            dataKey="step"
                                            stroke="#71717a"
                                            fontSize={10}
                                            tickFormatter={(val) => Number.isFinite(val) ? `${(val / 1000).toFixed(0)}k` : ''}
                                            tickLine={false}
                                            axisLine={false}
                                        />
                                        <YAxis stroke="#71717a" fontSize={10} tickLine={false} axisLine={false} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px' }}
                                            itemStyle={{ color: '#a78bfa' }}
                                        />
                                        <Line type="monotone" dataKey="avg_trade_return" stroke="#a78bfa" dot={false} strokeWidth={2} name="Avg Return %" />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>

                    {/* Row 4: Trade Frequency & Asset History */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pb-6">
                        <div className="card p-4 h-[300px] flex flex-col">
                            <h3 className="text-sm font-medium text-muted mb-4 flex items-center gap-2">
                                <Activity size={16} />
                                TRADES / 7 DAYS
                            </h3>
                            <div className="flex-1 w-full min-h-0">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={metricsHistory.filter(m => m.trades_per_day !== undefined && m.trades_per_day !== null)}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                                        <XAxis
                                            dataKey="step"
                                            stroke="#71717a"
                                            fontSize={10}
                                            tickFormatter={(val) => Number.isFinite(val) ? `${(val / 1000).toFixed(0)}k` : ''}
                                            tickLine={false}
                                            axisLine={false}
                                        />
                                        <YAxis stroke="#71717a" fontSize={10} tickLine={false} axisLine={false} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px' }}
                                            itemStyle={{ color: '#10B981' }}
                                        />
                                        <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                                        <Line
                                            type="monotone"
                                            dataKey="trades_per_day"
                                            name="Trades / Day"
                                            stroke="#10B981"
                                            strokeWidth={2}
                                            dot={false}
                                            isAnimationActive={false}
                                        />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Asset History (New) */}
                        <div className="card p-4 h-[300px] flex flex-col">
                            <h3 className="text-sm font-medium text-muted mb-4 flex items-center gap-2">
                                <DollarSign size={16} />
                                ASSET HISTORY (Mean PV)
                            </h3>
                            <div className="flex-1 w-full min-h-0">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={metricsHistory.filter(m => m.mean_portfolio_value !== undefined)}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                                        <XAxis
                                            dataKey="step"
                                            stroke="#71717a"
                                            fontSize={10}
                                            tickFormatter={(val) => Number.isFinite(val) ? `${(val / 1000).toFixed(0)}k` : ''}
                                            tickLine={false}
                                            axisLine={false}
                                        />
                                        <YAxis stroke="#71717a" fontSize={10} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px' }}
                                            itemStyle={{ color: '#F59E0B' }}
                                        />
                                        <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                                        <Line
                                            type="monotone"
                                            dataKey="mean_portfolio_value"
                                            name="Mean PV"
                                            stroke="#F59E0B"
                                            strokeWidth={2}
                                            dot={false}
                                            isAnimationActive={false}
                                        />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>

                </div>

                {/* 右カラム: ログ & 情報 */}
                <div className="flex flex-col gap-6">
                    <div className="card bg-gradient-to-br from-surface to-background border-primary/20">
                        <h3 className="text-sm font-semibold mb-4 text-primary">Server Info</h3>
                        <div className="space-y-3 font-mono text-xs">
                            <div className="flex justify-between border-b border-border pb-2">
                                <span className="text-muted">Status</span>
                                <span className={isConnected ? "text-primary" : "text-red-400"}>
                                    {isConnected ? 'Online' : 'Offline'}
                                </span>
                            </div>
                            <div className="flex justify-between border-b border-border pb-2">
                                <span className="text-muted">WebSocket</span>
                                <span className="text-text truncate max-w-[120px]">{WS_URL}</span>
                            </div>
                            <div className="flex justify-between border-b border-border pb-2">
                                <span className="text-muted">Data Points</span>
                                <span className="text-text">{metricsHistory.length}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-muted">FPS</span>
                                <span className="text-text">{lastMessage?.fps?.toFixed(0) ?? 'N/A'}</span>
                            </div>
                        </div>
                    </div>

                    {/* Action Distribution Bar */}
                    <div className="card">
                        <h3 className="text-sm font-semibold mb-4 text-secondary">Action Distribution</h3>
                        <div className="space-y-2">
                            <div className="flex h-3 w-full rounded-full overflow-hidden bg-surface border border-border">
                                <div
                                    className="bg-green-500 h-full transition-all duration-500"
                                    style={{ width: `${currentStats.longPct}%` }}
                                    title={`Long: ${currentStats.longPct.toFixed(1)}%`}
                                />
                                <div
                                    className="bg-red-500 h-full transition-all duration-500"
                                    style={{ width: `${currentStats.shortPct}%` }}
                                    title={`Short: ${currentStats.shortPct.toFixed(1)}%`}
                                />
                                <div
                                    className="bg-zinc-600 h-full transition-all duration-500"
                                    style={{ width: `${currentStats.holdPct}%` }}
                                    title={`Hold: ${currentStats.holdPct.toFixed(1)}%`}
                                />
                            </div>
                            <div className="flex justify-between text-[10px] font-mono text-muted">
                                <span className="text-green-400">L: {currentStats.longPct.toFixed(0)}%</span>
                                <span className="text-red-400">S: {currentStats.shortPct.toFixed(0)}%</span>
                                <span className="text-zinc-400">H: {currentStats.holdPct.toFixed(0)}%</span>
                            </div>
                        </div>
                    </div>

                    {/* エラー表示 */}
                    {error && (
                        <div className="card bg-red-500/10 border-red-500/50 text-red-400 text-sm">
                            ⚠️ {error}
                        </div>
                    )}

                    <div className="flex-1 min-h-[300px]">
                        <LogTerminal logs={logs} maxHeight="calc(100vh - 500px)" />
                    </div>
                </div>

                {/* 設定モーダル */}
                <TrainingConfigPanel
                    isOpen={isSettingsOpen}
                    onClose={() => setIsSettingsOpen(false)}
                    onSave={handleSaveConfig}
                    initialConfig={config}
                    defaultConfig={defaultConfig}
                />
            </div>
        </div >
    );
};
