import { useState, useEffect, useCallback, useRef } from 'react';

export interface TrainingMetrics {
    type: string;
    timestamp: number;
    step: number;
    progress: number;
    n_updates: number;
    policy_loss: number | null;
    value_loss: number | null;
    entropy_loss: number | null;
    approx_kl: number | null;
    clip_fraction: number | null;
    explained_variance: number | null;
    learning_rate: number | null;
    mean_reward: number | null;
    std_reward: number | null;
    n_episodes: number | null;
    elapsed_time: number | null;
    fps: number | null;
    total_timesteps: number | null;
    total_steps: number | null;
    final_mean_reward: number | null;
    mean_win_rate: number | null;
    mean_max_drawdown: number | null;
    mean_portfolio_value?: number;
    long_pct?: number;
    short_pct?: number;
    hold_pct?: number;
    // New Metrics
    apy?: number;
    max_trade_profit?: number;
    max_trade_loss?: number;
    trades_per_day?: number;
    avg_trade_return?: number;
}

interface UseWebSocketOptions {
    url: string;
    reconnectInterval?: number;
    maxRetries?: number;
}

interface UseWebSocketReturn {
    isConnected: boolean;
    lastMessage: TrainingMetrics | null;
    metricsHistory: TrainingMetrics[];
    error: string | null;
    send: (message: string) => void;
    connect: () => void;
    disconnect: () => void;
}

/**
 * WebSocket接続を管理するカスタムフック
 * 
 * @param options - 接続オプション
 * @returns WebSocket状態とメソッド
 */
export function useWebSocket(options: UseWebSocketOptions): UseWebSocketReturn {
    const { url, reconnectInterval = 3000, maxRetries = 5 } = options;

    const [isConnected, setIsConnected] = useState(false);
    const [lastMessage, setLastMessage] = useState<TrainingMetrics | null>(null);
    const [metricsHistory, setMetricsHistory] = useState<TrainingMetrics[]>([]);
    const [error, setError] = useState<string | null>(null);

    const wsRef = useRef<WebSocket | null>(null);
    const retriesRef = useRef(0);
    const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const bufferRef = useRef<TrainingMetrics[]>([]);

    // バッファーをフラッシュしてStateを更新
    const flushBuffer = useCallback(() => {
        if (bufferRef.current.length === 0) return;

        // バッファーのスナップショットを取得してクリア
        const updates = bufferRef.current;
        bufferRef.current = [];

        setMetricsHistory(prev => {
            // 最適化: バッファーだけで表示数を超えている場合は、バッファーの最新分だけを使う
            // これにより、[...prev, ...huge_buffer] のような巨大配列生成（OOMの原因）を防ぐ
            if (updates.length >= 300) {
                return updates.slice(-300);
            }

            // 通常時: 結合して末尾300件
            const newHistory = [...prev, ...updates];
            if (newHistory.length > 300) {
                return newHistory.slice(-300);
            }
            return newHistory;
        });
    }, []);

    // 定期的にバッファーをフラッシュ (200msごと = 5FPS)
    // これにより、バックエンドが高速でもReactのレンダリング回数は抑えられる
    useEffect(() => {
        const interval = setInterval(flushBuffer, 200);
        return () => clearInterval(interval);
    }, [flushBuffer]);

    // 接続処理
    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            return;
        }

        try {
            const ws = new WebSocket(url);

            ws.onopen = () => {
                setIsConnected(true);
                setError(null);
                retriesRef.current = 0;
                console.log('[WebSocket] Connected to', url);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    // 履歴データ（一括）の場合 - 即時更新
                    if (data.type === 'history' && Array.isArray(data.data)) {
                        setMetricsHistory(data.data.slice(-300));
                        return;
                    }

                    // バッファー保護: バックグラウンドタブなどでflushされない間に
                    // 配列が無限肥大化するのを防ぐ (Max 2000)
                    if (bufferRef.current.length > 2000) {
                        bufferRef.current.shift(); // 古いものを捨てる
                    }

                    // 学習ステップメトリクスの場合 - バッファーに追加
                    if (data.type === 'training_step') {
                        setLastMessage(data as TrainingMetrics);
                        bufferRef.current.push(data as TrainingMetrics);
                    }

                    // 学習開始/終了通知/システムログ - バッファーに追加
                    if (data.type === 'training_start' || data.type === 'training_end' || data.type === 'log') {
                        bufferRef.current.push(data as TrainingMetrics);
                    }
                } catch (e) {
                    console.error('[WebSocket] Failed to parse message:', e);
                }
            };

            ws.onerror = (event) => {
                console.error('[WebSocket] Error:', event);
                setError('WebSocket connection error');
            };

            ws.onclose = () => {
                setIsConnected(false);
                console.log('[WebSocket] Disconnected');

                // 自動再接続
                if (retriesRef.current < maxRetries) {
                    retriesRef.current += 1;
                    console.log(`[WebSocket] Reconnecting in ${reconnectInterval}ms... (attempt ${retriesRef.current}/${maxRetries})`);

                    reconnectTimeoutRef.current = setTimeout(() => {
                        connect();
                    }, reconnectInterval);
                } else {
                    setError('Max reconnection attempts reached');
                }
            };

            wsRef.current = ws;
        } catch (e) {
            setError(`Failed to create WebSocket: ${e}`);
        }
    }, [url, reconnectInterval, maxRetries]);

    // 切断処理
    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
        }

        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }

        setIsConnected(false);
    }, []);

    // メッセージ送信
    const send = useCallback((message: string) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(message);
        }
    }, []);

    // マウント時に接続、アンマウント時に切断
    useEffect(() => {
        connect();

        return () => {
            disconnect();
        };
    }, [connect, disconnect]);

    // Ping/Pong でキープアライブ
    useEffect(() => {
        if (!isConnected) return;

        const interval = setInterval(() => {
            send('ping');
        }, 30000);

        return () => clearInterval(interval);
    }, [isConnected, send]);

    return {
        isConnected,
        lastMessage,
        metricsHistory,
        error,
        send,
        connect,
        disconnect,
    };
}

export default useWebSocket;
