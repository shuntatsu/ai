import { useState, useCallback, useEffect } from 'react';

// バックエンドAPIのベースURL
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

export interface TrainingConfig {
    total_timesteps: number;
    learning_rate: number;
    min_learning_rate: number;
    n_steps: number;
    batch_size: number;
    n_epochs: number;
    gamma: number;
    gae_lambda: number;
    clip_range: number;
    ent_coef: number;
    vf_coef: number;
    symbol: string;
    interval: string;
    initial_inventory: number;
    max_steps: number;
    output_dir: string;
    checkpoint_freq: number;
    // New parameters
    trade_fee: number;
    y_impact: number;
    lambda_risk: number;
    use_dsr: boolean;
    reward_scale: number;
    dsr_warmup_steps: number;
    // Advanced/New Features

    use_usdt: boolean;
    policy_layers: number[];
    value_layers: number[];
    // Evolution Strategy
    use_evolution: boolean;
    population_size: number;
    n_generations: number;
    steps_per_generation: number;
    grid_bins: number;
    eval_episodes: number;
    // Model Management
    load_model_path: string | null;
    device?: string;
}

export interface TrainingStatus {
    status: 'idle' | 'starting' | 'running' | 'stopping' | 'completed' | 'error';
    is_running: boolean;
    config?: TrainingConfig;
    elapsed_time?: number;
    current_step?: number;
    progress?: number;
    error?: string;
}

interface UseTrainingControlReturn {
    // 状態
    trainingStatus: TrainingStatus;
    isLoading: boolean;
    error: string | null;
    progressMessage: string | null;

    // アクション
    startTraining: (config?: Partial<TrainingConfig>) => Promise<boolean>;
    stopTraining: () => Promise<boolean>;
    refreshStatus: () => Promise<void>;
    getDefaultConfig: () => Promise<TrainingConfig | null>;
}

/**
 * 学習制御用カスタムフック
 * 
 * 学習の開始・停止・状態取得を管理
 */
export function useTrainingControl(): UseTrainingControlReturn {
    const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
        status: 'idle',
        is_running: false,
    });
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [progressMessage, setProgressMessage] = useState<string | null>(null);

    /**
     * 状態を更新
     */
    const refreshStatus = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/training/status`);
            if (response.ok) {
                const data = await response.json();
                setTrainingStatus(data);
            }
        } catch (e) {
            console.error('[TrainingControl] Failed to fetch status:', e);
        }
    }, []);

    // WebSocket 接続（ログ受信）
    useEffect(() => {
        const wsUrl = API_BASE.replace(/^http/, 'ws') + '/ws/metrics';
        let ws: WebSocket | null = null;

        try {
            ws = new WebSocket(wsUrl);

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'log') {
                        setProgressMessage(data.message);
                    } else if (data.type === 'training_start') {
                        setProgressMessage("Training started...");
                        setTimeout(() => setProgressMessage(null), 3000);
                    } else if (data.type === 'error') {
                        // エラー時はメッセージを残す
                    }
                } catch (e) {
                    // ignore
                }
            };
        } catch (e) {
            console.error("WebSocket connection failed", e);
        }

        return () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
        };
    }, []);

    /**
     * 学習を開始
     */
    const startTraining = useCallback(async (config?: Partial<TrainingConfig>): Promise<boolean> => {
        setIsLoading(true);
        setError(null);
        setProgressMessage("Initializing..."); // 初期メッセージ

        try {
            const response = await fetch(`${API_BASE}/api/training/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: config ? JSON.stringify(config) : undefined,
            });

            const data = await response.json();

            if (data.success) {
                setTrainingStatus({
                    status: 'starting',
                    is_running: true,
                    config: data.config,
                });
                return true;
            } else {
                setError(data.error || 'Failed to start training');
                setProgressMessage(null);
                return false;
            }
        } catch (e) {
            const message = e instanceof Error ? e.message : 'Unknown error';
            setError(`Failed to start training: ${message}`);
            setProgressMessage(null);
            return false;
        } finally {
            setIsLoading(false);
        }
    }, []);

    /**
     * 学習を停止
     */
    const stopTraining = useCallback(async (): Promise<boolean> => {
        setIsLoading(true);
        setError(null);

        try {
            const response = await fetch(`${API_BASE}/api/training/stop`, {
                method: 'POST',
            });

            const data = await response.json();

            if (data.success) {
                setTrainingStatus((prev) => ({
                    ...prev,
                    status: 'stopping',
                }));
                setProgressMessage("Stopping...");
                return true;
            } else {
                setError(data.error || 'Failed to stop training');
                return false;
            }
        } catch (e) {
            const message = e instanceof Error ? e.message : 'Unknown error';
            setError(`Failed to stop training: ${message}`);
            return false;
        } finally {
            setIsLoading(false);
        }
    }, []);

    /**
     * デフォルト設定を取得
     */
    const getDefaultConfig = useCallback(async (): Promise<TrainingConfig | null> => {
        try {
            const response = await fetch(`${API_BASE}/api/training/config`);
            if (response.ok) {
                return await response.json();
            }
            return null;
        } catch (e) {
            console.error('[TrainingControl] Failed to fetch config:', e);
            return null;
        }
    }, []);

    // マウント時に状態を取得
    useEffect(() => {
        refreshStatus();
    }, [refreshStatus]);

    // 学習中は定期的に状態を更新
    useEffect(() => {
        if (!trainingStatus.is_running) return;

        const interval = setInterval(() => {
            refreshStatus();
        }, 2000);

        return () => clearInterval(interval);
    }, [trainingStatus.is_running, refreshStatus]);

    return {
        trainingStatus,
        isLoading,
        error,
        progressMessage,
        startTraining,
        stopTraining,
        refreshStatus,
        getDefaultConfig,
    };
}
export default useTrainingControl;
