import React, { useState, useEffect } from 'react';
import { X, Save, RotateCcw } from 'lucide-react';
import type { TrainingConfig } from '../hooks/useTrainingControl';
import { ModelSelect } from './ModelSelect';

interface TrainingConfigPanelProps {
    isOpen: boolean;
    onClose: () => void;
    onSave: (config: TrainingConfig) => void;
    initialConfig: TrainingConfig | null;
    defaultConfig: TrainingConfig | null;
}

export const TrainingConfigPanel: React.FC<TrainingConfigPanelProps> = ({
    isOpen,
    onClose,
    onSave,
    initialConfig,
    defaultConfig,
}) => {
    const [config, setConfig] = useState<TrainingConfig | null>(null);
    const [availableData, setAvailableData] = useState<Record<string, string[]>>({});
    const [noLimitSteps, setNoLimitSteps] = useState(false);

    useEffect(() => {
        if (isOpen && initialConfig) {
            setConfig({ ...initialConfig });
            // Max Stepsが大きい値ならNo Limit扱いにする
            if (initialConfig.max_steps > 1000000) {
                setNoLimitSteps(true);
            } else {
                setNoLimitSteps(false);
            }
        }
    }, [isOpen, initialConfig]);

    // 利用可能なデータを取得
    const fetchData = React.useCallback(async () => {
        try {
            const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';
            const res = await fetch(`${API_BASE}/api/data/available`);
            if (res.ok) {
                const data = await res.json();
                setAvailableData(data.available || {});
            }
        } catch (e) {
            console.error("Failed to fetch available data", e);
        }
    }, []);

    // 初回マウント時と、パネルが開かれた時に最新データを取得
    useEffect(() => {
        if (isOpen) {
            fetchData();
        }
    }, [isOpen, fetchData]);

    if (!isOpen || !config) return null;

    const handleChange = (key: keyof TrainingConfig, value: string | number | boolean | number[] | null) => {
        setConfig((prev) => {
            if (!prev) return null;
            return {
                ...prev,
                [key]: typeof value === 'string' && !['symbol', 'interval', 'side', 'load_model_path', 'device'].includes(key) ? Number(value) : value,
            };
        });
    };

    // Max StepsのNo Limit切り替え
    const handleNoLimitChange = (checked: boolean) => {
        setNoLimitSteps(checked);
        setConfig((prev) => {
            if (!prev) return null;
            return {
                ...prev,
                max_steps: checked ? 1000000000 : 100, // No Limitなら巨大な値を入れる
            };
        });
    };

    const handleReset = () => {
        if (defaultConfig) {
            setConfig({ ...defaultConfig });
            setNoLimitSteps(false);
        }
    };

    const handleSave = () => {
        if (config) {
            onSave(config);
            onClose();
        }
    };

    const symbols = Object.keys(availableData).sort();
    // 現在選択されているSymbolがavailableDataにない場合も表示されるようにする
    const displaySymbols = config.symbol && !symbols.includes(config.symbol)
        ? [config.symbol, ...symbols]
        : symbols;

    const intervals = config.symbol && availableData[config.symbol] ? availableData[config.symbol] : [];
    const displayIntervals = config.interval && !intervals.includes(config.interval)
        ? [config.interval, ...intervals]
        : intervals;

    return (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
            <div className="bg-surface border border-border rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] flex flex-col">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-border">
                    <h2 className="text-lg font-bold text-text">Training Configuration</h2>
                    <button onClick={onClose} className="text-muted hover:text-text">
                        <X size={20} />
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6">

                    {/* Model Management (New) */}
                    <section className="bg-surface/50 p-4 rounded border border-primary/20">
                        <h3 className="text-sm font-semibold text-primary mb-3 uppercase tracking-wider flex items-center gap-2">
                            Resume Training
                        </h3>
                        <div className="space-y-2">
                            <label className="text-xs font-medium text-muted block">Load Model (Checkpoint)</label>
                            <ModelSelect
                                value={config.load_model_path}
                                onChange={(val) => handleChange('load_model_path', val)}
                            />
                            <p className="text-[10px] text-muted/70">
                                Select a saved model or checkpoint to resume training from. Default starts from scratch.
                            </p>
                        </div>
                    </section>

                    {/* Environment Settings */}
                    <section>
                        <h3 className="text-sm font-semibold text-primary mb-3 uppercase tracking-wider">Environment</h3>
                        <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-muted block">Symbol</label>
                                <select
                                    value={config.symbol}
                                    onChange={(e) => handleChange('symbol', e.target.value)}
                                    className="w-full bg-background border border-border rounded px-3 py-2 text-sm text-text focus:outline-none focus:border-primary"
                                >
                                    {displaySymbols.length === 0 && <option value={config.symbol}>{config.symbol}</option>}
                                    {displaySymbols.map(s => (
                                        <option key={s} value={s}>{s}</option>
                                    ))}
                                </select>
                            </div>

                            <div className="space-y-1">
                                <label className="text-xs font-medium text-muted block">Interval</label>
                                <select
                                    value={config.interval}
                                    onChange={(e) => handleChange('interval', e.target.value)}
                                    className="w-full bg-background border border-border rounded px-3 py-2 text-sm text-text focus:outline-none focus:border-primary"
                                >
                                    {displayIntervals.length === 0 && <option value={config.interval}>{config.interval}</option>}
                                    {displayIntervals.map(i => (
                                        <option key={i} value={i}>{i}</option>
                                    ))}
                                </select>
                            </div>

                            <InputField
                                label="Initial Inventory"
                                type="number"
                                value={config.initial_inventory}
                                onChange={(v) => handleChange('initial_inventory', v)}
                            />

                            <div className="space-y-1">
                                <label className="text-xs font-medium text-muted block">Max Steps</label>
                                <div className="flex items-center gap-2">
                                    <input
                                        type="number"
                                        value={noLimitSteps ? '' : config.max_steps}
                                        onChange={(e) => handleChange('max_steps', e.target.value)}
                                        disabled={noLimitSteps}
                                        className="w-full bg-background border border-border rounded px-3 py-2 text-sm text-text focus:outline-none focus:border-primary disabled:opacity-30"
                                        placeholder={noLimitSteps ? "Unlimited" : "100"}
                                    />
                                    <label className="flex items-center gap-1.5 cursor-pointer whitespace-nowrap">
                                        <input
                                            type="checkbox"
                                            checked={noLimitSteps}
                                            onChange={(e) => handleNoLimitChange(e.target.checked)}
                                            className="w-4 h-4 rounded border-border bg-background text-primary focus:ring-primary"
                                        />
                                        <span className="text-xs text-muted">No Limit</span>
                                    </label>
                                </div>
                            </div>
                        </div>
                    </section>

                    {/* Market Physics (New) */}
                    <section className="bg-surface/50 p-4 rounded border border-primary/20">
                        <h3 className="text-sm font-semibold text-primary mb-3 uppercase tracking-wider flex items-center gap-2">
                            Strategy & Physics
                        </h3>
                        <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-1">
                                <div className="flex justify-between items-center mb-1">
                                    <label className="text-xs font-medium text-muted block">
                                        {config.use_usdt ? 'Invest Amount (USDT)' : 'Quantity (Base)'}
                                    </label>
                                    <label className="flex items-center gap-1 cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={config.use_usdt}
                                            onChange={(e) => {
                                                const useUsdt = e.target.checked;
                                                handleChange('use_usdt', useUsdt);
                                            }}
                                            className="w-3 h-3 rounded border-border"
                                        />
                                        <span className="text-[10px] text-primary">Use USDT (Buy)</span>
                                    </label>
                                </div>
                                <input
                                    type="number"
                                    value={config.initial_inventory}
                                    onChange={(e) => handleChange('initial_inventory', e.target.value)}
                                    className="w-full bg-background border border-border rounded px-3 py-2 text-sm text-text focus:outline-none focus:border-primary"
                                />
                                <p className="text-[10px] text-muted/70">
                                    {config.use_usdt
                                        ? "Buying crypto with this USDT amount."
                                        : "Selling this amount of crypto."}
                                </p>
                            </div>

                            <InputField
                                label="Trading Fee Rate"
                                type="number"
                                step="0.0001"
                                value={config.trade_fee}
                                onChange={(v) => handleChange('trade_fee', v)}
                                description="Cost per trade (0.001 = 0.1%)"
                            />
                            <InputField
                                label="Market Impact (Gamma)"
                                type="number"
                                step="0.1"
                                value={config.y_impact}
                                onChange={(v) => handleChange('y_impact', v)}
                            />

                            {/* Reward Settings */}
                            <div className="col-span-2 border-t border-border/50 pt-2 mt-2">
                                <h4 className="text-xs font-semibold text-primary mb-2">Reward Configuration</h4>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="space-y-1">
                                        <div className="flex items-center justify-between">
                                            <label className="text-xs font-medium text-muted">Use DSR (Sharpe)</label>
                                            <input
                                                type="checkbox"
                                                checked={config.use_dsr}
                                                onChange={(e) => handleChange('use_dsr', e.target.checked)}
                                                className="w-4 h-4 rounded border-border bg-background text-primary"
                                            />
                                        </div>
                                        <p className="text-[10px] text-muted/70">
                                            {config.use_dsr ? "Optimizing Risk-Adjusted Return (Harder)" : "Optimizing Simple Profit (Start Here)"}
                                        </p>
                                    </div>
                                    <InputField
                                        label="Reward Scale"
                                        type="number"
                                        step="0.01"
                                        value={config.reward_scale}
                                        onChange={(v) => handleChange('reward_scale', v)}
                                        description="Scale reward for stability (0.1 ~ 0.01)"
                                    />
                                    <InputField
                                        label="DSR Warmup Steps"
                                        type="number"
                                        value={config.dsr_warmup_steps}
                                        onChange={(v) => handleChange('dsr_warmup_steps', v)}
                                        description="Steps to use PnL before switching to DSR (0 to disable)"
                                    />
                                </div>
                            </div>
                        </div>
                    </section>

                    {/* Evolution Strategy */}
                    <section className="bg-surface/50 p-4 rounded border border-primary/20">
                        <div className="flex justify-between items-center mb-3">
                            <h3 className="text-sm font-semibold text-primary uppercase tracking-wider flex items-center gap-2">
                                Evolution Strategy (PBT-MAP-Elites)
                            </h3>
                            <label className="flex items-center gap-2 cursor-pointer">
                                <span className="text-xs font-medium text-muted">Enable</span>
                                <div className="relative inline-flex items-center cursor-pointer">
                                    <input
                                        type="checkbox"
                                        checked={config.use_evolution}
                                        onChange={(e) => handleChange('use_evolution', e.target.checked)}
                                        className="sr-only peer"
                                    />
                                    <div className="w-9 h-5 bg-border peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-primary"></div>
                                </div>
                            </label>
                        </div>

                        {config.use_evolution && (
                            <div className="grid grid-cols-2 gap-4 animate-in fade-in slide-in-from-top-2 duration-200">
                                <InputField
                                    label="Generations"
                                    type="number"
                                    value={config.n_generations}
                                    onChange={(v) => handleChange('n_generations', v)}
                                    description="Number of PBT generations"
                                />
                                <InputField
                                    label="Population Size"
                                    type="number"
                                    value={config.population_size}
                                    onChange={(v) => handleChange('population_size', v)}
                                    description="Number of parallel agents"
                                />
                                <InputField
                                    label="Steps per Generation"
                                    type="number"
                                    value={config.steps_per_generation}
                                    onChange={(v) => handleChange('steps_per_generation', v)}
                                    description="Training steps per agent/gen"
                                />
                                <InputField
                                    label="Grid Bins"
                                    type="number"
                                    value={config.grid_bins}
                                    onChange={(v) => handleChange('grid_bins', v)}
                                    description="Archive grid size (e.g. 5 means 5x5)"
                                />
                            </div>
                        )}
                    </section>

                    {/* PPO Hyperparameters */}
                    <section>
                        <h3 className="text-sm font-semibold text-muted mb-3 uppercase tracking-wider">Model Architecture</h3>

                        {/* Layer Config Helper */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-4">
                            <LayerConfigInput
                                label="Policy Network (Actor)"
                                layers={config.policy_layers}
                                onChange={(layers) => handleChange('policy_layers', layers)}
                            />
                            <LayerConfigInput
                                label="Value Network (Critic)"
                                layers={config.value_layers}
                                onChange={(layers) => handleChange('value_layers', layers)}
                            />
                        </div>

                        <h3 className="text-sm font-semibold text-muted mb-3 uppercase tracking-wider">PPO Parameters</h3>
                        <div className="grid grid-cols-2 gap-4">
                            <InputField
                                label="Total Timesteps"
                                type="number"
                                value={config.total_timesteps}
                                onChange={(v) => handleChange('total_timesteps', v)}
                            />
                            <InputField
                                label="Learning Rate"
                                type="number"
                                step="0.00001"
                                value={config.learning_rate}
                                onChange={(v) => handleChange('learning_rate', v)}
                            />
                            <InputField
                                label="Min Learning Rate"
                                type="number"
                                step="0.000001"
                                value={config.min_learning_rate}
                                onChange={(v) => handleChange('min_learning_rate', v)}
                                description="Rule B: Clamp lower bound"
                            />
                            <InputField
                                label="N Steps"
                                type="number"
                                value={config.n_steps}
                                onChange={(v) => handleChange('n_steps', v)}
                            />
                            <InputField
                                label="Batch Size"
                                type="number"
                                value={config.batch_size}
                                onChange={(v) => handleChange('batch_size', v)}
                            />
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-muted block">Device</label>
                                <select
                                    value={config.device || "auto"}
                                    onChange={(e) => handleChange('device', e.target.value)}
                                    className="w-full bg-background border border-border rounded px-3 py-2 text-sm text-text focus:outline-none focus:border-primary"
                                >
                                    <option value="auto">Auto</option>
                                    <option value="cpu">CPU</option>
                                    <option value="cuda">GPU (CUDA)</option>
                                </select>
                            </div>
                        </div>
                    </section>
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-border flex justify-between bg-surface/50">
                    <button
                        onClick={handleReset}
                        className="flex items-center gap-2 px-4 py-2 text-sm text-muted hover:text-text transition-colors"
                    >
                        <RotateCcw size={16} />
                        Reset to Defaults
                    </button>
                    <div className="flex gap-3">
                        <button
                            onClick={onClose}
                            className="px-4 py-2 rounded text-sm font-medium hover:bg-white/5 transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleSave}
                            className="flex items-center gap-2 px-6 py-2 rounded bg-primary hover:bg-primary/90 text-white text-sm font-bold transition-colors shadow-lg shadow-primary/20"
                        >
                            <Save size={16} />
                            Save Configuration
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

interface InputFieldProps {
    label: string;
    type?: 'text' | 'number';
    value: string | number | undefined;
    onChange: (value: string | number) => void;
    step?: string;
    disabled?: boolean;
    description?: string;
}

const InputField: React.FC<InputFieldProps> = ({
    label,
    type = 'text',
    value,
    onChange,
    step,
    disabled,
    description
}) => (
    <div className="space-y-1">
        <label className="text-xs font-medium text-muted block">{label}</label>
        <input
            type={type}
            value={value}
            onChange={(e) => onChange(type === 'number' ? parseFloat(e.target.value) : e.target.value)}
            disabled={disabled}
            step={step}
            className="w-full bg-background border border-border rounded px-3 py-2 text-sm text-text focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary disabled:opacity-50"
        />
        {description && <p className="text-[10px] text-muted/70">{description}</p>}
    </div>
);

interface LayerConfigInputProps {
    label: string;
    layers: number[];
    onChange: (layers: number[]) => void;
}

const LayerConfigInput: React.FC<LayerConfigInputProps> = ({ label, layers, onChange }) => {
    // Determine unit size and depth from existing layers
    // Assuming uniform layers for now as per user request (e.g. [128, 128] -> 128, 2)
    const currentUnit = layers.length > 0 ? layers[0] : 128;
    const currentDepth = layers.length;

    const handleUnitChange = (val: number) => {
        const newLayers = Array(currentDepth).fill(val);
        onChange(newLayers);
    };

    const handleDepthChange = (val: number) => {
        // Limit depth to reasonable numbers (e.g. 1-10)
        const depth = Math.max(1, Math.min(10, val));
        const newLayers = Array(depth).fill(currentUnit);
        onChange(newLayers);
    };

    return (
        <div className="space-y-1 p-3 bg-background/50 rounded border border-border">
            <label className="text-xs font-medium text-primary block mb-2">{label}</label>
            <div className="flex gap-4">
                <div className="flex-1 space-y-1">
                    <label className="text-[10px] text-muted">Unit Size</label>
                    <input
                        type="number"
                        value={currentUnit}
                        onChange={(e) => handleUnitChange(parseInt(e.target.value) || 128)}
                        className="w-full bg-background border border-border rounded px-2 py-1 text-xs"
                    />
                </div>
                <div className="flex-1 space-y-1">
                    <label className="text-[10px] text-muted">Layer Count</label>
                    <input
                        type="number"
                        min="1"
                        max="10"
                        value={currentDepth}
                        onChange={(e) => handleDepthChange(parseInt(e.target.value) || 1)}
                        className="w-full bg-background border border-border rounded px-2 py-1 text-xs"
                    />
                </div>
            </div>
            <div className="text-[10px] text-muted/60 mt-1">
                Result: [{layers.join(', ')}]
            </div>
        </div>
    );
};
