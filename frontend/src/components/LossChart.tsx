import React from 'react';
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
import { TrendingDown } from 'lucide-react';
import type { TrainingMetrics } from '../hooks/useWebSocket';

interface LossChartProps {
    data: TrainingMetrics[];
    showPolicyLoss?: boolean;
    showValueLoss?: boolean;
    showEntropy?: boolean;
    showKL?: boolean;
}

/**
 * 学習損失グラフコンポーネント
 * 
 * Policy Loss, Value Loss, Entropy, KL Divergenceを表示
 */
export const LossChart: React.FC<LossChartProps> = ({
    data,
    showPolicyLoss = true,
    showValueLoss = true,
    showEntropy = true,
    showKL = false,
}) => {
    // training_stepタイプのみフィルタ
    const chartData = data
        .filter((d) => d.type === 'training_step')
        .map((d) => ({
            step: d.step,
            policyLoss: d.policy_loss != null ? Math.abs(d.policy_loss) : null,
            valueLoss: d.value_loss != null ? d.value_loss : null,
            entropy: d.entropy_loss != null ? Math.abs(d.entropy_loss) : null,
            kl: d.approx_kl != null ? d.approx_kl : null,
        }));

    // データがない場合
    if (chartData.length === 0) {
        return (
            <div className="card h-full min-h-[250px] flex flex-col">
                <div className="flex items-center gap-2 mb-4">
                    <TrendingDown size={16} className="text-secondary" />
                    <h3 className="text-sm font-semibold tracking-wide">TRAINING LOSSES</h3>
                </div>
                <div className="flex-1 flex items-center justify-center text-muted">
                    Waiting for training data...
                </div>
            </div>
        );
    }

    return (
        <div className="card h-full min-h-[250px] flex flex-col">
            <div className="flex items-center gap-2 mb-4">
                <TrendingDown size={16} className="text-secondary" />
                <h3 className="text-sm font-semibold tracking-wide">TRAINING LOSSES</h3>
                <span className="ml-auto text-xs text-muted font-mono">
                    {chartData.length} updates
                </span>
            </div>

            <div className="flex-1 w-full h-full min-h-[200px]">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                        <XAxis
                            dataKey="step"
                            stroke="#71717a"
                            fontSize={10}
                            tickFormatter={(val) => `${(val / 1000).toFixed(0)}k`}
                            tickLine={false}
                            axisLine={false}
                        />
                        <YAxis
                            stroke="#71717a"
                            fontSize={10}
                            tickLine={false}
                            axisLine={false}
                            tickFormatter={(val) => val.toFixed(3)}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#18181b',
                                borderColor: '#27272a',
                                borderRadius: '8px',
                                fontSize: '12px'
                            }}
                            itemStyle={{ color: '#f4f4f5' }}
                            labelStyle={{ color: '#71717a' }}
                            formatter={(value: number | undefined) => value?.toFixed(4) ?? 'N/A'}
                            labelFormatter={(label) => `Step: ${label.toLocaleString()}`}
                        />
                        <Legend
                            wrapperStyle={{ fontSize: '11px' }}
                            iconType="line"
                        />

                        {showPolicyLoss && (
                            <Line
                                type="monotone"
                                dataKey="policyLoss"
                                name="Policy Loss"
                                stroke="#3b82f6"
                                strokeWidth={1.5}
                                dot={false}
                                animationDuration={300}
                            />
                        )}

                        {showValueLoss && (
                            <Line
                                type="monotone"
                                dataKey="valueLoss"
                                name="Value Loss"
                                stroke="#ef4444"
                                strokeWidth={1.5}
                                dot={false}
                                animationDuration={300}
                            />
                        )}

                        {showEntropy && (
                            <Line
                                type="monotone"
                                dataKey="entropy"
                                name="Entropy"
                                stroke="#10b981"
                                strokeWidth={1.5}
                                dot={false}
                                animationDuration={300}
                            />
                        )}

                        {showKL && (
                            <Line
                                type="monotone"
                                dataKey="kl"
                                name="KL Div"
                                stroke="#f59e0b"
                                strokeWidth={1.5}
                                dot={false}
                                animationDuration={300}
                            />
                        )}
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
