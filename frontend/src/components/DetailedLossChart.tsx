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
import { Activity } from 'lucide-react';
import type { TrainingMetrics } from '../hooks/useWebSocket';

interface DetailedLossChartProps {
    data: TrainingMetrics[];
}

export const DetailedLossChart: React.FC<DetailedLossChartProps> = ({ data }) => {
    // データがなければ表示しない
    if (!data || data.length === 0) return null;

    // 表示用にデータをフィルタリング
    const chartData = data.filter(d => d.type === 'training_step' && d.policy_loss !== null);

    return (
        <div className="card h-[600px] flex flex-col gap-4 p-4">
            {/* Chart 1: Losses */}
            <div className="flex-1 flex flex-col min-h-0">
                <div className="flex items-center gap-2 mb-2">
                    <Activity size={16} className="text-red-400" />
                    <h3 className="text-sm font-semibold tracking-wide">LOSS METRICS</h3>
                </div>
                <div className="flex-1 w-full min-h-0">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                            <XAxis
                                dataKey="step"
                                stroke="#71717a"
                                fontSize={10}
                                tickFormatter={(val) => `${val / 1000}k`}
                                tickLine={false}
                                axisLine={false}
                                hide // Hide X axis for top chart to reduce clutter
                            />
                            <YAxis
                                stroke="#71717a"
                                fontSize={10}
                                tickLine={false}
                                axisLine={false}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: '#18181b',
                                    borderColor: '#27272a',
                                    borderRadius: '8px',
                                    fontSize: '12px'
                                }}
                                itemStyle={{ color: '#f4f4f5' }}
                            />
                            <Legend verticalAlign="top" height={36} />
                            <Line
                                type="monotone"
                                dataKey="policy_loss"
                                name="Policy Loss"
                                stroke="#ef4444"
                                dot={false}
                                strokeWidth={2}
                            />
                            <Line
                                type="monotone"
                                dataKey="value_loss"
                                name="Value Loss"
                                stroke="#3b82f6"
                                dot={false}
                                strokeWidth={2}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Separator */}
            <div className="h-px bg-border/50" />

            {/* Chart 2: Stability (Entropy & KL) */}
            <div className="flex-1 flex flex-col min-h-0">
                <div className="flex items-center gap-2 mb-2">
                    <Activity size={16} className="text-yellow-400" />
                    <h3 className="text-sm font-semibold tracking-wide">STABILITY (ENTROPY / KL)</h3>
                </div>
                <div className="flex-1 w-full min-h-0">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                            <XAxis
                                dataKey="step"
                                stroke="#71717a"
                                fontSize={10}
                                tickFormatter={(val) => `${val / 1000}k`}
                                tickLine={false}
                                axisLine={false}
                            />
                            <YAxis
                                yAxisId="left"
                                stroke="#71717a"
                                fontSize={10}
                                tickLine={false}
                                axisLine={false}
                            />
                            <YAxis
                                yAxisId="right"
                                orientation="right"
                                stroke="#71717a"
                                fontSize={10}
                                tickLine={false}
                                axisLine={false}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: '#18181b',
                                    borderColor: '#27272a',
                                    borderRadius: '8px',
                                    fontSize: '12px'
                                }}
                                itemStyle={{ color: '#f4f4f5' }}
                            />
                            <Legend verticalAlign="top" height={36} />
                            <Line
                                yAxisId="left"
                                type="monotone"
                                dataKey="entropy_loss"
                                name="Entropy"
                                stroke="#eab308"
                                dot={false}
                                strokeWidth={2}
                            />
                            <Line
                                yAxisId="right"
                                type="monotone"
                                dataKey="approx_kl"
                                name="Approx KL"
                                stroke="#a855f7"
                                dot={false}
                                strokeWidth={1}
                                strokeDasharray="3 3"
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};
