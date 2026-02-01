import React from 'react';
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer
} from 'recharts';
import { Activity } from 'lucide-react';

export interface TrainingDataPoint {
    step: number;
    meanReward: number;
    valueLoss: number;
    entropy: number;
}

interface TrainingChartProps {
    data: TrainingDataPoint[];
}

export const TrainingChart: React.FC<TrainingChartProps> = ({ data }) => {
    return (
        <div className="card h-full min-h-[300px] flex flex-col">
            <div className="flex items-center gap-2 mb-4">
                <Activity size={16} className="text-primary" />
                <h3 className="text-sm font-semibold tracking-wide">MEAN REWARD (Last 100 eps)</h3>
            </div>

            <div className="flex-1 w-full h-full min-h-[250px]">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data}>
                        <defs>
                            <linearGradient id="colorReward" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                            </linearGradient>
                        </defs>
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
                            labelStyle={{ color: '#71717a' }}
                        />
                        <Area
                            type="monotone"
                            dataKey="meanReward"
                            stroke="#10b981"
                            strokeWidth={2}
                            fillOpacity={1}
                            fill="url(#colorReward)"
                            animationDuration={500}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
