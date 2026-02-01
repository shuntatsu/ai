import React, { useEffect, useState } from 'react';
import {
    ComposedChart,
    Line,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceDot
} from 'recharts';
import { Activity, Play } from 'lucide-react';

interface TradingData {
    type: 'trading_data';
    step: number;
    p_base: number;
    p_exec: number;
    action: number;
    side?: 'buy' | 'sell' | 'hold';
    inventory_after: number;
    reward: number;
    timestamp: number;
    event?: 'normal' | 'liquidation';
}

interface TradingVisualizerProps {
    lastMessage: any | null;
}

export const TradingVisualizer: React.FC<TradingVisualizerProps> = ({ lastMessage }) => {
    const [data, setData] = useState<TradingData[]>([]);
    const maxPoints = 100;

    useEffect(() => {
        if (lastMessage && lastMessage.type === 'trading_data') {
            setData(prev => {
                const newData = [...prev, lastMessage as TradingData];
                if (newData.length > maxPoints) {
                    return newData.slice(-maxPoints);
                }
                return newData;
            });
        }
    }, [lastMessage]);

    if (data.length === 0) {
        return (
            <div className="card h-[400px] flex items-center justify-center text-muted">
                <div className="text-center">
                    <Activity size={48} className="mx-auto mb-2 opacity-50 animate-pulse text-primary" />
                    <p className="text-lg font-medium">Waiting for Execution Data...</p>
                    <p className="text-xs text-muted mt-1">Training is initializing or no trades executed yet.</p>
                </div>
            </div>
        );
    }

    const last = data[data.length - 1];

    return (
        <div className="card h-[400px] flex flex-col">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <Play size={16} className="text-primary" />
                    <h3 className="text-sm font-semibold tracking-wide">LIVE TRADING VIEW</h3>
                </div>
                <div className="flex gap-4 text-xs font-mono">
                    <div>
                        <span className="text-muted">Price: </span>
                        <span className="text-text">{last.p_base.toFixed(2)}</span>
                    </div>
                    <div>
                        <span className="text-muted">Inv: </span>
                        <span className={`font-bold ${last.inventory_after > 0 ? 'text-green-400' : last.inventory_after < 0 ? 'text-red-400' : 'text-muted'}`}>
                            {last.inventory_after.toFixed(2)}
                        </span>
                    </div>
                    <div>
                        <span className="text-muted">Action: </span>
                        <span className={last.side === 'buy' ? 'text-green-400' : last.side === 'sell' ? 'text-red-400' : 'text-muted'}>
                            {last.side ? last.side.toUpperCase() : ''} {last.action.toFixed(3)}
                        </span>
                    </div>
                    {last.event === 'liquidation' && (
                        <div className="flex items-center gap-1 text-red-500 font-bold animate-pulse">
                            <span>💀 REKT</span>
                        </div>
                    )}
                </div>
            </div>

            <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={data}>
                        <defs>
                            <linearGradient id="inventoryGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                        <XAxis
                            dataKey="step"
                            stroke="#71717a"
                            fontSize={10}
                            tick={false}
                            axisLine={false}
                        />
                        <YAxis
                            yAxisId="price"
                            domain={['auto', 'auto']}
                            orientation="right"
                            stroke="#71717a"
                            fontSize={10}
                            tickLine={false}
                            axisLine={false}
                            tickFormatter={(val) => val.toFixed(0)}
                        />
                        <YAxis
                            yAxisId="inventory"
                            domain={['auto', 'auto']}
                            orientation="left"
                            stroke="#10b981"
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
                        {/* Price Line */}
                        <Line
                            yAxisId="price"
                            type="monotone"
                            dataKey="p_base"
                            stroke="#52525b"
                            dot={false}
                            strokeWidth={1}
                        />

                        {/* Buy/Sell Markers */}
                        {data.map((entry, index) => (
                            entry.action > 0.01 && (
                                <ReferenceDot
                                    key={`dot-${index}`}
                                    yAxisId="price"
                                    x={entry.step}
                                    y={entry.p_exec}
                                    r={3 + (entry.action * 3)}
                                    fill={entry.side === 'buy' ? '#4ade80' : '#f43f5e'}
                                    stroke="none"
                                    opacity={0.8}
                                />
                            )
                        ))}

                        {/* Liquidation Skull Marker */}
                        {data.map((entry, index) => (
                            entry.event === 'liquidation' && (
                                <ReferenceDot
                                    key={`rekt-${index}`}
                                    yAxisId="price"
                                    x={entry.step}
                                    y={entry.p_exec}
                                    r={8}
                                    fill="#ef4444"
                                    stroke="#ffffff"
                                    strokeWidth={2}
                                    label={{ position: 'top', value: '💀', fontSize: 20 }}
                                />
                            )
                        ))}

                        {/* Inventory Area */}
                        <Area
                            yAxisId="inventory"
                            type="stepAfter"
                            dataKey="inventory_after"
                            stroke="#10b981"
                            fill="url(#inventoryGradient)"
                            strokeWidth={2}
                        />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
