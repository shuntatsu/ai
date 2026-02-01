import React from 'react';
import { cn } from '../lib/utils';

interface StatsCardProps {
    title: string;
    value: string | number;
    icon: React.ElementType;
    trend?: {
        value: number;
        isPositive: boolean;
    };
    className?: string;
    valueClassName?: string;
}

export const StatsCard: React.FC<StatsCardProps> = ({
    title,
    value,
    icon: Icon,
    trend,
    className,
    valueClassName,
}) => {
    return (
        <div className={cn("card flex flex-col justify-between hover:border-primary/50 transition-colors", className)}>
            <div className="flex justify-between items-start mb-2">
                <span className="text-muted text-sm font-medium">{title}</span>
                <div className="p-2 rounded-md bg-surface border border-border">
                    <Icon size={16} className="text-primary" />
                </div>
            </div>

            <div className="flex items-end justify-between">
                <div className={cn("text-2xl font-bold font-mono text-text", valueClassName)}>
                    {value}
                </div>

                {trend && (
                    <div className={cn(
                        "text-xs font-medium px-1.5 py-0.5 rounded flex items-center gap-1",
                        trend.isPositive ? "text-primary bg-primary/10" : "text-red-400 bg-red-400/10"
                    )}>
                        {trend.value > 0 ? "+" : ""}{trend.value}%
                    </div>
                )}
            </div>
        </div>
    );
};
