import React, { useEffect, useRef } from 'react';
import { Terminal, Circle } from 'lucide-react';

export interface LogEntry {
    id: string;
    timestamp: string;
    level: 'info' | 'warning' | 'error' | 'success';
    message: string;
}

interface LogTerminalProps {
    logs: LogEntry[];
    maxHeight?: string;
}

export const LogTerminal: React.FC<LogTerminalProps> = ({ logs, maxHeight = "300px" }) => {
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    const getLevelColor = (level: string) => {
        switch (level) {
            case 'info': return 'text-muted';
            case 'warning': return 'text-accent';
            case 'error': return 'text-red-400';
            case 'success': return 'text-primary';
            default: return 'text-muted';
        }
    };

    return (
        <div className="card flex flex-col h-full bg-black/40 border-primary/20">
            <div className="flex items-center gap-2 mb-3 pb-2 border-b border-white/5">
                <Terminal size={14} className="text-secondary" />
                <span className="text-xs font-mono text-secondary">SYSTEM LOG</span>
                <div className="ml-auto flex gap-1.5">
                    <Circle size={8} className="fill-red-500 text-red-500 opacity-50" />
                    <Circle size={8} className="fill-yellow-500 text-yellow-500 opacity-50" />
                    <Circle size={8} className="fill-green-500 text-green-500 opacity-50" />
                </div>
            </div>

            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto font-mono text-xs space-y-1 pr-2"
                style={{ maxHeight }}
            >
                {logs.length === 0 ? (
                    <div className="text-muted/50 italic p-2">No logs available...</div>
                ) : (
                    logs.map((log) => (
                        <div key={log.id} className="flex gap-3 hover:bg-white/5 p-0.5 rounded px-1 transition-colors">
                            <span className="text-muted/50 w-16 shrink-0">{log.timestamp}</span>
                            <span className={getLevelColor(log.level)}>{log.message}</span>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};
