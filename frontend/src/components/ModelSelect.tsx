import React, { useState, useEffect } from 'react';
import { Loader2 } from 'lucide-react';

interface ModelInfo {
    id: string;
    path: string;
    size_bytes: number;
    modified_at: number;
    name?: string;
    is_checkpoint?: boolean;
}

interface ModelSelectProps {
    value: string | null;
    onChange: (value: string | null) => void;
}

export const ModelSelect: React.FC<ModelSelectProps> = ({ value, onChange }) => {
    const [models, setModels] = useState<ModelInfo[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const fetchModels = async () => {
            setLoading(true);
            try {
                const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';
                const res = await fetch(`${API_BASE}/api/models`);
                if (res.ok) {
                    const data = await res.json();
                    setModels(data.models || []);
                }
            } catch (e) {
                console.error("Failed to fetch models", e);
            } finally {
                setLoading(false);
            }
        };
        fetchModels();
    }, []);

    const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const val = e.target.value;
        onChange(val === "" ? null : val);
    };

    return (
        <div className="relative">
            <select
                value={value || ""}
                onChange={handleChange}
                disabled={loading}
                className="w-full bg-background border border-border rounded px-3 py-2 text-sm text-text focus:outline-none focus:border-primary disabled:opacity-50"
            >
                <option value="">(Start from Scratch)</option>
                {models.map(m => (
                    <option key={m.path} value={m.path}>
                        {m.is_checkpoint ? '[Checkpoint] ' : '[Model] '}
                        {m.name || m.id} ({new Date(m.modified_at * 1000).toLocaleString()})
                    </option>
                ))}
            </select>
            {loading && (
                <div className="absolute right-2 top-1/2 -translate-y-1/2">
                    <Loader2 size={16} className="animate-spin text-muted" />
                </div>
            )}
        </div>
    );
};
