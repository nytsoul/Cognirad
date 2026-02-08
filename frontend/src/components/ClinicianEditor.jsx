import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Edit3, Check, X, RefreshCw, AlertCircle, Microscope } from 'lucide-react';
import clsx from 'clsx';

const ClinicianEditor = ({ predictions = [], onUpdate, onRegenerate, isRegenerating }) => {
    const [editedPredictions, setEditedPredictions] = useState([...predictions]);
    const [hasChanges, setHasChanges] = useState(false);

    const toggleStatus = (idx) => {
        const newPreds = [...editedPredictions];
        newPreds[idx] = { ...newPreds[idx], probability: newPreds[idx].probability > 0.5 ? 0.1 : 0.9 };
        setEditedPredictions(newPreds);
        setHasChanges(true);
    };

    return (
        <div className="card-premium h-fit overflow-hidden">
            <div className="p-4 bg-white/5 border-b border-border flex items-center justify-between">
                <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest flex items-center gap-3">
                    <Edit3 size={16} className="text-primary-500" />
                    Clinician Override
                </h3>
                {hasChanges && (
                    <button
                        onClick={onRegenerate}
                        disabled={isRegenerating}
                        className="flex items-center gap-2 px-3 py-1.5 bg-primary-600 hover:bg-primary-500 text-white text-[10px] font-black uppercase rounded-lg transition-all shadow-glow disabled:opacity-50"
                    >
                        {isRegenerating ? <div className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <RefreshCw size={12} />}
                        Sync Changes
                    </button>
                )}
            </div>

            <div className="p-6 space-y-5">
                <p className="text-[11px] text-slate-500 leading-relaxed italic">
                    AI hypotheses can be adjusted to influence the RCTA verification loop.
                </p>

                <div className="grid grid-cols-2 gap-2.5">
                    {editedPredictions.map((pred, idx) => {
                        const isPresent = pred.probability > 0.5;
                        return (
                            <button
                                key={pred.label}
                                onClick={() => toggleStatus(idx)}
                                className={clsx(
                                    "flex items-center justify-between p-3 rounded-xl border text-[11px] font-bold transition-all transition-[transform,shadow]",
                                    isPresent
                                        ? "bg-primary-900/20 border-primary-500/50 text-white shadow-glow translate-y-[-1px]"
                                        : "bg-background border-border text-slate-600 hover:border-slate-700"
                                )}
                            >
                                <span className="truncate pr-2">{pred.label}</span>
                                <div className={clsx(
                                    "shrink-0 w-4 h-4 rounded flex items-center justify-center",
                                    isPresent ? "bg-primary-500 text-white" : "bg-slate-800 text-slate-500"
                                )}>
                                    {isPresent ? <Check size={10} strokeWidth={4} /> : <X size={10} strokeWidth={4} />}
                                </div>
                            </button>
                        );
                    })}
                </div>

                <div className="mt-6 p-4 bg-primary-950/20 rounded-2xl flex gap-4 border border-primary-900/10">
                    <div className="p-2 bg-primary-500/10 rounded-xl h-fit">
                        <Microscope size={16} className="text-primary-400" />
                    </div>
                    <p className="text-[10px] text-slate-400 leading-relaxed font-medium">
                        <strong className="text-primary-400 block mb-1">PRO-FA Feedback Active</strong>
                        Manual adjustments will re-align the visual perception layers and update the final structured report narrative.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default ClinicianEditor;
