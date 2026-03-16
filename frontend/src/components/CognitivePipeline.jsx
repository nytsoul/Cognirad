import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle2, Circle, ArrowRight } from 'lucide-react';
import clsx from 'clsx';

const CognitivePipeline = ({ currentStage, stages = [] }) => {
    return (
        <div className="w-full py-4 px-2">
            <div className="flex items-center justify-between max-w-2xl mx-auto">
                {stages.map((stage, idx) => {
                    const isComplete = idx < currentStage;
                    const isActive = idx === currentStage;

                    return (
                        <React.Fragment key={stage.name}>
                            <div className="flex flex-col items-center gap-1.5 relative group">
                                <motion.div
                                    initial={false}
                                    animate={{
                                        scale: isActive ? 1.05 : 1,
                                        backgroundColor: isComplete ? '#0ea5e9' : (isActive ? 'rgba(14,165,233,0.1)' : '#12151c'),
                                        borderColor: isComplete || isActive ? '#0ea5e9' : '#1e293b'
                                    }}
                                    className={clsx(
                                        "w-8 h-8 rounded-lg border flex items-center justify-center transition-all",
                                        isActive && "ring-2 ring-primary-500/20 shadow-glow"
                                    )}
                                >
                                    {isComplete ? (
                                        <CheckCircle2 size={16} className="text-white" />
                                    ) : (
                                        <span className={clsx(
                                            "text-[10px] font-black",
                                            isActive ? "text-primary-500" : "text-slate-600"
                                        )}>{idx + 1}</span>
                                    )}
                                </motion.div>
                                <div className="text-center">
                                    <p className={clsx(
                                        "text-[9px] font-black uppercase tracking-widest",
                                        isActive ? "text-primary-400" : "text-slate-600"
                                    )}>{stage.id}</p>
                                </div>
                            </div>

                            {idx < stages.length - 1 && (
                                <div className="flex-1 h-[1px] bg-slate-800 mx-3 -mt-4 relative overflow-hidden">
                                    <motion.div
                                        initial={{ left: '-100%' }}
                                        animate={{ left: isComplete ? '0%' : '-100%' }}
                                        className="absolute inset-0 bg-primary-500 transition-all duration-700"
                                    />
                                </div>
                            )}
                        </React.Fragment>
                    );
                })}
            </div>
        </div>
    );
};

export default CognitivePipeline;
