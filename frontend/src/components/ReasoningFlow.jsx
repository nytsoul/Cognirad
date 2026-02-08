import React from 'react';
import { motion } from 'framer-motion';
import { Image, FileText, Activity, ArrowRight, RefreshCw } from 'lucide-react';
import clsx from 'clsx';

const ReasoningStep = ({ icon: Icon, label, title, desc, isActive, isComplete }) => (
    <div className={clsx(
        "flex flex-col items-center p-6 rounded-2xl border transition-all relative z-10",
        isActive ? "bg-primary-500/10 border-primary-500 shadow-glow scale-105" : "bg-white/5 border-white/5",
        isComplete && "border-primary-500/30"
    )}>
        <div className={clsx(
            "p-4 rounded-xl mb-4 transition-all duration-300",
            isActive ? "bg-primary-500 text-white shadow-glow" : (isComplete ? "bg-primary-900/40 text-primary-400" : "bg-slate-800 text-slate-500")
        )}>
            <Icon size={28} />
        </div>
        <span className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-600 mb-2">{label}</span>
        <h4 className="text-sm font-bold text-slate-200">{title}</h4>
        <p className="text-[11px] text-slate-500 text-center mt-3 leading-relaxed opacity-80">{desc}</p>

        {isActive && (
            <motion.div
                layoutId="active-indicator"
                className="absolute top-2 right-2"
            >
                <div className="w-2.5 h-2.5 bg-primary-500 rounded-full animate-ping shadow-glow" />
            </motion.div>
        )}
    </div>
);

const ReasoningFlow = ({ stages = [], activeStage = 0 }) => {
    return (
        <div className="relative p-10 bg-surface/50 overflow-hidden isolate">
            {/* Background Neural Grid (Faded) */}
            <div className="absolute inset-0 -z-10 opacity-[0.03] pointer-events-none"
                style={{ backgroundImage: 'radial-gradient(#fff 1px, transparent 1px)', backgroundSize: '24px 24px' }} />

            {/* Connection Line */}
            <div className="absolute top-1/2 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-primary-500/20 to-transparent -translate-y-1/2 -z-10" />

            <div className="grid grid-cols-1 md:grid-cols-3 gap-10 relative">
                <ReasoningStep
                    icon={Image}
                    label="PRO-FA"
                    title="Hierarchical Perception"
                    desc="Multi-scale concept extraction."
                    isActive={activeStage === 0}
                    isComplete={activeStage > 0}
                />

                <ReasoningStep
                    icon={Activity}
                    label="MIX-MLP"
                    title="Clinical Diagnosis"
                    desc="Multi-path MLP hypothesis."
                    isActive={activeStage === 1}
                    isComplete={activeStage > 1}
                />

                <ReasoningStep
                    icon={RefreshCw}
                    label="RCTA"
                    title="Triangular Verification"
                    desc="Closed-loop verification."
                    isActive={activeStage === 2}
                    isComplete={activeStage > 2}
                />
            </div>

            <div className="mt-12 flex flex-col items-center">
                <div className="px-5 py-2.5 bg-background border border-border text-[10px] font-black font-mono rounded-xl flex items-center gap-4 tracking-[0.1em]">
                    <span className="text-primary-500">RAW_RAD_DATA</span>
                    <ArrowRight size={14} className="text-slate-700" />
                    <span className="text-primary-400 font-bold uppercase">cognitive_verification_stream</span>
                    <div className="flex h-2 w-2 rounded-full bg-primary-500/40 animate-pulse ml-2" />
                </div>
            </div>
        </div>
    );
};

export default ReasoningFlow;
