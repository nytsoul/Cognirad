import React from 'react';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import { Activity, AlertTriangle, CheckCircle, FileText, Brain } from 'lucide-react';

const ProbabilityBar = ({ label, probability, confidence, isUncertain }) => {
    return (
        <div className="mb-5 last:mb-0">
            <div className="flex justify-between items-end mb-2">
                <span className={clsx("font-bold text-[11px] uppercase tracking-wider", isUncertain ? "text-accent" : "text-slate-400")}>
                    {label}
                </span>
                <span className="text-xs font-black text-white">{(probability * 100).toFixed(1)}%</span>
            </div>
            <div className="h-1.5 w-full bg-slate-900 rounded-full overflow-hidden border border-white/5">
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${probability * 100}%` }}
                    transition={{ duration: 1, ease: "easeOut" }}
                    className={clsx(
                        "h-full rounded-full transition-all",
                        isUncertain ? "bg-accent shadow-[0_0_10px_#f59e0b]" : "bg-primary-500 shadow-glow"
                    )}
                />
            </div>
            {isUncertain && (
                <p className="mt-1.5 text-[9px] font-bold text-accent/70 flex items-center gap-1.5">
                    <AlertTriangle size={10} />
                    CONFIDENCE_THRESHOLD_UNMET
                </p>
            )}
        </div>
    );
};

const ReportSection = ({ title, content, icon: Icon }) => (
    <div className="last:mb-0">
        <h3 className="text-[10px] uppercase tracking-[0.2em] text-slate-600 font-black mb-3 flex items-center gap-3">
            <div className="w-1.5 h-1.5 rounded-full bg-primary-500/40" />
            {title}
        </h3>
        <p className="text-slate-300 leading-relaxed text-sm bg-background border border-border p-5 rounded-2xl font-medium tracking-wide">
            {content || "Generating clinical synthesis..."}
        </p>
    </div>
);

const ReportDisplay = ({ report }) => {
    if (!report) return null;

    const { findings, impression, predicted_diseases, uncertain_findings, warnings } = report;

    // Combine predictions for display
    const allPredictions = [
        ...(predicted_diseases || []),
        ...(uncertain_findings || [])
    ].sort((a, b) => b.probability - a.probability);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-surface rounded-2xl shadow-glass border border-border overflow-hidden"
        >
            <div className="p-6 border-b border-border bg-white/5 flex items-center justify-between">
                <h2 className="text-xl font-black text-white flex items-center gap-3">
                    <div className="p-2 bg-primary-500/10 rounded-lg">
                        <FileText className="text-primary-500" size={20} />
                    </div>
                    Automated Radiology Synthesis
                </h2>
                {warnings && warnings.length > 0 && (
                    <div className="flex items-center gap-2 px-3 py-1 bg-danger/10 text-danger text-[10px] font-black uppercase tracking-widest rounded-full border border-danger/20">
                        <AlertTriangle size={12} />
                        INTERVENTION_REQUIRED
                    </div>
                )}
            </div>

            <div className="p-8 grid grid-cols-1 lg:grid-cols-3 gap-10">
                {/* Main Content */}
                <div className="lg:col-span-2 space-y-8">
                    <div className="grid grid-cols-1 gap-8">
                        <ReportSection title="Clinical Context" content={report.clinical_indication} icon={Activity} />
                        <ReportSection title="Findings & Observations" content={findings} icon={Brain} />
                        <ReportSection title="Diagnostic Impression" content={impression} icon={CheckCircle} />
                    </div>

                    {warnings && warnings.length > 0 && (
                        <div className="p-5 bg-danger/5 border border-danger/10 rounded-2xl">
                            <h4 className="text-[10px] font-black text-danger uppercase mb-3 flex items-center gap-2 tracking-[0.2em]">
                                <AlertTriangle size={14} />
                                Clinical Safeguard Alerts
                            </h4>
                            <ul className="space-y-2">
                                {warnings.map((w, i) => (
                                    <li key={i} className="text-xs text-danger/80 flex items-start gap-2">
                                        <span className="mt-1.5 w-1 h-1 rounded-full bg-danger opacity-40 shrink-0" />
                                        {w}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>

                {/* Sidebar Analysis */}
                <div className="space-y-6">
                    <div className="bg-background/50 rounded-2xl p-6 border border-border shadow-inner">
                        <h3 className="text-xs font-black text-slate-400 mb-6 flex items-center gap-3 uppercase tracking-widest">
                            <Activity size={18} className="text-primary-500" />
                            Disease Likelihood
                        </h3>

                        {allPredictions.length > 0 ? (
                            allPredictions.map((pred, idx) => (
                                <ProbabilityBar
                                    key={idx}
                                    label={pred.label}
                                    probability={pred.probability}
                                    confidence={pred.confidence}
                                    isUncertain={uncertain_findings?.some(u => u.label === pred.label)}
                                />
                            ))
                        ) : (
                            <p className="text-xs text-slate-600 italic">No specific pathologies identified above threshold.</p>
                        )}
                    </div>

                    <div className="p-6 border border-white/5 rounded-2xl bg-white/[0.02]">
                        <h4 className="text-[10px] font-black text-slate-600 uppercase mb-3 tracking-widest">Expert Verification</h4>
                        <p className="text-[11px] text-slate-500 leading-relaxed">
                            Report generated using CogniRad++ v1.0 engine. Human clinician review is mandatory for all diagnostic signatures.
                        </p>
                    </div>
                </div>
            </div>
        </motion.div>
    );
};

export default ReportDisplay;
