import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';
import {
    Activity, AlertTriangle, CheckCircle, FileText, Brain, Download,
    FileDown, Copy, Check, Clock, Printer, ChevronDown,
    Sparkles, BarChart3, ShieldCheck, Clipboard
} from 'lucide-react';

/* ───────────── Probability Bar ───────────── */
const ProbabilityBar = ({ label, probability, confidence, isUncertain, index }) => (
    <motion.div
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: index * 0.06 }}
        className="mb-4 last:mb-0 group"
    >
        <div className="flex justify-between items-end mb-1.5">
            <span className={clsx(
                "font-bold text-[11px] uppercase tracking-wider flex items-center gap-2",
                isUncertain ? "text-accent" : "text-slate-400"
            )}>
                {isUncertain && <AlertTriangle size={10} className="text-accent" />}
                {label}
            </span>
            <div className="flex items-center gap-2">
                <span className="text-[9px] font-mono text-slate-600">
                    conf {(confidence * 100).toFixed(0)}%
                </span>
                <span className="text-xs font-black text-white tabular-nums">
                    {(probability * 100).toFixed(1)}%
                </span>
            </div>
        </div>
        <div className="h-2 w-full bg-slate-900/80 rounded-full overflow-hidden border border-white/5">
            <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${probability * 100}%` }}
                transition={{ duration: 0.8, ease: "easeOut", delay: index * 0.06 }}
                className={clsx(
                    "h-full rounded-full",
                    isUncertain
                        ? "bg-gradient-to-r from-accent/80 to-accent shadow-[0_0_10px_#f59e0b40]"
                        : probability > 0.8
                            ? "bg-gradient-to-r from-danger/80 to-danger shadow-[0_0_10px_#ef444440]"
                            : "bg-gradient-to-r from-primary-600 to-primary-400 shadow-glow"
                )}
            />
        </div>
    </motion.div>
);

/* ───────────── Report Section ───────────── */
const ReportSection = ({ title, content, icon: Icon, accentColor = "primary" }) => (
    <div className="group">
        <h3 className="text-[10px] uppercase tracking-[0.2em] text-slate-600 font-black mb-3 flex items-center gap-3">
            <div className={clsx(
                "w-5 h-5 rounded-md flex items-center justify-center",
                accentColor === "primary" ? "bg-primary-500/10" : accentColor === "success" ? "bg-success/10" : "bg-accent/10"
            )}>
                <Icon size={11} className={clsx(
                    accentColor === "primary" ? "text-primary-400" : accentColor === "success" ? "text-success" : "text-accent"
                )} />
            </div>
            {title}
        </h3>
        <div className="text-slate-300 leading-relaxed text-sm bg-background/80 border border-border p-5 rounded-2xl font-medium tracking-wide hover:border-primary-500/20 transition-colors duration-300 relative overflow-hidden">
            <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-primary-500/40 to-transparent rounded-l-2xl" />
            <p className="pl-2">{content || "Generating clinical synthesis..."}</p>
        </div>
    </div>
);

/* ───────────── Download Helpers ───────────── */
function buildReportText(report) {
    const timestamp = new Date().toLocaleString();
    const divider = '='.repeat(60);
    const thinDivider = '-'.repeat(60);

    let text = '';
    text += `${divider}\n`;
    text += `  COGNIRAD++ AUTOMATED RADIOLOGY REPORT\n`;
    text += `${divider}\n`;
    text += `  Generated: ${timestamp}\n`;
    text += `  Engine:    CogniRad++ v1.0 — Cognitive Reasoning Pipeline\n`;
    text += `${divider}\n\n`;

    text += `CLINICAL INDICATION\n${thinDivider}\n`;
    text += `${report.clinical_indication || 'N/A'}\n\n`;

    text += `FINDINGS\n${thinDivider}\n`;
    text += `${report.findings || 'N/A'}\n\n`;

    text += `IMPRESSION\n${thinDivider}\n`;
    text += `${report.impression || 'N/A'}\n\n`;

    if (report.predicted_diseases?.length) {
        text += `DETECTED PATHOLOGIES\n${thinDivider}\n`;
        report.predicted_diseases.forEach(d => {
            text += `  * ${d.label.padEnd(30)} Prob: ${(d.probability * 100).toFixed(1)}%  Conf: ${(d.confidence * 100).toFixed(0)}%\n`;
        });
        text += '\n';
    }

    if (report.uncertain_findings?.length) {
        text += `UNCERTAIN FINDINGS (REQUIRES REVIEW)\n${thinDivider}\n`;
        report.uncertain_findings.forEach(d => {
            text += `  ! ${d.label.padEnd(30)} Prob: ${(d.probability * 100).toFixed(1)}%  Conf: ${(d.confidence * 100).toFixed(0)}%\n`;
        });
        text += '\n';
    }

    if (report.warnings?.length) {
        text += `CLINICAL SAFEGUARD ALERTS\n${thinDivider}\n`;
        report.warnings.forEach(w => { text += `  ! ${w}\n`; });
        text += '\n';
    }

    text += `${divider}\n`;
    text += `  DISCLAIMER: AI-generated report. Human clinician review is\n`;
    text += `  mandatory before any diagnostic or treatment decisions.\n`;
    text += `${divider}\n`;
    return text;
}

function downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function buildReportJSON(report) {
    return JSON.stringify({
        meta: {
            generator: 'CogniRad++ v1.0',
            timestamp: new Date().toISOString(),
            disclaimer: 'AI-generated report. Human clinician review is mandatory.',
        },
        clinical_indication: report.clinical_indication,
        findings: report.findings,
        impression: report.impression,
        predicted_diseases: report.predicted_diseases,
        uncertain_findings: report.uncertain_findings,
        warnings: report.warnings,
    }, null, 2);
}

/* ───────────── Download Dropdown ───────────── */
const DownloadMenu = ({ report }) => {
    const [open, setOpen] = useState(false);
    const [copied, setCopied] = useState(false);

    const handleDownloadTxt = () => {
        const text = buildReportText(report);
        const ts = new Date().toISOString().slice(0, 10);
        downloadFile(text, `CogniRad_Report_${ts}.txt`, 'text/plain');
        setOpen(false);
    };

    const handleDownloadJson = () => {
        const json = buildReportJSON(report);
        const ts = new Date().toISOString().slice(0, 10);
        downloadFile(json, `CogniRad_Report_${ts}.json`, 'application/json');
        setOpen(false);
    };

    const handleCopy = async () => {
        const text = buildReportText(report);
        await navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
        setOpen(false);
    };

    const handlePrint = () => {
        const text = buildReportText(report);
        const win = window.open('', '_blank');
        win.document.write(`<html><head><title>CogniRad++ Report</title>
            <style>body{font-family:'Courier New',monospace;white-space:pre-wrap;padding:40px;font-size:13px;line-height:1.6;color:#1a1a1a;}</style>
            </head><body>${text.replace(/\n/g, '<br>')}</body></html>`);
        win.document.close();
        win.print();
        setOpen(false);
    };

    return (
        <div className="relative">
            <button
                onClick={() => setOpen(!open)}
                className="flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-500 text-white text-xs font-black uppercase tracking-wider rounded-xl transition-all shadow-glow border border-primary-500/50"
            >
                <Download size={14} />
                Export Report
                <ChevronDown size={12} className={clsx("transition-transform", open && "rotate-180")} />
            </button>

            <AnimatePresence>
                {open && (
                    <>
                        <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
                        <motion.div
                            initial={{ opacity: 0, y: -8, scale: 0.95 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, y: -8, scale: 0.95 }}
                            transition={{ duration: 0.15 }}
                            className="absolute right-0 top-full mt-2 z-50 w-56 bg-surface border border-border rounded-2xl shadow-glass overflow-hidden"
                        >
                            <div className="p-1.5">
                                <button onClick={handleDownloadTxt} className="w-full flex items-center gap-3 px-4 py-3 text-sm text-slate-300 hover:bg-white/5 rounded-xl transition-colors font-medium">
                                    <FileDown size={16} className="text-primary-400" />
                                    Download .txt
                                </button>
                                <button onClick={handleDownloadJson} className="w-full flex items-center gap-3 px-4 py-3 text-sm text-slate-300 hover:bg-white/5 rounded-xl transition-colors font-medium">
                                    <FileText size={16} className="text-blue-400" />
                                    Download .json
                                </button>
                                <div className="h-px bg-border mx-3 my-1" />
                                <button onClick={handleCopy} className="w-full flex items-center gap-3 px-4 py-3 text-sm text-slate-300 hover:bg-white/5 rounded-xl transition-colors font-medium">
                                    {copied ? <Check size={16} className="text-success" /> : <Clipboard size={16} className="text-slate-500" />}
                                    {copied ? 'Copied!' : 'Copy to Clipboard'}
                                </button>
                                <button onClick={handlePrint} className="w-full flex items-center gap-3 px-4 py-3 text-sm text-slate-300 hover:bg-white/5 rounded-xl transition-colors font-medium">
                                    <Printer size={16} className="text-slate-500" />
                                    Print Report
                                </button>
                            </div>
                        </motion.div>
                    </>
                )}
            </AnimatePresence>
        </div>
    );
};

/* ───────────── Stats Row ───────────── */
const StatCard = ({ label, value, icon: Icon, color = "primary" }) => (
    <div className="flex items-center gap-3 p-3 bg-background/60 rounded-xl border border-white/5">
        <div className={clsx(
            "p-2 rounded-lg",
            color === "primary" ? "bg-primary-500/10" : color === "success" ? "bg-success/10" : color === "danger" ? "bg-danger/10" : "bg-accent/10"
        )}>
            <Icon size={14} className={clsx(
                color === "primary" ? "text-primary-400" : color === "success" ? "text-success" : color === "danger" ? "text-danger" : "text-accent"
            )} />
        </div>
        <div>
            <div className="text-white text-sm font-black tabular-nums">{value}</div>
            <div className="text-[9px] font-bold text-slate-600 uppercase tracking-widest">{label}</div>
        </div>
    </div>
);

/* ═══════════════ MAIN COMPONENT ═══════════════ */
const ReportDisplay = ({ report }) => {
    if (!report) return null;

    const { findings, impression, predicted_diseases, uncertain_findings, warnings } = report;

    const allPredictions = [
        ...(predicted_diseases || []),
        ...(uncertain_findings || [])
    ].sort((a, b) => b.probability - a.probability);

    const highRisk = allPredictions.filter(p => p.probability > 0.8).length;
    const maxConf = allPredictions.length
        ? Math.max(...allPredictions.map(p => p.confidence))
        : 0;

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-surface rounded-2xl shadow-glass border border-border overflow-hidden"
        >
            {/* ─── Header ─── */}
            <div className="p-6 border-b border-border bg-gradient-to-r from-white/[0.04] to-transparent flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div className="flex items-center gap-4">
                    <div className="p-2.5 bg-primary-500/10 rounded-xl border border-primary-500/20">
                        <FileText className="text-primary-500" size={22} />
                    </div>
                    <div>
                        <h2 className="text-xl font-black text-white tracking-tight">Automated Radiology Synthesis</h2>
                        <div className="flex items-center gap-3 mt-1">
                            <span className="text-[10px] font-bold text-slate-500 flex items-center gap-1.5">
                                <Clock size={10} />
                                {new Date().toLocaleString()}
                            </span>
                            {warnings?.length > 0 && (
                                <span className="flex items-center gap-1.5 px-2 py-0.5 bg-danger/10 text-danger text-[9px] font-black uppercase tracking-widest rounded-full border border-danger/20">
                                    <AlertTriangle size={9} />
                                    INTERVENTION_REQUIRED
                                </span>
                            )}
                        </div>
                    </div>
                </div>

                <DownloadMenu report={report} />
            </div>

            {/* ─── Stats Row ─── */}
            <div className="px-6 py-4 border-b border-border bg-white/[0.015] grid grid-cols-2 md:grid-cols-4 gap-3">
                <StatCard label="Pathologies" value={allPredictions.length} icon={BarChart3} color="primary" />
                <StatCard label="High Risk" value={highRisk} icon={AlertTriangle} color={highRisk > 0 ? "danger" : "success"} />
                <StatCard label="Uncertain" value={uncertain_findings?.length || 0} icon={Activity} color="accent" />
                <StatCard label="Peak Conf." value={`${(maxConf * 100).toFixed(0)}%`} icon={ShieldCheck} color="success" />
            </div>

            {/* ─── Body ─── */}
            <div className="p-8 grid grid-cols-1 lg:grid-cols-3 gap-10">
                {/* Main Content */}
                <div className="lg:col-span-2 space-y-6">
                    <ReportSection title="Clinical Context" content={report.clinical_indication} icon={Activity} accentColor="accent" />
                    <ReportSection title="Findings & Observations" content={findings} icon={Brain} accentColor="primary" />
                    <ReportSection title="Diagnostic Impression" content={impression} icon={CheckCircle} accentColor="success" />

                    {/* Warnings */}
                    {warnings?.length > 0 && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="p-5 bg-danger/5 border border-danger/10 rounded-2xl"
                        >
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
                        </motion.div>
                    )}
                </div>

                {/* Sidebar Analysis */}
                <div className="space-y-6">
                    <div className="bg-background/50 rounded-2xl p-6 border border-border shadow-inner">
                        <h3 className="text-xs font-black text-slate-400 mb-5 flex items-center gap-3 uppercase tracking-widest">
                            <Activity size={16} className="text-primary-500" />
                            Disease Likelihood
                        </h3>

                        {allPredictions.length > 0 ? (
                            allPredictions.map((pred, idx) => (
                                <ProbabilityBar
                                    key={idx}
                                    index={idx}
                                    label={pred.label}
                                    probability={pred.probability}
                                    confidence={pred.confidence}
                                    isUncertain={uncertain_findings?.some(u => u.label === pred.label)}
                                />
                            ))
                        ) : (
                            <div className="text-center py-6">
                                <Sparkles size={24} className="text-success mx-auto mb-2 opacity-60" />
                                <p className="text-xs text-slate-500 font-medium">No pathologies detected above threshold.</p>
                            </div>
                        )}
                    </div>

                    {/* Verification Badge */}
                    <div className="p-5 border border-white/5 rounded-2xl bg-gradient-to-b from-white/[0.03] to-transparent">
                        <div className="flex items-center gap-3 mb-3">
                            <div className="p-2 bg-primary-500/10 rounded-lg">
                                <ShieldCheck size={16} className="text-primary-400" />
                            </div>
                            <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Expert Verification</h4>
                        </div>
                        <p className="text-[11px] text-slate-500 leading-relaxed">
                            Generated by <span className="text-primary-400 font-bold">CogniRad++ v1.0</span> cognitive reasoning engine. Human clinician review is mandatory for all diagnostic signatures.
                        </p>
                        <div className="mt-3 flex items-center gap-2 text-[9px] font-bold text-slate-600">
                            <div className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />
                            PRO-FA / MIX-MLP / RCTA pipeline verified
                        </div>
                    </div>
                </div>
            </div>
        </motion.div>
    );
};

export default ReportDisplay;
