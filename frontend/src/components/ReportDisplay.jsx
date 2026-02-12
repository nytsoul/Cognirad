import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';
import {
    Activity, AlertTriangle, CheckCircle, FileText, Brain, Download,
    FileDown, Copy, Check, Clock, Printer, ChevronDown,
    Sparkles, BarChart3, ShieldCheck, Clipboard
} from 'lucide-react';

/* ───────────── Analysis Card ───────────── */
const AnalysisCard = ({ label, probability, confidence, isUncertain, index }) => {
    const getSeverity = (prob) => {
        if (prob > 0.8) return {
            color: "danger", label: "Critical", icon: AlertTriangle,
            glow: "shadow-[0_2px_10px_#ef444410]"
        };
        if (prob > 0.5) return {
            color: "accent", label: "Elevated", icon: Activity,
            glow: "shadow-[0_2px_10px_#f59e0b10]"
        };
        if (prob > 0.2) return {
            color: "primary", label: "Stable", icon: FileText,
            glow: "shadow-[0_2px_10px_#0ea5e908]"
        };
        return {
            color: "success", label: "Normal", icon: CheckCircle,
            glow: "shadow-[0_2px_10px_#10b98108]"
        };
    };

    const severity = getSeverity(probability);
    const StatusIcon = severity.icon;

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.05 }}
            className={clsx(
                "p-3 rounded-xl border transition-all duration-300 flex flex-col gap-2 group",
                "bg-surface/40 hover:bg-surface/60 border-border hover:border-white/10",
                severity.glow
            )}
        >
            <div className="flex justify-between items-start">
                <div className={clsx(
                    "text-[8px] font-black uppercase tracking-widest flex items-center gap-1",
                    severity.color === "danger" ? "text-danger" :
                        severity.color === "accent" ? "text-accent" :
                            severity.color === "primary" ? "text-primary-400" : "text-success"
                )}>
                    <StatusIcon size={9} />
                    {severity.label}
                </div>
                <div className="text-sm font-black text-white tabular-nums">
                    {(probability * 100).toFixed(0)}%
                </div>
            </div>

            <h4 className="text-[10px] font-bold text-slate-200 truncate pr-2 uppercase tracking-wide">
                {label}
            </h4>

            <div className="flex items-center gap-1.5 mt-1">
                <div className="flex-1 h-1 bg-slate-900/60 rounded-full overflow-hidden">
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${probability * 100}%` }}
                        transition={{ duration: 0.8, delay: index * 0.05 }}
                        className={clsx(
                            "h-full rounded-full",
                            severity.color === "danger" ? "bg-danger" :
                                severity.color === "accent" ? "bg-accent" :
                                    severity.color === "primary" ? "bg-primary-500" : "bg-success"
                        )}
                    />
                </div>
                {isUncertain && <div className="w-1 h-1 rounded-full bg-accent animate-pulse" title="Requires Review" />}
            </div>
        </motion.div>
    );
};

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

            {/* ─── Body ─── */}
            <div className="p-6 space-y-8">
                {/* Analysis Grid */}
                <div>
                    <h3 className="text-[10px] font-black text-slate-500 mb-5 flex items-center gap-3 uppercase tracking-[0.2em] pl-1">
                        <Activity size={14} className="text-primary-500" />
                        Diagnostic Likelihood
                    </h3>

                    {allPredictions.length > 0 ? (
                        <div className="grid grid-cols-2 gap-3">
                            {allPredictions.map((pred, idx) => (
                                <AnalysisCard
                                    key={idx}
                                    index={idx}
                                    label={pred.label}
                                    probability={pred.probability}
                                    confidence={pred.confidence}
                                    isUncertain={uncertain_findings?.some(u => u.label === pred.label)}
                                />
                            ))}
                        </div>
                    ) : (
                        <div className="bg-background/50 rounded-2xl p-8 border border-border text-center">
                            <Sparkles size={20} className="text-success mx-auto mb-3 opacity-60" />
                            <p className="text-[10px] text-slate-600 font-black uppercase tracking-widest">Baseline Normal</p>
                        </div>
                    )}
                </div>

                {/* Synthesis Sections */}
                <div className="space-y-6 pt-4 border-t border-border/50">
                    <ReportSection title="Context" content={report.clinical_indication} icon={Activity} accentColor="accent" />
                    <ReportSection title="Findings" content={findings} icon={Brain} accentColor="primary" />
                    <ReportSection title="Impression" content={impression} icon={CheckCircle} accentColor="success" />

                    {warnings?.length > 0 && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="p-5 bg-danger/5 border border-danger/10 rounded-2xl"
                        >
                            <h4 className="text-[10px] font-black text-danger uppercase mb-3 flex items-center gap-2 tracking-[0.2em]">
                                <AlertTriangle size={14} />
                                Safeguard Alerts
                            </h4>
                            <ul className="space-y-2">
                                {warnings.map((w, i) => (
                                    <li key={i} className="text-[11px] text-danger/80 flex items-start gap-2">
                                        <span className="mt-1.5 w-1 h-1 rounded-full bg-danger opacity-40 shrink-0" />
                                        {w}
                                    </li>
                                ))}
                            </ul>
                        </motion.div>
                    )}
                </div>

                {/* Verification Badge */}
                <div className="pt-6 border-t border-border/50">
                    <div className="flex items-center gap-3 mb-3">
                        <div className="p-2 bg-primary-500/10 rounded-lg">
                            <ShieldCheck size={16} className="text-primary-400" />
                        </div>
                        <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest">System Signature</h4>
                    </div>
                    <p className="text-[10px] text-slate-600 leading-relaxed italic">
                        Verified by <span className="text-primary-400 font-bold">CogniRad++ v1.0</span>. This AI output requires expert human validation before clinical integration.
                    </p>
                </div>
            </div>
        </motion.div>
    );
};

export default ReportDisplay;
