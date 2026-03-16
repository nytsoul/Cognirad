import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Share2, Download, Eye, Sliders, BarChart3, Brain, Zap
} from 'lucide-react';
import { motion } from 'framer-motion';

function EvidenceMap() {
  const { caseId } = useParams();
  const navigate = useNavigate();
  const [activeStep, setActiveStep] = useState(0);
  const [viewMode, setViewMode] = useState('heatmap');

  const reasoningSteps = [
    {
      title: 'Visual Feature Extraction',
      description: 'Deep CNN identified patchy consolidations in the left lower lobe (Confidence: 94%)',
      icon: Eye,
    },
    {
      title: 'Disease Prediction',
      description: 'Classified as Bacterial Pneumonia vs "Normal" using multi-label attention heads',
      icon: Brain,
    },
    {
      title: 'Hypothesis Verification',
      description: 'Counterfactual analysis confirms: removing region LDL would drop pneumonia probability to 12%',
      icon: Zap,
    },
    {
      title: 'Report Generation',
      description: 'Natural language summary generated based on identified visual anchors',
      icon: BarChart3,
    },
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Breadcrumb */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex items-center gap-2 text-xs font-bold text-slate-600 uppercase tracking-widest"
      >
        <span>PATIENTS</span>
        <span className="text-slate-600">&gt;</span>
        <span>CASE #{caseId}</span>
        <span className="text-slate-600">&gt;</span>
        <span className="text-primary-400">XAI EVIDENCE MAP</span>
      </motion.div>

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-start justify-between gap-6"
      >
        <div>
          <h1 className="text-3xl font-black text-white mb-2">Explainable AI Evidence Map</h1>
          <p className="text-slate-500 text-sm font-medium">
            <span className="text-danger">HIGH CONFIDENCE ANALYSIS</span> • Patient ID: CXR-2024-8492 | Diagnosis: Pneumonia (89.4%)
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button className="px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-sm font-bold flex items-center gap-2 transition-all">
            <Share2 size={16} />
            Share Findings
          </button>
          <button className="px-6 py-3 bg-primary-600 hover:bg-primary-500 text-white rounded-xl text-sm font-bold flex items-center gap-2 transition-all shadow-glow">
            <Download size={16} />
            Download Report
          </button>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Heatmap Visualization */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="lg:col-span-2"
        >
          <div className="bg-surface border border-white/10 rounded-2xl overflow-hidden">
            {/* Controls */}
            <div className="bg-black/50 border-b border-white/10 px-6 py-4 flex items-center justify-between">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Eye size={16} className="text-primary-400" />
                Attention Heatmap Analysis
              </h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setViewMode('heatmap')}
                  className={`px-3 py-1 text-xs font-bold rounded transition-all ${
                    viewMode === 'heatmap'
                      ? 'bg-primary-600 text-white'
                      : 'bg-white/5 text-slate-400 hover:bg-white/10'
                  }`}
                >
                  Heatmap
                </button>
                <button
                  onClick={() => setViewMode('original')}
                  className={`px-3 py-1 text-xs font-bold rounded transition-all ${
                    viewMode === 'original'
                      ? 'bg-primary-600 text-white'
                      : 'bg-white/5 text-slate-400 hover:bg-white/10'
                  }`}
                >
                  Original
                </button>
              </div>
            </div>

            {/* Heatmap Image */}
            <div className="bg-black aspect-square flex items-center justify-center relative overflow-hidden">
              <div className="w-4/5 h-4/5 bg-gradient-to-br from-slate-800 via-slate-900 to-black rounded-lg flex items-center justify-center border border-white/10 relative">
                <div className="absolute inset-0 bg-gradient-radial from-red-500/30 via-yellow-500/10 to-transparent rounded-lg" />
                <Eye size={48} className="text-slate-600 relative z-10" />
              </div>

              {/* Legend */}
              <div className="absolute bottom-6 left-6 flex items-center gap-2">
                <div className="text-xs text-slate-400 font-bold">Low</div>
                <div className="flex items-center gap-1">
                  {['#0f172a', '#1e40af', '#0ea5e9', '#fbbf24', '#ef4444'].map((color, i) => (
                    <div
                      key={i}
                      className="w-4 h-4 rounded"
                      style={{ backgroundColor: color }}
                    />
                  ))}
                </div>
                <div className="text-xs text-slate-400 font-bold">High Focus</div>
              </div>

              {/* AI Badge */}
              <div className="absolute bottom-6 right-6 text-xs text-primary-300 font-bold bg-primary-500/20 border border-primary-400/30 px-3 py-1.5 rounded-full">
                Sub-segmental opacity detected! Right lower lobe. Pixel density gradient in the right paracardial region.
              </div>
            </div>
          </div>
        </motion.div>

        {/* Model Reasoning Workflow */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="space-y-4"
        >
          <div className="bg-surface border border-white/10 rounded-2xl p-6">
            <h3 className="text-xs font-bold text-white uppercase tracking-widest mb-6 flex items-center gap-2">
              <Brain size={14} className="text-primary-400" />
              Model Reasoning Workflow
            </h3>

            <div className="space-y-3">
              {reasoningSteps.map((step, i) => (
                <motion.div
                  key={step.title}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.2 + i * 0.08 }}
                  onClick={() => setActiveStep(i)}
                  className={`group p-4 rounded-xl border transition-all cursor-pointer ${
                    activeStep === i
                      ? 'bg-primary-500/10 border-primary-500/30'
                      : 'bg-white/[0.02] border-white/10 hover:border-primary-500/20'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div className={`p-2 rounded-lg shrink-0 ${
                      activeStep === i
                        ? 'bg-primary-500/20'
                        : 'bg-white/5'
                    }`}>
                      <step.icon size={16} className={activeStep === i ? 'text-primary-400' : 'text-slate-600'} />
                    </div>
                    <div>
                      <div className={`text-xs font-bold uppercase tracking-widest ${
                        activeStep === i ? 'text-primary-400' : 'text-slate-600'
                      }`}>
                        Step {i + 1}
                      </div>
                      <h4 className="text-sm font-bold text-white mt-1">{step.title}</h4>
                      <p className="text-[11px] text-slate-500 mt-1.5 font-medium leading-relaxed">{step.description}</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Dominant Features */}
          <div className="bg-surface border border-white/10 rounded-2xl p-6">
            <h4 className="text-xs font-bold text-white uppercase tracking-widest mb-4">Dominant Visual Features</h4>
            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Alveolar opacities</span>
                <span className="text-primary-400 font-bold">0.92 weight</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Pleural effusion</span>
                <span className="text-accent font-bold">0.14 weight</span>
              </div>
            </div>
          </div>

          {/* System Info */}
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-800/20 border border-white/10 rounded-2xl p-4 text-xs">
            <p className="text-slate-500 font-medium">
              <span className="text-primary-400 font-bold">ℹ System Transparency Note</span><br/>
              CogniRad++ uses integrated gradients and layer-wise relevance propagation to visualize decision-making. Highlighted regions represent the 10% of pixels contributing most to the final classification result.
            </p>
          </div>
        </motion.div>
      </div>

      {/* Region Analysis */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="grid grid-cols-1 md:grid-cols-2 gap-6"
      >
        {/* Dominant Region */}
        <div className="bg-surface border border-white/10 rounded-2xl p-6">
          <h4 className="text-xs font-bold text-white uppercase tracking-widest mb-4">Dominant Region Focus</h4>
          <div className="space-y-4">
            {[
              { region: 'L-LOWER', confidence: 0.94, color: 'primary' },
              { region: 'L-UPPER', confidence: 0.45, color: 'accent' },
              { region: 'R-LOWER', confidence: 0.12, color: 'slate' },
            ].map((item) => (
              <div key={item.region} className="text-sm">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-bold text-white">{item.region}</span>
                  <span className={`font-bold text-${item.color}-400`}>{(item.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                  <div
                    className={`h-full bg-${item.color}-500 rounded-full`}
                    style={{ width: `${item.confidence * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Report AI Bias */}
        <div className="bg-gradient-to-br from-accent/10 to-accent/5 border border-accent/20 rounded-2xl p-6">
          <h4 className="text-xs font-bold text-accent uppercase tracking-widest mb-4">Model Bias Assessment</h4>
          <p className="text-xs text-slate-400 font-medium leading-relaxed">
            Model shows balanced sensitivity and specificity across demographic groups (Age, Gender). No significant disparities detected in 50+ unlabeled cases.
          </p>
        </div>
      </motion.div>

      {/* Navigation */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.35 }}
        className="flex items-center gap-4 justify-end pt-4 border-t border-white/10"
      >
        <button
          onClick={() => navigate(`/report/${caseId}`)}
          className="px-6 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-sm font-bold transition-all"
        >
          ← Back to Report
        </button>
        <button
          onClick={() => navigate('/patient-reports')}
          className="px-6 py-3 bg-primary-600 hover:bg-primary-500 text-white rounded-xl text-sm font-bold transition-all shadow-glow"
        >
          Back to Patient Reports
        </button>
      </motion.div>
    </div>
  );
}

export default EvidenceMap;
