import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ZoomIn, ZoomOut, RotateCcw, Settings, Download, Share2, Eye,
  AlertCircle, CheckCircle, TrendingUp, Layers, Microscope
} from 'lucide-react';
import { motion } from 'framer-motion';

function Analysis() {
  const { caseId } = useParams();
  const navigate = useNavigate();
  const [zoomLevel, setZoomLevel] = useState(100);
  const [activeTab, setActiveTab] = useState('findings');
  const [selectedView, setSelectedView] = useState('original');

  // Mock data
  const predictions = [
    { name: 'Pneumonia', confidence: 87, severity: 'HIGH', evidence: 'Right lower lobe consolidation' },
    { name: 'Lung Opacity', confidence: 78, severity: 'MEDIUM', evidence: 'Diffuse patchy infiltrates' },
    { name: 'Pleural Effusion', confidence: 22, severity: 'LOW', evidence: 'Small bilateral effusions' },
    { name: 'Pneumothorax', confidence: 0, severity: 'NONE', evidence: 'Not detected' },
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Breadcrumb */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex items-center gap-2 text-xs font-bold text-secondary uppercase tracking-widest"
      >
        <span>PATIENTS</span>
        <span className="text-secondary">&gt;</span>
        <span className="text-tertiary">CASE #{caseId}</span>
        <span className="text-secondary">&gt;</span>
        <span className="text-medical">AI DIAGNOSTIC VIEWER</span>
      </motion.div>

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-start justify-between gap-6"
      >
        <div>
          <h1 className="text-3xl font-black text-primary mb-1">Chest X-Ray Analysis (PA View)</h1>
          <div className="flex items-center gap-4 text-sm font-bold text-secondary mt-3">
            <span>PATIENT: JOHN DOE</span>
            <span className="text-border">|</span>
            <span>ID: 99283-A</span>
            <span className="text-border">|</span>
            <span>OCT 24, 2023 14:22 PM</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button className="px-4 py-3 bg-surface-secondary hover:bg-surface-tertiary border border-border rounded-xl text-sm font-bold flex items-center gap-2 transition-all">
            <Download size={16} />
            Export DICOM
          </button>
          <button className="px-6 py-3 bg-primary-600 hover:bg-primary-500 text-white rounded-xl text-sm font-bold flex items-center gap-2 transition-all shadow-glow">
            <CheckCircle size={16} />
            Finalize Report
          </button>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 items-start">
        {/* Image Viewer */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="xl:col-span-2"
        >
          <div className="bg-surface border border-border rounded-2xl overflow-hidden">
            {/* Toolbar */}
            <div className="bg-surface-secondary border-b border-border px-4 py-3 flex items-center justify-between">
              <div className="hidden md:flex items-center gap-2">
                <button className="p-2 hover:bg-surface-tertiary rounded-lg transition-colors text-secondary">
                  <ZoomIn size={16} />
                </button>
                <div className="text-xs text-secondary font-bold w-20 text-right">{zoomLevel}%</div>
                <button className="p-2 hover:bg-surface-tertiary rounded-lg transition-colors text-secondary">
                  <ZoomOut size={16} />
                </button>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => setSelectedView('original')}
                  className={`px-3 py-1 text-xs font-bold rounded transition-all ${
                    selectedView === 'original'
                      ? 'bg-primary-600 text-white'
                      : 'bg-white/5 text-slate-400 hover:bg-white/10'
                  }`}
                >
                  ORIGINAL
                </button>
                <button
                  onClick={() => setSelectedView('heatmap')}
                  className={`px-3 py-1 text-xs font-bold rounded transition-all ${
                    selectedView === 'heatmap'
                      ? 'bg-primary-600 text-white'
                      : 'bg-white/5 text-slate-400 hover:bg-white/10'
                  }`}
                >
                  HEATMAP
                </button>
                <button
                  onClick={() => setSelectedView('segmentation')}
                  className={`px-3 py-1 text-xs font-bold rounded transition-all ${
                    selectedView === 'segmentation'
                      ? 'bg-primary-600 text-white'
                      : 'bg-white/5 text-slate-400 hover:bg-white/10'
                  }`}
                >
                  SEGMENTATION
                </button>
              </div>

              <div className="hidden md:flex items-center gap-2">
                <button className="p-2 hover:bg-white/10 rounded-lg transition-colors text-slate-400">
                  <RotateCcw size={16} />
                </button>
                <button className="p-2 hover:bg-white/10 rounded-lg transition-colors text-slate-400">
                  <Settings size={16} />
                </button>
              </div>
            </div>

            {/* Image Area */}
            <div className="bg-black aspect-square flex items-center justify-center relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-br from-primary-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
              
              <div className="relative w-full h-full flex items-center justify-center">
                <div className="w-4/5 h-4/5 bg-gradient-to-br from-slate-700 to-slate-800 rounded-lg flex items-center justify-center border border-white/10">
                  <div className="text-center">
                    <Eye size={48} className="text-slate-600 mx-auto mb-3" />
                    <p className="text-sm text-slate-600">X-Ray Image Preview</p>
                  </div>
                </div>
              </div>

              {/* Annotations */}
              <div className="absolute bottom-6 left-6 text-xs text-slate-500 font-bold bg-black/50 px-2 py-1 rounded">
                1024 x 942 | WINDOW: 400 LEVEL: 40
              </div>

              {/* AI Label */}
              <div className="absolute bottom-6 right-6 text-xs text-primary-400 font-bold bg-primary-500/10 border border-primary-500/20 px-3 py-1.5 rounded">
                AI PROCESSED 100%
              </div>
            </div>
          </div>
        </motion.div>

        {/* Disease Prediction Panel */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="space-y-4"
        >
          <div className="bg-surface border border-white/10 rounded-2xl overflow-hidden">
            <div className="bg-white/[0.05] border-b border-white/10 px-6 py-4 flex items-center justify-between">
              <h3 className="text-xs font-black uppercase tracking-[0.2em] text-slate-500 flex items-center gap-2">
                <Microscope size={14} className="text-primary-400" />
                Disease Prediction
              </h3>
              <span className="text-xs font-black text-primary-400">EXPLAINABLE AI</span>
            </div>

            <div className="px-6 py-4 space-y-3 max-h-[600px] overflow-y-auto">
              {predictions.map((pred, i) => (
                <motion.div
                  key={pred.name}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.25 + i * 0.05 }}
                  className="group p-4 border border-white/5 rounded-xl hover:border-primary-500/30 transition-all cursor-pointer bg-white/[0.02]"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <h4 className="font-bold text-white text-sm">{pred.name}</h4>
                      <p className="text-[10px] text-slate-600 mt-1 font-medium">{pred.evidence}</p>
                    </div>
                    <div className={`text-right ${
                      pred.confidence > 70 ? 'text-danger' :
                      pred.confidence > 40 ? 'text-accent' :
                      'text-slate-600'
                    }`}>
                      <div className="text-sm font-black">{pred.confidence}%</div>
                      <span className={`text-[8px] font-bold uppercase px-2 py-0.5 rounded mt-1 inline-block ${
                        pred.confidence > 70 ? 'bg-danger/10 text-danger' :
                        pred.confidence > 40 ? 'bg-accent/10 text-accent' :
                        'bg-white/5 text-slate-600'
                      }`}>
                        {pred.severity}
                      </span>
                    </div>
                  </div>
                  <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        pred.confidence > 70 ? 'bg-danger' :
                        pred.confidence > 40 ? 'bg-accent' :
                        'bg-slate-600'
                      }`}
                      style={{ width: `${pred.confidence}%` }}
                    />
                  </div>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Model Confidence */}
          <div className="bg-gradient-to-br from-primary-600/10 to-primary-600/5 border border-primary-500/20 rounded-2xl p-6">
            <h4 className="font-black text-white mb-4 text-sm uppercase tracking-widest">Model Confidence</h4>
            <div className="space-y-3">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-slate-400">Sensitivity</span>
                  <span className="text-sm font-black text-white">94%</span>
                </div>
                <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                  <div className="h-full w-[94%] bg-primary-400 rounded-full" />
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-slate-400">Specificity</span>
                  <span className="text-sm font-black text-white">89%</span>
                </div>
                <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                  <div className="h-full w-[89%] bg-primary-400 rounded-full" />
                </div>
              </div>
            </div>
            <p className="text-[10px] text-slate-600 mt-4 font-medium">
              Tested against 1,284 labeled cases
            </p>
          </div>
        </motion.div>
      </div>

      {/* Navigation Buttons */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="flex items-center gap-4 justify-end pt-4 border-t border-white/10"
      >
        <button
          onClick={() => navigate('/patient-reports')}
          className="px-6 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-sm font-bold transition-all"
        >
          Back to Reports
        </button>
        <button
          onClick={() => navigate(`/evidence/${caseId}`)}
          className="px-6 py-3 bg-primary-600 hover:bg-primary-500 text-white rounded-xl text-sm font-bold transition-all shadow-glow"
        >
          View Evidence Map →
        </button>
      </motion.div>
    </div>
  );
}

export default Analysis;
