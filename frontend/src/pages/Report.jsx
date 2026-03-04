import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Download, Share2, Edit, CheckCircle, Eye, FileText, AlertCircle } from 'lucide-react';
import { motion } from 'framer-motion';

function Report() {
  const { caseId } = useParams();
  const navigate = useNavigate();
  const [isEditing, setIsEditing] = useState(false);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Breadcrumb */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex items-center gap-2 text-xs font-bold text-slate-600 uppercase tracking-widest"
      >
        <span>ARCHIVE</span>
        <span className="text-slate-600">&gt;</span>
        <span className="text-slate-500">CHEST-XRAY</span>
        <span className="text-slate-600">&gt;</span>
        <span className="text-primary-400">REPORT CR-99210</span>
      </motion.div>

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-start justify-between gap-6"
      >
        <div>
          <h1 className="text-3xl font-black text-white mb-2">AI Radiology Report</h1>
          <div className="flex items-center gap-3 text-xs font-bold">
            <span className="text-success">✓ VERIFIED BY AI</span>
            <span className="text-slate-700">|</span>
            <span className="text-slate-500">Patient ID: CR-99210</span>
            <span className="text-slate-700">|</span>
            <span className="text-slate-500">Study Date: Oct 24, 2023</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button className="px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-sm font-bold flex items-center gap-2 transition-all">
            <Download size={16} />
            PDF
          </button>
          <button className="px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-sm font-bold flex items-center gap-2 transition-all">
            <Edit size={16} />
            Edit
          </button>
          <button className="px-6 py-3 bg-primary-600 hover:bg-primary-500 text-white rounded-xl text-sm font-bold flex items-center gap-2 transition-all shadow-glow">
            <Share2 size={16} />
            Share with Clinician
          </button>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Report */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="lg:col-span-2"
        >
          <div className="bg-surface border border-white/10 rounded-2xl overflow-hidden">
            {/* Report Image */}
            <div className="bg-black/50 p-6 border-b border-white/10">
              <div className="bg-black aspect-[4/3] rounded-xl flex items-center justify-center border border-white/10 relative group">
                <div className="absolute inset-0 bg-gradient-to-br from-primary-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity rounded-xl" />
                <div className="w-full h-full flex items-center justify-center">
                  <div className="text-center">
                    <Eye size={32} className="text-slate-600 mx-auto mb-2" />
                    <p className="text-sm text-slate-600">Medical Imaging Analysis</p>
                  </div>
                </div>
                {/* Evidence Badge */}
                <div className="absolute bottom-4 left-4 text-xs text-primary-300 font-bold bg-primary-500/20 border border-primary-400/30 px-3 py-1.5 rounded-full flex items-center gap-1.5">
                  <Eye size={12} />
                  HEM Confidence
                </div>
              </div>
            </div>

            {/* Report Content */}
            <div className="p-8 space-y-8">
              {/* Impression */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.15 }}
                className="bg-primary-500/5 border border-primary-500/20 rounded-2xl p-6"
              >
                <h3 className="text-xs font-black uppercase tracking-[0.2em] text-primary-400 mb-4 flex items-center gap-2">
                  <CheckCircle size={14} />
                  Impression
                </h3>
                <p className="text-sm leading-relaxed text-white font-medium">
                  Right lower lobe pneumonia.
                </p>
                <p className="text-xs text-slate-500 mt-3 font-medium">
                  Clinical correlation is recommended. Follow-up imaging in 4-6 weeks to ensure resolution.
                </p>
              </motion.div>

              {/* Findings */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
              >
                <h3 className="text-xs font-black uppercase tracking-[0.2em] text-slate-400 mb-4 flex items-center gap-2">
                  <FileText size={14} className="text-primary-400" />
                  Findings
                </h3>
                <div className="space-y-4 text-sm text-slate-300 leading-relaxed font-medium">
                  <p>
                    <span className="font-bold text-white">Lungs:</span> The lungs are clear without focal consolidation on diffusion, or pneumothorax. There is a subtle focal opacity in the right lower lobe, which is more apparent on the lateral view.
                  </p>
                  <p>
                    <span className="font-bold text-white">Cardiac:</span> Cardiometiastinal silhouette is within normal limits. The heart size is normal. No evidence of pulmonary vascular congestion.
                  </p>
                  <p>
                    <span className="font-bold text-white">Bony Structures:</span> Intact. No acute fractures or suspicious osseous lesions are identified.
                  </p>
                  <p>
                    <span className="font-bold text-white">Soft Tissues:</span> Unremarkable.
                  </p>
                </div>
              </motion.div>

              {/* Evidence Support */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.25 }}
                className="bg-white/[0.02] border border-white/10 rounded-xl p-4"
              >
                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-1">
                  <AlertCircle size={12} className="text-primary-400" />
                  AI Evidence Support
                </h4>
                <ul className="space-y-2 text-xs text-slate-500 font-medium">
                  <li className="flex items-start gap-2">
                    <span className="text-primary-400 mt-1">•</span>
                    <span>Abnormal pixel density gradient detected in the right paracardial region.</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-primary-400 mt-1">•</span>
                    <span>Silhouette sign present against the right diaphragm interface.</span>
                  </li>
                </ul>
              </motion.div>
            </div>

            {/* Footer */}
            <div className="bg-white/[0.02] border-t border-white/10 px-8 py-4 text-xs text-slate-600 font-medium flex items-center justify-between">
              <div>
                Electronically Signed by <span className="text-white font-bold">Dr. Sarah Chen, MD</span> • Signed at Oct 24, 2023 14:32 PST
              </div>
              <div className="flex items-center gap-4 text-[10px]">
                <a href="#" className="hover:text-slate-400 transition-colors">Report History</a>
                <a href="#" className="hover:text-slate-400 transition-colors">Privacy Policy</a>
                <div className="flex items-center gap-1">
                  <CheckCircle size={12} className="text-success" />
                  <span className="text-success">HIPAA Compliant</span>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Sidebar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="space-y-4"
        >
          {/* Patient Info */}
          <div className="bg-surface border border-white/10 rounded-2xl p-6">
            <h4 className="text-xs font-black uppercase tracking-[0.2em] text-slate-500 mb-4">Patient Information</h4>
            <div className="space-y-3 text-sm">
              <div>
                <span className="text-slate-600 font-bold">Name</span>
                <div className="text-white font-bold mt-1">John Doe</div>
              </div>
              <div>
                <span className="text-slate-600 font-bold">Patient ID</span>
                <div className="text-white font-bold mt-1">CR-99210</div>
              </div>
              <div>
                <span className="text-slate-600 font-bold">Age / Gender</span>
                <div className="text-white font-bold mt-1">55M</div>
              </div>
              <div>
                <span className="text-slate-600 font-bold">Study Date</span>
                <div className="text-white font-bold mt-1">Oct 24, 2023</div>
              </div>
            </div>
          </div>

          {/* AI Validation */}
          <div className="bg-gradient-to-br from-success/10 to-success/5 border border-success/20 rounded-2xl p-6">
            <h4 className="text-xs font-bold text-success uppercase tracking-widest mb-3">AI Validation</h4>
            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Model Name</span>
                <span className="text-white font-bold">CogniRad++ v2.4</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Processing Time</span>
                <span className="text-white font-bold">1.2s</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Confidence Score</span>
                <span className="text-green-400 font-bold">94%</span>
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Navigation */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.35 }}
        className="flex items-center gap-4 justify-end pt-4 border-t border-white/10"
      >
        <button
          onClick={() => navigate(`/analysis/${caseId}`)}
          className="px-6 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-sm font-bold transition-all"
        >
          ← Back to Analysis
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

export default Report;
