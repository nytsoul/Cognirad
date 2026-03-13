import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Upload, Cloud, FileText, CheckCircle, AlertCircle, Plus,
  X, FileImage, Info
} from 'lucide-react';
import { motion } from 'framer-motion';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';

function UploadScan() {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [patientAge, setPatientAge] = useState('');
  const [gender, setGender] = useState('');
  const [indication, setIndication] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setError(null);
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('indication', indication);
      // Optional: tweak this threshold in the UI if needed
      formData.append('confidence_threshold', '0.7');

      const res = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorBody = await res.json().catch(() => ({}));
        throw new Error(errorBody.error || 'Analysis failed');
      }

      const report = await res.json();
      const imageUrl = URL.createObjectURL(file);

      navigate(`/analysis/CASE${Date.now()}`, {
        state: { report, imageUrl },
      });
    } catch (err) {
      console.error('Analysis failed', err);
      setError(err?.message || 'Analysis failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Page Header */}
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-4xl font-black text-white mb-2">Upload Chest X-Ray Scan</h1>
        <p className="text-slate-400 text-sm">Power your diagnostics with explainable AI. Our system identifies common pathologies and provides visual heatmaps for clinical validation.</p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Upload Area */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="lg:col-span-2"
        >
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-3xl p-12 text-center transition-all ${
              dragActive
                ? 'border-primary-500 bg-primary-500/5'
                : 'border-white/20 bg-white/[0.05] hover:border-primary-500/50'
            }`}
          >
            <div className="mb-6">
              <div className="w-20 h-20 bg-primary-500/10 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <Cloud size={32} className="text-primary-400" />
              </div>
              <h3 className="text-2xl font-bold text-white mb-2">
                Drag & drop X-ray scan here
              </h3>
              <p className="text-slate-400 text-sm">or browse your computer to select a file</p>
            </div>

            <div className="flex items-center justify-center gap-2 mb-6">
              {['JPG', 'PNG', 'DICOM'].map((fmt) => (
                <span key={fmt} className="px-3 py-1 bg-white/5 border border-white/10 rounded-full text-xs font-bold text-slate-400">
                  {fmt}
                </span>
              ))}
            </div>

            <div className="flex items-center justify-center gap-3">
              <input
                type="file"
                id="fileInput"
                onChange={handleChange}
                accept="image/*,.dcm"
                className="hidden"
              />
              <label
                htmlFor="fileInput"
                className="px-8 py-3 bg-primary-600 text-white rounded-xl text-sm font-bold hover:bg-primary-500 transition-all cursor-pointer inline-block"
              >
                Browse Files
              </label>
            </div>

            <p className="text-[11px] text-slate-600 mt-6 font-medium">
              Maximum file size: 50MB per scan
            </p>
          </div>

          {/* File Preview */}
          {file && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-6 bg-surface border border-white/10 rounded-2xl p-6 flex items-center justify-between"
            >
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 bg-primary-500/10 rounded-xl flex items-center justify-center">
                  <FileImage size={24} className="text-primary-400" />
                </div>
                <div>
                  <h4 className="font-bold text-white text-sm">{file.name}</h4>
                  <p className="text-xs text-slate-600 font-medium">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              </div>
              <button
                onClick={() => setFile(null)}
                className="p-2 hover:bg-white/5 rounded-lg transition-colors"
              >
                <X size={18} className="text-slate-400" />
              </button>
            </motion.div>
          )}
        </motion.div>

        {/* Clinical Information */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="bg-surface border border-white/10 rounded-2xl p-8 h-fit sticky top-28"
        >
          <div className="flex items-center gap-3 mb-6 pb-4 border-b border-white/10">
            <Info size={16} className="text-primary-400" />
            <h3 className="text-sm font-bold text-white uppercase tracking-widest">Clinical Information</h3>
          </div>

          <div className="space-y-6">
            <div>
              <label className="block text-[10px] font-black text-slate-600 uppercase tracking-widest mb-2">
                Patient Age
              </label>
              <input
                type="text"
                placeholder="e.g. 55"
                value={patientAge}
                onChange={(e) => setPatientAge(e.target.value)}
                className="w-full px-4 py-3 bg-background border border-white/10 rounded-xl focus:ring-2 focus:ring-primary-500 outline-none transition-all text-sm font-medium"
              />
            </div>

            <div>
              <label className="block text-[10px] font-black text-slate-600 uppercase tracking-widest mb-2">
                Gender
              </label>
              <select
                value={gender}
                onChange={(e) => setGender(e.target.value)}
                className="w-full px-4 py-3 bg-background border border-white/10 rounded-xl focus:ring-2 focus:ring-primary-500 outline-none transition-all text-sm font-medium"
              >
                <option>Select gender</option>
                <option>Male</option>
                <option>Female</option>
                <option>Other</option>
              </select>
            </div>

            <div>
              <label className="block text-[10px] font-black text-slate-600 uppercase tracking-widest mb-2">
                Clinical Indication
              </label>
              <textarea
                placeholder="e.g. 55-year-old male with persistent fever and cough for 3 days. History of smoking."
                value={indication}
                onChange={(e) => setIndication(e.target.value)}
                className="w-full px-4 py-3 bg-background border border-white/10 rounded-xl focus:ring-2 focus:ring-primary-500 outline-none transition-all text-sm font-medium h-32 resize-none"
              />
            </div>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleAnalyze}
              disabled={!file || isLoading}
              className={`w-full py-3 rounded-xl text-sm font-black uppercase tracking-widest transition-all flex items-center justify-center gap-2 ${
                file && !isLoading
                  ? 'bg-primary-600 text-white hover:bg-primary-500 shadow-glow'
                  : 'bg-white/5 text-slate-600 cursor-not-allowed'
              }`}
            >
              {isLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <FileText size={16} />
                  Generate AI Report
                </>
              )}
            </motion.button>

            {error && (
              <div className="mt-3 text-sm text-red-300 bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-3">
                <p className="font-semibold">Analysis failed</p>
                <p className="text-xs">{error}</p>
              </div>
            )}

            <div className="text-[10px] text-slate-600 font-medium bg-white/[0.02] border border-white/5 rounded-lg p-3">
              <p className="font-bold text-slate-500 mb-1">Upload Guidelines:</p>
              <ul className="space-y-1 list-disc list-inside">
                <li>Ensure the chest X-ray is in PA or AP view</li>
                <li>High-contrast DICOM files yield the most accurate results</li>
                <li>Avoid cropping edges of the lungs</li>
              </ul>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

export default UploadScan;
