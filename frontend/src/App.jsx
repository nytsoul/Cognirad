import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Brain, FileText, ChevronRight, Github, Activity, CheckCircle, Upload, Shield, Database, Microscope } from 'lucide-react';
import UploadZone from './components/UploadZone';
import ReportDisplay from './components/ReportDisplay';
import ImageViewer from './components/ImageViewer';
import CognitivePipeline from './components/CognitivePipeline';
import ReasoningFlow from './components/ReasoningFlow';
import ClinicianEditor from './components/ClinicianEditor';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';

const COGNITIVE_STAGES = [
  { id: 'PRO-FA', name: 'Hierarchical Perception', icon: Microscope },
  { id: 'MIX-MLP', name: 'Disease Classification', icon: FileText },
  { id: 'RCTA', name: 'Triangular Reasoning', icon: Activity },
  { id: 'REPORT', name: 'Final Analysis', icon: Shield }
];

function App() {
  const [file, setFile] = useState(null);
  const [report, setReport] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [indication, setIndication] = useState('');
  const [imagePreview, setImagePreview] = useState(null);
  const [activeStage, setActiveStage] = useState(0);

  useEffect(() => {
    // Add font from Google Fonts
    const link = document.createElement('link');
    link.href = 'https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap';
    link.rel = 'stylesheet';
    document.head.appendChild(link);
  }, []);

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    if (selectedFile) {
      setReport(null);
      setActiveStage(0);
      const reader = new FileReader();
      reader.onloadend = () => setImagePreview(reader.result);
      reader.readAsDataURL(selectedFile);
    } else {
      setImagePreview(null);
    }
  };

  const generateReport = async () => {
    if (!file) return;

    setIsLoading(true);
    // Simulate cognitive steps for UI effect
    for (let i = 0; i < 3; i++) {
      setActiveStage(i);
      await new Promise(r => setTimeout(r, 1200));
    }

    const formData = new FormData();
    formData.append('image', file);
    formData.append('indication', indication || "Chest X-ray");

    try {
      const response = await axios.post('http://localhost:5001/api/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setReport(response.data);
      setActiveStage(3);
    } catch (error) {
      console.error("Error analyzing image:", error);
      alert("Failed to connect to CogniRad Backend.");
      setActiveStage(0);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRegenerate = async () => {
    setIsLoading(true);
    setActiveStage(2);
    await new Promise(r => setTimeout(r, 1500));
    await generateReport();
  };

  return (
    <div className="min-h-screen bg-background bg-gradient-medical text-slate-100 pb-20 overflow-x-hidden">

      {/* Premium Header */}
      <header className="sticky top-0 z-50 glass border-b border-white/5 h-20">
        <div className="max-w-[1700px] mx-auto px-6 h-full flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-11 h-11 bg-primary-600 rounded-xl flex items-center justify-center text-white shadow-glow relative overflow-hidden group">
              <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300" />
              <Brain size={24} className="relative z-10" />
            </div>
            <div>
              <h1 className="text-2xl font-extrabold tracking-tight">
                CogniRad<span className="text-primary-500">++</span>
              </h1>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Diagnostic Station</span>
                <span className="w-1.5 h-1.5 rounded-full bg-success shadow-[0_0_8px_#10b981]" />
              </div>
            </div>
          </div>

          <div className="flex-1 max-w-2xl mx-12 hidden lg:block">
            {report && <CognitivePipeline currentStage={activeStage} stages={COGNITIVE_STAGES} />}
          </div>

          <div className="flex items-center gap-6">
            <div className="hidden xl:flex flex-col items-end border-r border-white/10 pr-6 mr-2">
              <span className="text-xs font-bold text-slate-400">Environment</span>
              <span className="text-xs font-medium text-primary-400">Clinical v2.4-SaaS</span>
            </div>
            <button className="p-2.5 bg-white/5 hover:bg-white/10 rounded-full transition-colors border border-white/5">
              <Database size={18} className="text-slate-400" />
            </button>
            <a href="#" className="p-2.5 bg-white/5 hover:bg-white/10 rounded-full transition-colors border border-white/5">
              <Github size={18} className="text-slate-400" />
            </a>
          </div>
        </div>
      </header>

      <main className="max-w-[1700px] mx-auto px-6 py-8">

        <AnimatePresence mode="wait">
          {!report && !isLoading ? (
            <motion.div
              key="landing"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="flex flex-col items-center justify-center min-h-[70vh] text-center"
            >
              <div className="p-4 rounded-3xl bg-primary-500/10 border border-primary-500/20 mb-8 backdrop-blur-sm">
                <Microscope size={48} className="text-primary-500" />
              </div>
              <h2 className="text-7xl font-black tracking-tighter mb-6 bg-clip-text text-transparent bg-gradient-to-b from-white to-slate-500">
                The Future of <br /> Cognitive Radiology.
              </h2>
              <p className="text-xl text-slate-400 max-w-2xl leading-relaxed mb-12">
                Upload high-resolution DICOM or radiograph images to activate the multi-stage neural perception and triangular reasoning pipeline.
              </p>

              <div className="w-full max-w-xl space-y-6">
                <UploadZone onFileSelect={handleFileSelect} isProcessing={isLoading} />

                <AnimatePresence>
                  {file && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 10 }}
                      className="flex justify-center"
                    >
                      <button
                        onClick={generateReport}
                        disabled={isLoading}
                        className="group relative px-12 py-4 bg-primary-600 text-white rounded-2xl text-lg font-black hover:bg-primary-500 transition-all shadow-glow flex items-center gap-4 overflow-hidden"
                      >
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
                        {isLoading ? <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <Microscope size={24} />}
                        {isLoading ? "Analyzing Neural Pathways..." : "Process Case Data"}
                        <ChevronRight size={20} className="group-hover:translate-x-1 transition-transform" />
                      </button>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              <div className="mt-20 grid grid-cols-2 lg:grid-cols-4 gap-8 w-full max-w-5xl opacity-40">
                {COGNITIVE_STAGES.map(s => (
                  <div key={s.id} className="flex flex-col items-center p-6 border border-white/5 rounded-2xl bg-white/5">
                    <s.icon size={24} className="mb-3" />
                    <span className="text-[10px] font-black uppercase tracking-widest">{s.id}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="grid grid-cols-1 xl:grid-cols-12 gap-8 items-start"
            >
              {/* Left Pane: Perception & Evidence */}
              <div className="xl:col-span-5 space-y-6 lg:sticky lg:top-28">
                <div className="card-premium p-1 relative overflow-hidden group">
                  <ImageViewer
                    imageUrl={imagePreview}
                    isLoading={isLoading}
                    perceptionLayers={report?.perception_layers}
                    attentionMap={report?.attention_maps?.lungs}
                  />
                </div>

                <div className="card-premium p-6">
                  <h3 className="text-xs uppercase tracking-[0.2em] font-black text-slate-500 mb-6 flex items-center gap-3">
                    <Activity size={16} className="text-primary-500" />
                    02. Clinical Workload
                  </h3>

                  <div className="space-y-4">
                    <div>
                      <label className="block text-[10px] font-black text-slate-600 uppercase tracking-widest mb-3">Patient Presentation</label>
                      <textarea
                        className="w-full px-5 py-4 bg-background border border-border rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent text-sm h-32 resize-none transition-all placeholder:text-slate-700"
                        placeholder="Enter clinical indication, symptoms, and history..."
                        value={indication}
                        onChange={(e) => setIndication(e.target.value)}
                        disabled={isLoading}
                      />
                    </div>

                    <div className="flex gap-4">
                      <button
                        onClick={() => handleFileSelect(null)}
                        className="px-6 py-3 bg-surface-lighter hover:bg-slate-800 text-slate-300 rounded-xl text-sm font-bold transition-all border border-white/5 flex-1"
                        disabled={isLoading}
                      >
                        Drop Case
                      </button>
                      <button
                        onClick={generateReport}
                        disabled={isLoading}
                        className={clsx(
                          "px-8 py-3 bg-primary-600 text-white rounded-xl text-sm font-black hover:bg-primary-500 transition-all shadow-glow flex-[2] flex items-center justify-center gap-3",
                          isLoading && "opacity-50 pointer-events-none"
                        )}
                      >
                        {isLoading ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <Microscope size={18} />}
                        {isLoading ? "Running Pipeline..." : "Initiate Analysis"}
                      </button>
                    </div>
                  </div>
                </div>

                {report && (
                  <div className="animate-in fade-in slide-in-from-bottom-4 duration-700">
                    <ClinicianEditor
                      predictions={report.predicted_diseases}
                      isRegenerating={isLoading}
                      onRegenerate={handleRegenerate}
                    />
                  </div>
                )}
              </div>

              {/* Right Pane: Reasoning Engine & Reporting */}
              <div className="xl:col-span-7 space-y-6">
                <div className="grid grid-cols-1 gap-6">
                  <div className="card-premium h-fit overflow-hidden">
                    <div className="p-4 bg-white/5 border-b border-border flex items-center justify-between">
                      <span className="text-[10px] font-black uppercase tracking-wider text-slate-500">Neural Reasoning Engine</span>
                      <div className="flex items-center gap-1.5 text-[10px] text-primary-400 font-bold">
                        <Shield size={12} />
                        RCTA SECURE
                      </div>
                    </div>
                    <ReasoningFlow activeStage={isLoading ? activeStage : 3} />
                  </div>

                  <AnimatePresence mode="wait">
                    {isLoading && activeStage < 3 ? (
                      <motion.div
                        key="loading-state"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="card-premium p-16 flex flex-col items-center justify-center text-center min-h-[400px] relative overflow-hidden"
                      >
                        <div className="absolute inset-0 bg-primary-500/5 animate-pulse-slow pointer-events-none" />
                        <div className="relative mb-8">
                          <div className="w-20 h-20 border-3 border-primary-500/20 border-t-primary-500 rounded-full animate-spin" />
                          <div className="absolute inset-0 flex items-center justify-center">
                            <Brain size={24} className="text-primary-500" />
                          </div>
                        </div>
                        <h4 className="text-2xl font-black text-white mb-3">
                          {COGNITIVE_STAGES[activeStage].id} <span className="text-slate-500 tracking-normal font-medium">Stage Active</span>
                        </h4>
                        <p className="text-slate-400 max-w-sm leading-relaxed">
                          {COGNITIVE_STAGES[activeStage].name} is processing visual features and clinical co-occurrence patterns.
                        </p>
                      </motion.div>
                    ) : report ? (
                      <motion.div
                        key="report-state"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                      >
                        <ReportDisplay report={report} />
                      </motion.div>
                    ) : null}
                  </AnimatePresence>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

      </main>
    </div>
  );
}

export default App;
