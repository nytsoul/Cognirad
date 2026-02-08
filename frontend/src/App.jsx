import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Brain, FileText, ChevronRight, Github, Activity, CheckCircle,
  Upload, Shield, Database, Microscope, Clock, Heart, Zap, BarChart3,
  Globe, Layers, Cpu, ScanEye, Sparkles, ArrowRight, Eye, Workflow,
  MonitorCheck, Lock, BookOpen, Stethoscope
} from 'lucide-react';
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
  const [analysisCount, setAnalysisCount] = useState(() => {
    return parseInt(localStorage.getItem('cognirad_analysis_count') || '0', 10);
  });
  const [analysisTime, setAnalysisTime] = useState(null);

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
    setAnalysisTime(null);
    const startTime = Date.now();

    // Simulate cognitive steps for UI effect
    for (let i = 0; i < 3; i++) {
      setActiveStage(i);
      await new Promise(r => setTimeout(r, 1200));
    }

    const formData = new FormData();
    formData.append('image', file);
    formData.append('indication', indication || "Chest X-ray");

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:5001';
      const response = await axios.post(`${apiUrl}/api/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setReport(response.data);
      setActiveStage(3);
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      setAnalysisTime(elapsed);
      const newCount = analysisCount + 1;
      setAnalysisCount(newCount);
      localStorage.setItem('cognirad_analysis_count', String(newCount));
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

          <div className="flex items-center gap-4">
            <div className="hidden xl:flex items-center gap-3 border-r border-white/10 pr-5 mr-1">
              <div className="flex flex-col items-end">
                <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest">Session</span>
                <span className="text-xs font-bold text-primary-400 tabular-nums">{analysisCount} analyses</span>
              </div>
              {analysisTime && (
                <div className="flex flex-col items-end">
                  <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest">Last Run</span>
                  <span className="text-xs font-bold text-success tabular-nums">{analysisTime}s</span>
                </div>
              )}
            </div>
            <div className="hidden lg:flex items-center gap-1.5 px-3 py-1.5 bg-success/10 border border-success/20 rounded-full">
              <div className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />
              <span className="text-[10px] font-black text-success uppercase tracking-widest">Online</span>
            </div>
            <button className="p-2.5 bg-white/5 hover:bg-white/10 rounded-xl transition-colors border border-white/5">
              <Database size={16} className="text-slate-400" />
            </button>
            <a href="https://github.com" target="_blank" rel="noreferrer" className="p-2.5 bg-white/5 hover:bg-white/10 rounded-xl transition-colors border border-white/5">
              <Github size={16} className="text-slate-400" />
            </a>
          </div>
        </div>
      </header>

      <main className="max-w-[1700px] mx-auto px-6 py-8">

        <AnimatePresence mode="wait">
          {!report && !isLoading ? (
            <motion.div
              key="landing"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, scale: 0.97 }}
              className="relative"
            >
              {/* ═══════ HERO SECTION ═══════ */}
              <section className="relative min-h-[88vh] flex items-center overflow-hidden -mx-6 px-6">
                {/* Animated background grid */}
                <div className="absolute inset-0 -z-10 pointer-events-none overflow-hidden">
                  <div className="absolute inset-0 opacity-[0.03]"
                    style={{ backgroundImage: 'radial-gradient(#0ea5e9 1px, transparent 1px)', backgroundSize: '40px 40px' }} />
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 120, repeat: Infinity, ease: 'linear' }}
                    className="absolute -top-[40%] -right-[20%] w-[800px] h-[800px] rounded-full border border-primary-500/5"
                  />
                  <motion.div
                    animate={{ rotate: -360 }}
                    transition={{ duration: 180, repeat: Infinity, ease: 'linear' }}
                    className="absolute -bottom-[30%] -left-[15%] w-[600px] h-[600px] rounded-full border border-primary-500/[0.03]"
                  />
                  <div className="absolute top-1/4 right-1/4 w-96 h-96 bg-primary-500/[0.04] rounded-full blur-[120px]" />
                  <div className="absolute bottom-1/3 left-1/4 w-72 h-72 bg-primary-600/[0.03] rounded-full blur-[100px]" />
                </div>

                <div className="w-full grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-20 items-center max-w-[1700px] mx-auto py-12">

                  {/* Left: Hero Text */}
                  <div className="space-y-8">
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.1 }}
                      className="inline-flex items-center gap-2.5 px-4 py-2 bg-primary-500/10 border border-primary-500/20 rounded-full"
                    >
                      <div className="w-2 h-2 rounded-full bg-primary-500 animate-pulse shadow-glow" />
                      <span className="text-[11px] font-black text-primary-400 uppercase tracking-[0.15em]">Next-Gen Diagnostic AI</span>
                    </motion.div>

                    <motion.h2
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.2 }}
                      className="text-5xl md:text-6xl xl:text-7xl font-black tracking-tighter leading-[0.95]"
                    >
                      <span className="bg-clip-text text-transparent bg-gradient-to-b from-white via-white to-slate-400">
                        Cognitive
                      </span>
                      <br />
                      <span className="bg-clip-text text-transparent bg-gradient-to-b from-white via-slate-200 to-slate-500">
                        Radiology
                      </span>
                      <br />
                      <span className="bg-clip-text text-transparent bg-gradient-to-r from-primary-400 to-primary-600">
                        Reimagined.
                      </span>
                    </motion.h2>

                    <motion.p
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                      className="text-lg text-slate-400 max-w-lg leading-relaxed"
                    >
                      Three-stage neural pipeline that perceives, classifies, and reasons — generating comprehensive radiology reports with explainable AI evidence.
                    </motion.p>

                    {/* Stats row */}
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.4 }}
                      className="flex gap-8"
                    >
                      {[
                        { value: '14', label: 'Pathology\nClasses' },
                        { value: '3-Stage', label: 'Cognitive\nPipeline' },
                        { value: '<8s', label: 'Inference\nTime' },
                      ].map((stat, i) => (
                        <div key={i} className="text-center">
                          <div className="text-2xl font-black text-white">{stat.value}</div>
                          <div className="text-[9px] font-bold text-slate-600 uppercase tracking-widest whitespace-pre-line mt-1">{stat.label}</div>
                        </div>
                      ))}
                    </motion.div>

                    {/* CTA Buttons */}
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.5 }}
                      className="flex flex-wrap gap-4 pt-2"
                    >
                      <a
                        href="#upload-section"
                        className="group relative px-8 py-4 bg-primary-600 text-white rounded-2xl text-sm font-black hover:bg-primary-500 transition-all shadow-glow flex items-center gap-3 overflow-hidden"
                      >
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
                        <ScanEye size={18} />
                        Start Analysis
                        <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
                      </a>
                      <a
                        href="#how-it-works"
                        className="px-8 py-4 bg-white/5 text-slate-300 rounded-2xl text-sm font-bold hover:bg-white/10 transition-all border border-white/10 flex items-center gap-3"
                      >
                        <BookOpen size={16} />
                        How It Works
                      </a>
                    </motion.div>
                  </div>

                  {/* Right: Architecture Visualization */}
                  <motion.div
                    initial={{ opacity: 0, x: 40 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4, duration: 0.6 }}
                    className="relative hidden lg:block"
                  >
                    <div className="relative">
                      {/* Glow behind card */}
                      <div className="absolute -inset-8 bg-primary-500/[0.06] blur-3xl rounded-full" />

                      {/* Main architecture card */}
                      <div className="relative bg-surface/80 backdrop-blur-xl border border-white/10 rounded-3xl p-8 shadow-glass overflow-hidden">
                        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-primary-500/40 to-transparent" />

                        <div className="flex items-center justify-between mb-8">
                          <div className="flex items-center gap-3">
                            <div className="p-2 bg-primary-500/10 rounded-lg">
                              <Workflow size={16} className="text-primary-400" />
                            </div>
                            <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Architecture Pipeline</span>
                          </div>
                          <div className="flex gap-1.5">
                            <div className="w-2.5 h-2.5 rounded-full bg-danger/60" />
                            <div className="w-2.5 h-2.5 rounded-full bg-accent/60" />
                            <div className="w-2.5 h-2.5 rounded-full bg-success/60" />
                          </div>
                        </div>

                        {/* Pipeline steps */}
                        <div className="space-y-4">
                          {[
                            { icon: Eye, name: 'PRO-FA Encoder', desc: 'Pixel → Region → Organ alignment', color: 'primary', tag: 'PERCEPTION' },
                            { icon: Cpu, name: 'MIX-MLP Classifier', desc: 'Multi-path disease co-occurrence', color: 'accent', tag: 'DIAGNOSIS' },
                            { icon: Brain, name: 'RCTA Decoder', desc: 'Closed-loop triangular reasoning', color: 'success', tag: 'REASONING' },
                          ].map((step, i) => (
                            <motion.div
                              key={step.name}
                              initial={{ opacity: 0, x: 20 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: 0.6 + i * 0.15 }}
                              className="group flex items-center gap-4 p-4 rounded-2xl bg-background/60 border border-white/5 hover:border-primary-500/20 transition-all duration-300"
                            >
                              <div className={clsx(
                                "p-3 rounded-xl shrink-0 transition-colors duration-300",
                                step.color === 'primary' && "bg-primary-500/10 group-hover:bg-primary-500/20",
                                step.color === 'accent' && "bg-accent/10 group-hover:bg-accent/20",
                                step.color === 'success' && "bg-success/10 group-hover:bg-success/20",
                              )}>
                                <step.icon size={20} className={clsx(
                                  step.color === 'primary' && "text-primary-400",
                                  step.color === 'accent' && "text-accent",
                                  step.color === 'success' && "text-success",
                                )} />
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2">
                                  <span className="text-sm font-bold text-white">{step.name}</span>
                                  <span className={clsx(
                                    "text-[8px] font-black px-2 py-0.5 rounded-full uppercase tracking-widest",
                                    step.color === 'primary' && "bg-primary-500/10 text-primary-400",
                                    step.color === 'accent' && "bg-accent/10 text-accent",
                                    step.color === 'success' && "bg-success/10 text-success",
                                  )}>{step.tag}</span>
                                </div>
                                <p className="text-[11px] text-slate-500 mt-0.5 font-medium">{step.desc}</p>
                              </div>
                              <ChevronRight size={14} className="text-slate-700 group-hover:text-slate-400 transition-colors shrink-0" />
                            </motion.div>
                          ))}
                        </div>

                        {/* Output area */}
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: 1.1 }}
                          className="mt-6 p-4 rounded-2xl bg-gradient-to-r from-primary-500/5 to-success/5 border border-primary-500/10"
                        >
                          <div className="flex items-center gap-3">
                            <div className="p-2 bg-white/5 rounded-lg">
                              <FileText size={16} className="text-primary-300" />
                            </div>
                            <div>
                              <span className="text-xs font-bold text-white">Structured Report Output</span>
                              <div className="flex items-center gap-2 mt-0.5">
                                <span className="text-[9px] text-slate-500 font-medium">Findings + Impression + Evidence Maps</span>
                                <div className="flex items-center gap-1">
                                  <div className="w-1 h-1 rounded-full bg-success animate-pulse" />
                                  <span className="text-[8px] font-black text-success uppercase">Ready</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </motion.div>
                      </div>
                    </div>
                  </motion.div>
                </div>
              </section>

              {/* ═══════ HOW IT WORKS ═══════ */}
              <section id="how-it-works" className="py-20 -mx-6 px-6">
                <motion.div
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: "-100px" }}
                  className="text-center mb-16"
                >
                  <span className="text-[10px] font-black text-primary-500 uppercase tracking-[0.2em]">Three-Stage Pipeline</span>
                  <h3 className="text-4xl font-black text-white mt-3 tracking-tight">How CogniRad++ Works</h3>
                  <p className="text-slate-500 mt-3 max-w-lg mx-auto text-sm leading-relaxed">
                    A cognitive reasoning loop inspired by how expert radiologists perceive, diagnose, and verify their findings.
                  </p>
                </motion.div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl mx-auto relative">
                  {/* Connection lines */}
                  <div className="hidden md:block absolute top-1/2 left-0 right-0 h-px -translate-y-1/2 z-0">
                    <div className="mx-[16%] h-px bg-gradient-to-r from-primary-500/20 via-accent/20 to-success/20" />
                  </div>

                  {[
                    {
                      icon: Eye, step: '01', name: 'PRO-FA',
                      title: 'Hierarchical Perception',
                      desc: 'Extracts pixel, region, and organ-level features using factored attention. Aligns visual concepts across multiple scales to capture both local pathology markers and global anatomical context.',
                      color: 'primary'
                    },
                    {
                      icon: Layers, step: '02', name: 'MIX-MLP',
                      title: 'Disease Classification',
                      desc: 'A multi-path architecture combining residual, expansion, and compressed pathways. Models disease co-occurrence patterns to predict 14 CheXpert pathologies with calibrated confidence.',
                      color: 'accent'
                    },
                    {
                      icon: Brain, step: '03', name: 'RCTA',
                      title: 'Triangular Reasoning',
                      desc: 'Closed-loop decoder: Image queries Text, Context queries Diagnosis, Diagnosis queries Image. This triangular attention verifies findings before generating the final clinical report.',
                      color: 'success'
                    },
                  ].map((item, i) => (
                    <motion.div
                      key={item.name}
                      initial={{ opacity: 0, y: 30 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true, margin: "-50px" }}
                      transition={{ delay: i * 0.15 }}
                      className="relative z-10 group"
                    >
                      <div className="h-full p-8 rounded-3xl bg-surface border border-white/5 hover:border-primary-500/20 transition-all duration-500 hover:shadow-glow/20">
                        <div className="flex items-center justify-between mb-6">
                          <div className={clsx(
                            "p-3.5 rounded-2xl transition-colors duration-300",
                            item.color === 'primary' && "bg-primary-500/10 group-hover:bg-primary-500/15",
                            item.color === 'accent' && "bg-accent/10 group-hover:bg-accent/15",
                            item.color === 'success' && "bg-success/10 group-hover:bg-success/15",
                          )}>
                            <item.icon size={24} className={clsx(
                              item.color === 'primary' && "text-primary-400",
                              item.color === 'accent' && "text-accent",
                              item.color === 'success' && "text-success",
                            )} />
                          </div>
                          <span className="text-5xl font-black text-white/[0.04] select-none">{item.step}</span>
                        </div>

                        <div className="flex items-center gap-2 mb-2">
                          <span className={clsx(
                            "text-[9px] font-black px-2.5 py-1 rounded-lg uppercase tracking-widest",
                            item.color === 'primary' && "bg-primary-500/10 text-primary-400",
                            item.color === 'accent' && "bg-accent/10 text-accent",
                            item.color === 'success' && "bg-success/10 text-success",
                          )}>{item.name}</span>
                        </div>

                        <h4 className="text-lg font-bold text-white mb-3">{item.title}</h4>
                        <p className="text-[13px] text-slate-500 leading-relaxed">{item.desc}</p>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </section>

              {/* ═══════ FEATURES STRIP ═══════ */}
              <section className="py-12 -mx-6 px-6 border-y border-white/5 bg-white/[0.01]">
                <div className="max-w-6xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8">
                  {[
                    { icon: Shield, label: 'HIPAA Compliant', desc: 'Privacy-first architecture', color: 'text-primary-400' },
                    { icon: Zap, label: 'Real-time Inference', desc: 'Optimized CPU & GPU', color: 'text-accent' },
                    { icon: MonitorCheck, label: 'Explainable AI', desc: 'Evidence-backed reports', color: 'text-success' },
                    { icon: Lock, label: 'Clinician Override', desc: 'Human-in-the-loop', color: 'text-primary-300' },
                  ].map((feat, i) => (
                    <motion.div
                      key={feat.label}
                      initial={{ opacity: 0, y: 15 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true }}
                      transition={{ delay: i * 0.08 }}
                      className="flex items-start gap-4 group"
                    >
                      <div className="p-2.5 bg-white/5 rounded-xl border border-white/5 group-hover:border-primary-500/20 transition-colors shrink-0">
                        <feat.icon size={18} className={feat.color} />
                      </div>
                      <div>
                        <span className="text-xs font-bold text-white">{feat.label}</span>
                        <p className="text-[11px] text-slate-600 mt-0.5 font-medium">{feat.desc}</p>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </section>

              {/* ═══════ UPLOAD SECTION ═══════ */}
              <section id="upload-section" className="py-20">
                <motion.div
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: "-100px" }}
                  className="text-center mb-10"
                >
                  <span className="text-[10px] font-black text-primary-500 uppercase tracking-[0.2em]">Get Started</span>
                  <h3 className="text-4xl font-black text-white mt-3 tracking-tight">Analyze a Radiograph</h3>
                  <p className="text-slate-500 mt-3 max-w-md mx-auto text-sm leading-relaxed">
                    Upload a chest X-ray image to run the full cognitive reasoning pipeline and generate an automated report.
                  </p>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  className="max-w-xl mx-auto space-y-6"
                >
                  <UploadZone onFileSelect={handleFileSelect} isProcessing={isLoading} />

                  <AnimatePresence>
                    {file && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 10 }}
                        className="flex flex-col items-center gap-4"
                      >
                        <div className="w-full max-w-md">
                          <label className="block text-[10px] font-black text-slate-600 uppercase tracking-widest mb-2">Clinical Indication (optional)</label>
                          <input
                            type="text"
                            className="w-full px-4 py-3 bg-surface border border-border rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent text-sm transition-all placeholder:text-slate-700"
                            placeholder="e.g. 55M with chest pain and shortness of breath"
                            value={indication}
                            onChange={(e) => setIndication(e.target.value)}
                          />
                        </div>
                        <button
                          onClick={generateReport}
                          disabled={isLoading}
                          className="group relative px-12 py-4 bg-primary-600 text-white rounded-2xl text-lg font-black hover:bg-primary-500 transition-all shadow-glow flex items-center gap-4 overflow-hidden"
                        >
                          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
                          {isLoading ? <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <Microscope size={22} />}
                          {isLoading ? "Analyzing Neural Pathways..." : "Run Cognitive Analysis"}
                          <ChevronRight size={18} className="group-hover:translate-x-1 transition-transform" />
                        </button>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              </section>
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

      {/* Footer */}
      <footer className="mt-16 border-t border-white/5 bg-surface/30">
        <div className="max-w-[1700px] mx-auto px-6 py-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <Brain size={18} className="text-primary-500/50" />
              <span className="text-xs font-bold text-slate-600">
                CogniRad<span className="text-primary-500/60">++</span> v1.0 — Knowledge-Grounded Cognitive Radiology
              </span>
            </div>
            <div className="flex items-center gap-6 text-[10px] font-bold text-slate-700 uppercase tracking-widest">
              <span className="flex items-center gap-1.5"><Heart size={10} className="text-danger/40" /> Built for Clinicians</span>
              <span className="flex items-center gap-1.5"><Shield size={10} className="text-primary-500/40" /> AI-Assisted Only</span>
              <span className="flex items-center gap-1.5"><Clock size={10} className="text-slate-600" /> {new Date().getFullYear()}</span>
            </div>
          </div>
          <div className="mt-4 text-center">
            <p className="text-[10px] text-slate-700 leading-relaxed max-w-2xl mx-auto">
              Disclaimer: CogniRad++ is an AI-assisted diagnostic tool. All reports must be reviewed and validated by a qualified radiologist before clinical use.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
