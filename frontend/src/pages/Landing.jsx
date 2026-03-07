import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Brain, Upload, BarChart3, FileText, Shield, Zap,
  ArrowRight, Microscope, Activity, Layers, ChevronRight,
  Sparkles, Lock, Globe
} from 'lucide-react';

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.12, duration: 0.6, ease: [0.22, 1, 0.36, 1] },
  }),
};

const features = [
  {
    icon: Upload,
    title: 'Smart Scan Upload',
    desc: 'Upload chest X-rays in DICOM, JPEG, or PNG. Our pipeline preprocesses images in real-time.',
    color: 'from-sky-500 to-blue-600',
  },
  {
    icon: Brain,
    title: 'Cognitive AI Analysis',
    desc: 'Multi-stage reasoning with perception, diagnosis, and verification layers for accurate reports.',
    color: 'from-violet-500 to-purple-600',
  },
  {
    icon: FileText,
    title: 'Auto Report Generation',
    desc: 'Generate structured radiology reports with findings, impressions, and clinical correlations.',
    color: 'from-emerald-500 to-green-600',
  },
  {
    icon: Microscope,
    title: 'Evidence Mapping',
    desc: 'Visual attention maps and reasoning chains show exactly why the AI reached its conclusion.',
    color: 'from-amber-500 to-orange-600',
  },
  {
    icon: Activity,
    title: 'Disease Prediction',
    desc: 'Detect 14 thoracic conditions with calibrated probabilities and uncertainty estimation.',
    color: 'from-rose-500 to-pink-600',
  },
  {
    icon: BarChart3,
    title: 'Patient Tracking',
    desc: 'Full patient history with report versioning, comparisons, and longitudinal analysis.',
    color: 'from-teal-500 to-cyan-600',
  },
];

const stats = [
  { value: '14+', label: 'Conditions Detected' },
  { value: '95%', label: 'Accuracy Rate' },
  { value: '<3s', label: 'Analysis Speed' },
  { value: '24/7', label: 'Availability' },
];

export default function Landing() {
  return (
    <div className="min-h-screen bg-background text-primary overflow-x-hidden">
      {/* ─── Navbar ──────────────────────────────────── */}
      <motion.nav
        initial={{ y: -60, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="fixed top-0 inset-x-0 z-50 bg-surface/70 backdrop-blur-xl border-b border-border"
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between px-6 h-16">
          <Link to="/" className="flex items-center gap-3 no-underline hover:no-underline">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center text-white">
              <Brain size={20} />
            </div>
            <span className="text-lg font-bold text-primary tracking-tight">CogniRad++</span>
          </Link>
          <div className="flex items-center gap-3">
            <Link
              to="/login"
              className="btn btn-ghost text-sm px-4 py-2 no-underline hover:no-underline"
            >
              Sign In
            </Link>
            <Link
              to="/register"
              className="btn btn-primary text-sm px-5 py-2 no-underline hover:no-underline"
            >
              Get Started <ArrowRight size={14} />
            </Link>
          </div>
        </div>
      </motion.nav>

      {/* ─── Hero ──────────────────────────────────── */}
      <section className="relative pt-32 pb-20 px-6 overflow-hidden">
        {/* Gradient orbs */}
        <div className="absolute -top-32 -right-32 w-[500px] h-[500px] rounded-full bg-primary-500/10 blur-[120px] pointer-events-none" />
        <div className="absolute -bottom-32 -left-32 w-[400px] h-[400px] rounded-full bg-violet-500/10 blur-[100px] pointer-events-none" />

        <div className="max-w-5xl mx-auto text-center relative z-10">
          <motion.div
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            custom={0}
            className="inline-flex items-center gap-2 px-4 py-1 rounded-full bg-primary-500/10 border border-primary-500/20 text-primary-500 text-xs font-semibold tracking-wide uppercase mb-6"
          >
            <Sparkles size={14} /> AI-Powered Radiology
          </motion.div>

          <motion.h1
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            custom={1}
            className="text-5xl md:text-7xl font-extrabold leading-[1.1] tracking-tight mb-0"
          >
            Intelligent Radiology
            <br />
            <span className="bg-gradient-to-r from-primary-400 via-primary-500 to-violet-500 bg-clip-text text-transparent">
              Report Generation
            </span>
          </motion.h1>

          <motion.p
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            custom={2}
            className="mt-6 text-lg md:text-xl text-secondary max-w-2xl mx-auto leading-relaxed"
          >
            CogniRad++ leverages a multi-stage cognitive pipeline to analyze chest X-rays, 
            detect diseases, and produce evidence-grounded radiology reports in seconds.
          </motion.p>

          <motion.div
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            custom={3}
            className="mt-10 flex flex-wrap items-center justify-center gap-4"
          >
            <Link
              to="/register"
              className="btn btn-primary btn-lg gap-2 text-base rounded-2xl shadow-lg shadow-primary-500/25 no-underline hover:no-underline"
            >
              Start Analyzing <ArrowRight size={18} />
            </Link>
            <Link
              to="/login"
              className="btn btn-outline btn-lg text-base rounded-2xl no-underline hover:no-underline"
            >
              Sign In
            </Link>
          </motion.div>
        </div>
      </section>

      {/* ─── Stats bar ────────────────────────────── */}
      <section className="py-10">
        <div className="max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-6 px-6">
          {stats.map((s, i) => (
            <motion.div
              key={s.label}
              variants={fadeUp}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              custom={i}
              className="text-center p-6 rounded-2xl bg-surface border border-border"
            >
              <div className="text-3xl md:text-4xl font-extrabold bg-gradient-to-r from-primary-500 to-violet-500 bg-clip-text text-transparent">
                {s.value}
              </div>
              <div className="text-xs text-secondary mt-1 font-medium uppercase tracking-wider">{s.label}</div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ─── Features Grid ────────────────────────── */}
      <section className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <motion.div
            variants={fadeUp}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="text-center mb-14"
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Everything You Need</h2>
            <p className="text-secondary max-w-xl mx-auto">
              A complete, end-to-end radiology workflow — from scan upload to final report.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((f, i) => (
              <motion.div
                key={f.title}
                variants={fadeUp}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                custom={i}
                className="group p-6 bg-surface border border-border rounded-2xl hover:border-primary-500/40 transition-all duration-300 hover:shadow-xl hover:shadow-primary-500/5"
              >
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${f.color} flex items-center justify-center text-white mb-4 group-hover:scale-110 transition-transform duration-300`}>
                  <f.icon size={22} />
                </div>
                <h3 className="text-lg font-semibold mb-2">{f.title}</h3>
                <p className="text-sm text-secondary leading-relaxed mb-0">{f.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── How it works ──────────────────────────── */}
      <section className="py-20 px-6 bg-surface/50">
        <div className="max-w-5xl mx-auto">
          <motion.div
            variants={fadeUp}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="text-center mb-14"
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-4">How It Works</h2>
            <p className="text-secondary max-w-xl mx-auto">Three simple steps to go from raw X-ray to clinical report.</p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              { step: '01', title: 'Upload', desc: 'Drag and drop your chest X-ray image.', icon: Upload },
              { step: '02', title: 'Analyze', desc: 'AI processes perception → diagnosis → verification.', icon: Brain },
              { step: '03', title: 'Report', desc: 'Get a structured, evidence-backed clinical report.', icon: FileText },
            ].map((item, i) => (
              <motion.div
                key={item.step}
                variants={fadeUp}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                custom={i}
                className="relative text-center"
              >
                <div className="text-6xl font-black text-primary-500/10 mb-2">{item.step}</div>
                <div className="w-14 h-14 mx-auto rounded-2xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center text-white mb-4">
                  <item.icon size={24} />
                </div>
                <h3 className="text-xl font-bold mb-2">{item.title}</h3>
                <p className="text-sm text-secondary">{item.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── CTA ──────────────────────────────────── */}
      <section className="py-24 px-6">
        <motion.div
          variants={fadeUp}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="max-w-4xl mx-auto text-center p-12 rounded-3xl bg-gradient-to-br from-primary-600 to-violet-600 text-white relative overflow-hidden"
        >
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(255,255,255,0.15),transparent_60%)]" />
          <div className="relative z-10">
            <h2 className="text-3xl md:text-4xl font-bold mb-4 text-white">
              Ready to Transform Your Workflow?
            </h2>
            <p className="text-blue-100 max-w-lg mx-auto mb-8 text-lg">
              Join healthcare professionals using CogniRad++ for faster, more accurate radiology reporting.
            </p>
            <Link
              to="/register"
              className="inline-flex items-center gap-2 px-8 py-3 bg-white text-primary-700 font-bold rounded-2xl hover:bg-blue-50 transition-colors no-underline hover:no-underline"
            >
              Create Free Account <ChevronRight size={18} />
            </Link>
          </div>
        </motion.div>
      </section>

      {/* ─── Footer ────────────────────────────────── */}
      <footer className="border-t border-border py-10 px-6">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-sm text-secondary">
            <Brain size={16} className="text-primary-500" />
            <span className="font-semibold text-primary">CogniRad++</span>
            <span>&copy; {new Date().getFullYear()}</span>
          </div>
          <div className="flex items-center gap-6 text-xs text-secondary">
            <span className="flex items-center gap-1"><Lock size={12} /> HIPAA Compliant</span>
            <span className="flex items-center gap-1"><Shield size={12} /> SOC 2</span>
            <span className="flex items-center gap-1"><Globe size={12} /> Open Source</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
