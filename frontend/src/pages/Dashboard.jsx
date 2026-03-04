import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  BarChart3, TrendingUp, CheckCircle, AlertCircle, Clock, Eye,
  FileText, Users, Activity, ArrowRight, Plus
} from 'lucide-react';
import { motion } from 'framer-motion';

function Dashboard() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('overview');

  // Mock data
  const stats = [
    { icon: FileText, label: 'TOTAL REPORTS', value: '1,284', change: '+12%', color: 'primary' },
    { icon: Eye, label: 'AVG. CONFIDENCE', value: '88%', change: 'Stable', color: 'success' },
    { icon: Activity, label: 'PATHOLOGIES DETECTED', value: '342', change: 'Across 14 categories', color: 'accent' },
    { icon: CheckCircle, label: 'SYSTEM STATUS', value: 'Online', change: 'v2.4.1', color: 'success' },
  ];

  const recentReports = [
    { id: '#PAT-8821', date: 'Oct 26, 2023, 10:45 AM', patient: 'James Miller', diagnosis: 'Pneumonia', confidence: 98.4, status: 'Reviewed' },
    { id: '#PAT-7742', date: 'Oct 24, 2023, 09:12 AM', patient: 'Maria Garcia', diagnosis: 'No Finding', confidence: 99.1, status: 'Reviewed' },
    { id: '#PAT-9910', date: 'Oct 23, 2023, 06:30 PM', patient: 'Robert Wilson', diagnosis: 'Pleural Effusion', confidence: 87.2, status: 'Pending' },
    { id: '#PAT-6631', date: 'Oct 25, 2023, 01:15 PM', patient: 'Sarah Chen', diagnosis: 'Cardiomegaly', confidence: 94.8, status: 'Reviewed' },
  ];

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Page Header */}
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-4xl font-black text-primary mb-2">Dashboard</h1>
        <p className="text-secondary text-sm">Advanced medical AI dashboard for explainable chest X-ray analysis, deep-tissue pathology detection, and automated reporting.</p>
      </motion.div>

      {/* Stats Grid */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        {stats.map((stat, i) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 + i * 0.05 }}
            className="bg-surface border border-border rounded-2xl p-6 hover:border-primary-500/20 transition-all group"
          >
            <div className="flex items-start justify-between mb-4">
              <div className={`p-3 rounded-xl bg-${stat.color}-500/10 group-hover:bg-${stat.color}-500/15 transition-colors`}>
                <stat.icon size={20} className={`text-${stat.color}-400`} />
              </div>
              <span className="text-[10px] font-bold text-secondary uppercase tracking-widest">{stat.label}</span>
            </div>
            <div className="mb-2">
              <div className="text-3xl font-black text-primary">{stat.value}</div>
              <div className={`text-xs font-bold mt-2 ${
                stat.change.includes('Stable') || stat.change.includes('Across') ? 'text-secondary' : 'text-success'
              }`}>
                {stat.change}
              </div>
            </div>
          </motion.div>
        ))}
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className="lg:col-span-1 space-y-4"
        >
          <h3 className="text-sm font-black uppercase tracking-[0.2em] text-secondary px-2">Quick Actions</h3>
          
          <button
            onClick={() => navigate('/upload')}
            className="group w-full bg-primary-600 rounded-2xl p-6 hover:bg-primary-500 transition-all overflow-hidden relative"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
            <div className="relative">
              <div className="flex items-start justify-between mb-3">
                <div className="w-12 h-12 bg-primary-500/20 rounded-xl flex items-center justify-center text-primary">
                  <Plus size={24} />
                </div>
              </div>
              <h4 className="text-lg font-bold text-primary text-left mb-1">Upload X-Ray</h4>
              <p className="text-sm text-secondary text-left">Instant analysis of new chest scans using explainable AI models. Supports DICOM and high-res JPEG formats.</p>
            </div>
          </button>

          <button className="group w-full bg-surface border border-border rounded-2xl p-6 hover:border-primary-500/30 transition-all">
            <div className="flex items-start justify-between">
              <div>
                <div className="w-12 h-12 bg-primary-500/10 rounded-xl flex items-center justify-center text-medical mb-3">
                  <FileText size={24} />
                </div>
                <h4 className="text-lg font-bold text-primary text-left mb-1">View Reports</h4>
                <p className="text-sm text-secondary text-left">Access historical analysis, generated findings, and collaborative notes from previous patient examinations.</p>
              </div>
              <ArrowRight size={18} className="text-secondary group-hover:text-primary shrink-0 mt-2 transition-colors" />
            </div>
          </button>
        </motion.div>

        {/* Recent Reports */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="lg:col-span-2"
        >
          <div className="bg-surface border border-border rounded-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-border flex items-center justify-between">
              <h3 className="text-sm font-bold text-primary uppercase tracking-widest">Recent Reports</h3>
              <button className="text-medical hover:text-primary text-xs font-bold uppercase tracking-widest transition-colors">
                View All
              </button>
            </div>

            <div className="divide-y divide-white/5">
              {recentReports.map((report, i) => (
                <motion.div
                  key={report.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 + i * 0.05 }}
                  className="px-6 py-4 hover:bg-white/5 transition-colors cursor-pointer group"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-4 flex-1">
                      <div className="text-sm font-bold text-white">{report.id}</div>
                      <div className="text-xs text-slate-600 font-medium">{report.patient}</div>
                    </div>
                    <div className={`text-xs font-bold px-2 py-1 rounded-full ${
                      report.status === 'Reviewed'
                        ? 'bg-success/10 text-success'
                        : 'bg-accent/10 text-accent'
                    }`}>
                      {report.status}
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-6">
                      <span className="text-[11px] text-slate-600 font-medium">{report.date}</span>
                      <span className="text-sm font-bold text-slate-300">{report.diagnosis}</span>
                    </div>
                    <div className="text-right">
                      <span className="text-sm font-bold text-primary-400">{report.confidence}%</span>
                      <div className="w-16 h-1.5 bg-white/5 rounded-full mt-1 overflow-hidden">
                        <div
                          className="h-full bg-primary-500 rounded-full"
                          style={{ width: `${report.confidence}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>

      {/* Analytics Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="grid grid-cols-1 lg:grid-cols-2 gap-6"
      >
        {/* Performance Metrics */}
        <div className="bg-surface border border-white/10 rounded-2xl p-6">
          <h3 className="text-sm font-bold text-white uppercase tracking-widest mb-6 flex items-center gap-2">
            <TrendingUp size={16} className="text-primary-400" />
            Performance Metrics
          </h3>
          <div className="space-y-4">
            {[
              { label: 'Model Accuracy', value: 92, color: 'primary' },
              { label: 'Sensitivity', value: 94, color: 'success' },
              { label: 'Specificity', value: 89, color: 'accent' },
            ].map((metric) => (
              <div key={metric.label}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">{metric.label}</span>
                  <span className="text-sm font-bold text-white">{metric.value}%</span>
                </div>
                <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                  <div
                    className={`h-full bg-${metric.color}-500 rounded-full`}
                    style={{ width: `${metric.value}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* System Info */}
        <div className="bg-gradient-to-br from-primary-600/20 to-primary-600/5 border border-primary-500/10 rounded-2xl p-6">
          <h3 className="text-sm font-bold text-white uppercase tracking-widest mb-6 flex items-center gap-2">
            <Activity size={16} className="text-primary-400" />
            System Information
          </h3>
          <div className="space-y-4 text-sm">
            <div className="flex justify-between items-center">
              <span className="text-slate-400">Model Version</span>
              <span className="font-bold text-white">v2.4.1</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-400">Inference Time</span>
              <span className="font-bold text-success">~1.2s</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-400">Pathology Classes</span>
              <span className="font-bold text-white">14</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-400">Hallucination Rate</span>
              <span className="font-bold text-success">&lt; 3%</span>
            </div>
            <div className="flex justify-between items-center pt-3 border-t border-white/10">
              <span className="text-slate-400">Status</span>
              <span className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
                <span className="font-bold text-success">Operational</span>
              </span>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

export default Dashboard;
