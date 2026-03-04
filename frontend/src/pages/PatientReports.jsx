import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, Search, Filter, ChevronRight, Eye, Download, Share2, AlertCircle, CheckCircle } from 'lucide-react';
import { motion } from 'framer-motion';

function PatientReports() {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('All');
  const [currentPage, setCurrentPage] = useState(1);

  // Mock data
  const reports = [
    {
      id: '#PAT-8821',
      patient: 'James Miller',
      date: 'Oct 26, 2023, 10:45 AM',
      diagnosis: 'Pneumonia',
      confidence: 98.4,
      status: 'Reviewed',
      statusColor: 'success'
    },
    {
      id: '#PAT-7742',
      patient: 'Maria Garcia',
      date: 'Oct 24, 2023, 09:12 AM',
      diagnosis: 'No Finding',
      confidence: 99.1,
      status: 'Reviewed',
      statusColor: 'success'
    },
    {
      id: '#PAT-9910',
      patient: 'Robert Wilson',
      date: 'Oct 23, 2023, 06:30 PM',
      diagnosis: 'Pleural Effusion',
      confidence: 87.2,
      status: 'Pending',
      statusColor: 'accent'
    },
    {
      id: '#PAT-6631',
      patient: 'Sarah Chen',
      date: 'Oct 25, 2023, 01:15 PM',
      diagnosis: 'Cardiomegaly',
      confidence: 94.8,
      status: 'Reviewed',
      statusColor: 'success'
    },
    {
      id: '#PAT-5520',
      patient: 'David Smith',
      date: 'Oct 22, 2023, 11:00 AM',
      diagnosis: 'Lung Nodule',
      confidence: 76.5,
      status: 'Pending',
      statusColor: 'accent'
    },
  ];

  const stats = [
    { icon: CheckCircle, label: 'Reports Reviewed Today', value: 142, color: 'success' },
    { icon: AlertCircle, label: 'Pending AI Validations', value: 14, color: 'accent' },
    { icon: Eye, label: 'Avg. Analysis Speed', value: '1.2s', color: 'primary' },
  ];

  const filteredReports = reports.filter((report) => {
    const matchesSearch =
      report.patient.toLowerCase().includes(searchTerm.toLowerCase()) ||
      report.id.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'All' || report.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Page Header */}
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-start justify-between gap-6">
          <div>
            <h1 className="text-4xl font-black text-white mb-2">Patient Reports History</h1>
            <p className="text-slate-400 text-sm">Explainable AI analysis for diagnostic imaging workflow management.</p>
          </div>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => navigate('/upload')}
            className="px-6 py-3 bg-primary-600 hover:bg-primary-500 text-white rounded-xl text-sm font-bold flex items-center gap-2 transition-all shadow-glow shrink-0"
          >
            <Plus size={18} />
            New Analysis
          </motion.button>
        </div>
      </motion.div>

      {/* Stats */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-6"
      >
        {stats.map((stat, i) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 + i * 0.05 }}
            className="bg-surface border border-white/10 rounded-2xl p-6 hover:border-primary-500/20 transition-all group"
          >
            <div className="flex items-start justify-between mb-3">
              <div className={`p-3 rounded-xl bg-${stat.color}-500/10 group-hover:bg-${stat.color}-500/15 transition-colors`}>
                <stat.icon size={20} className={`text-${stat.color}-400`} />
              </div>
            </div>
            <div className="text-3xl font-black text-white mb-1">{stat.value}</div>
            <div className="text-xs font-bold text-slate-600 uppercase tracking-widest">{stat.label}</div>
          </motion.div>
        ))}
      </motion.div>

      {/* Controls */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        className="flex flex-col md:flex-row gap-4"
      >
        {/* Search */}
        <div className="flex-1 relative">
          <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-600" />
          <input
            type="text"
            placeholder="Search Patient ID, Physician, or Diagnosis..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-12 pr-4 py-3 bg-surface border border-white/10 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none transition-all text-sm font-medium"
          />
        </div>

        {/* Filters */}
        <div className="flex items-center gap-3">
          <select
            className="px-4 py-3 bg-surface border border-white/10 rounded-xl focus:ring-2 focus:ring-primary-500 outline-none transition-all text-sm font-bold uppercase tracking-widest cursor-pointer"
          >
            <option>Date Range</option>
            <option>Last 7 days</option>
            <option>Last 30 days</option>
            <option>Last 90 days</option>
          </select>

          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-4 py-3 bg-surface border border-white/10 rounded-xl focus:ring-2 focus:ring-primary-500 outline-none transition-all text-sm font-bold uppercase tracking-widest cursor-pointer"
          >
            <option>Diagnosis: All</option>
            <option>Pneumonia</option>
            <option>No Finding</option>
            <option>Pleural Effusion</option>
            <option>Cardiomegaly</option>
          </select>
        </div>
      </motion.div>

      {/* Reports Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-surface border border-white/10 rounded-2xl overflow-hidden"
      >
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-white/[0.05] border-b border-white/10">
                <th className="px-6 py-4 text-left text-xs font-black text-slate-600 uppercase tracking-widest">
                  PATIENT ID
                </th>
                <th className="px-6 py-4 text-left text-xs font-black text-slate-600 uppercase tracking-widest">
                  SCAN DATE
                </th>
                <th className="px-6 py-4 text-left text-xs font-black text-slate-600 uppercase tracking-widest">
                  AI DIAGNOSIS
                </th>
                <th className="px-6 py-4 text-center text-xs font-black text-slate-600 uppercase tracking-widest">
                  CONFIDENCE
                </th>
                <th className="px-6 py-4 text-center text-xs font-black text-slate-600 uppercase tracking-widest">
                  STATUS
                </th>
                <th className="px-6 py-4 text-right text-xs font-black text-slate-600 uppercase tracking-widest">
                  ACTIONS
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {filteredReports.map((report, i) => (
                <motion.tr
                  key={report.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 + i * 0.03 }}
                  className="hover:bg-white/[0.02] transition-colors group"
                >
                  <td className="px-6 py-4">
                    <div>
                      <div className="text-sm font-bold text-white">{report.id}</div>
                      <div className="text-xs text-slate-600 font-medium mt-1">{report.patient}</div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <span className="text-xs font-bold text-slate-400">{report.date}</span>
                  </td>
                  <td className="px-6 py-4">
                    <span className="text-sm font-bold text-white">{report.diagnosis}</span>
                  </td>
                  <td className="px-6 py-4 text-center">
                    <div className="flex items-center justify-center gap-2">
                      <span className="text-sm font-bold text-primary-400">{report.confidence}%</span>
                      <div className="w-16 h-1.5 bg-white/5 rounded-full overflow-hidden hidden sm:block">
                        <div
                          className="h-full bg-primary-500 rounded-full"
                          style={{ width: `${report.confidence}%` }}
                        />
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-center">
                    <span className={`text-xs font-bold px-2 py-1 rounded-full ${
                      report.statusColor === 'success'
                        ? 'bg-success/10 text-success'
                        : 'bg-accent/10 text-accent'
                    }`}>
                      {report.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <button
                        onClick={() => navigate(`/analysis/${report.id}`)}
                        className="p-2 hover:bg-white/10 rounded-lg transition-colors text-slate-400 hover:text-slate-300"
                        title="View Report"
                      >
                        <Eye size={16} />
                      </button>
                      <button
                        className="p-2 hover:bg-white/10 rounded-lg transition-colors text-slate-400 hover:text-slate-300"
                        title="Download"
                      >
                        <Download size={16} />
                      </button>
                      <button
                        onClick={() => navigate(`/analysis/${report.id}`)}
                        className="p-2 hover:bg-primary-500/20 rounded-lg transition-colors text-slate-400 hover:text-primary-400 group-hover:flex hidden"
                      >
                        <ChevronRight size={16} />
                      </button>
                    </div>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="bg-white/[0.02] border-t border-white/10 px-6 py-4 flex items-center justify-between text-sm">
          <div className="text-slate-600 font-bold">
            Showing 1 to 5 of 156 results
          </div>
          <div className="flex items-center gap-2">
            <button className="p-2 hover:bg-white/10 rounded-lg transition-colors text-slate-600 hover:text-slate-400 disabled:opacity-50"
              disabled
            >
              ←
            </button>
            {[1, 2, 3].map((page) => (
              <button
                key={page}
                onClick={() => setCurrentPage(page)}
                className={`px-4 py-2 rounded-lg transition-all text-sm font-bold ${
                  currentPage === page
                    ? 'bg-primary-600 text-white'
                    : 'hover:bg-white/10 text-slate-400'
                }`}
              >
                {page}
              </button>
            ))}
            <span className="text-slate-600">...</span>
            <button className="px-4 py-2 rounded-lg hover:bg-white/10 text-slate-400 transition-all text-sm font-bold">
              32
            </button>
            <button className="p-2 hover:bg-white/10 rounded-lg transition-colors text-slate-400 hover:text-slate-300">
              →
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

export default PatientReports;
