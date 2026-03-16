import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Brain, Mail, Lock, Eye, EyeOff, User, ArrowRight, AlertCircle, Stethoscope } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const roles = [
    { value: 'radiologist', label: 'Radiologist', icon: '🩻' },
    { value: 'clinician', label: 'Clinician', icon: '🩺' },
    { value: 'researcher', label: 'Researcher', icon: '🔬' },
    { value: 'admin', label: 'Admin', icon: '⚙️' },
];

export default function Register() {
    const { register } = useAuth();
    const navigate = useNavigate();
    const [form, setForm] = useState({ name: '', email: '', password: '', role: 'radiologist' });
    const [showPw, setShowPw] = useState(false);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        if (form.password.length < 6) {
            setError('Password must be at least 6 characters');
            return;
        }
        setLoading(true);
        try {
            await register(form);
            navigate('/dashboard');
        } catch (err) {
            setError(err.message || 'Registration failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-background flex">
            {/* Left – decorative */}
            <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden bg-gradient-to-br from-emerald-600 via-teal-700 to-primary-700 items-center justify-center">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_30%,rgba(255,255,255,0.12),transparent_60%)]" />
                <div className="relative z-10 max-w-md text-center px-12">
                    <div className="w-20 h-20 mx-auto mb-8 rounded-2xl bg-white/10 backdrop-blur flex items-center justify-center">
                        <Stethoscope size={40} className="text-white" />
                    </div>
                    <h2 className="text-4xl font-extrabold text-white mb-4 leading-tight">Join CogniRad++</h2>
                    <p className="text-emerald-100 text-lg">
                        Create your account and start generating AI-powered radiology reports in minutes.
                    </p>
                    <div className="mt-10 space-y-3 text-left">
                        {[
                            'Multi-stage cognitive AI analysis',
                            'Evidence-mapped visual attention',
                            'Structured clinical reports',
                            'HIPAA-compliant data handling',
                        ].map((item) => (
                            <div key={item} className="flex items-center gap-3 text-white/90 text-sm">
                                <div className="w-5 h-5 rounded-full bg-white/20 flex items-center justify-center text-[10px]">✓</div>
                                {item}
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Right – form */}
            <div className="flex-1 flex items-center justify-center px-6 py-12">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    className="w-full max-w-md"
                >
                    <div className="lg:hidden flex items-center gap-2 mb-8">
                        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center text-white">
                            <Brain size={20} />
                        </div>
                        <span className="text-lg font-bold">CogniRad++</span>
                    </div>

                    <h1 className="text-3xl font-extrabold mb-1">Create Account</h1>
                    <p className="text-secondary text-sm mb-8">
                        Already have an account?{' '}
                        <Link to="/login" className="text-primary-500 font-medium no-underline hover:underline">
                            Sign in
                        </Link>
                    </p>

                    {error && (
                        <div className="flex items-center gap-2 p-3 mb-6 rounded-xl bg-red-500/10 border border-red-500/20 text-red-500 text-sm">
                            <AlertCircle size={16} /> {error}
                        </div>
                    )}

                    <form onSubmit={handleSubmit} className="space-y-5">
                        <div>
                            <label className="text-sm font-medium mb-1.5 block">Full Name</label>
                            <div className="relative">
                                <User size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-tertiary" />
                                <input
                                    id="register-name"
                                    type="text"
                                    required
                                    placeholder="Dr. Jane Smith"
                                    value={form.name}
                                    onChange={(e) => setForm({ ...form, name: e.target.value })}
                                    className="pl-10"
                                />
                            </div>
                        </div>

                        <div>
                            <label className="text-sm font-medium mb-1.5 block">Email</label>
                            <div className="relative">
                                <Mail size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-tertiary" />
                                <input
                                    id="register-email"
                                    type="email"
                                    required
                                    placeholder="jane@hospital.org"
                                    value={form.email}
                                    onChange={(e) => setForm({ ...form, email: e.target.value })}
                                    className="pl-10"
                                />
                            </div>
                        </div>

                        <div>
                            <label className="text-sm font-medium mb-1.5 block">Password</label>
                            <div className="relative">
                                <Lock size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-tertiary" />
                                <input
                                    id="register-password"
                                    type={showPw ? 'text' : 'password'}
                                    required
                                    placeholder="Min. 6 characters"
                                    value={form.password}
                                    onChange={(e) => setForm({ ...form, password: e.target.value })}
                                    className="pl-10 pr-10"
                                />
                                <button
                                    type="button"
                                    onClick={() => setShowPw(!showPw)}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 text-tertiary hover:text-secondary"
                                >
                                    {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
                                </button>
                            </div>
                        </div>

                        <div>
                            <label className="text-sm font-medium mb-2 block">Role</label>
                            <div className="grid grid-cols-2 gap-2">
                                {roles.map((r) => (
                                    <button
                                        key={r.value}
                                        type="button"
                                        onClick={() => setForm({ ...form, role: r.value })}
                                        className={`flex items-center gap-2 px-3 py-2.5 rounded-xl border text-sm font-medium transition-all ${form.role === r.value
                                                ? 'bg-primary-500/10 border-primary-500/40 text-primary-600'
                                                : 'bg-surface border-border text-secondary hover:border-primary-500/20'
                                            }`}
                                    >
                                        <span>{r.icon}</span> {r.label}
                                    </button>
                                ))}
                            </div>
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className="btn btn-primary w-full py-3 text-sm font-semibold"
                        >
                            {loading ? 'Creating account…' : 'Create Account'} {!loading && <ArrowRight size={16} />}
                        </button>
                    </form>
                </motion.div>
            </div>
        </div>
    );
}
