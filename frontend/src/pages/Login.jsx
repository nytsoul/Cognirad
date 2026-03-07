import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Brain, Mail, Lock, Eye, EyeOff, ArrowRight, AlertCircle } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function Login() {
    const { login, loginWithGoogle } = useAuth();
    const navigate = useNavigate();
    const [form, setForm] = useState({ email: '', password: '' });
    const [showPw, setShowPw] = useState(false);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            await login(form.email, form.password);
            navigate('/dashboard');
        } catch (err) {
            setError(err.message || 'Login failed');
        } finally {
            setLoading(false);
        }
    };

    const handleGoogleSuccess = async (credentialResponse) => {
        setError('');
        setLoading(true);
        try {
            await loginWithGoogle(credentialResponse.credential);
            navigate('/dashboard');
        } catch (err) {
            setError(err.message || 'Google login failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-background flex">
            {/* Left – decorative */}
            <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden bg-gradient-to-br from-primary-600 via-primary-700 to-violet-700 items-center justify-center">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_40%,rgba(255,255,255,0.12),transparent_60%)]" />
                <div className="relative z-10 max-w-md text-center px-12">
                    <div className="w-20 h-20 mx-auto mb-8 rounded-2xl bg-white/10 backdrop-blur flex items-center justify-center">
                        <Brain size={40} className="text-white" />
                    </div>
                    <h2 className="text-4xl font-extrabold text-white mb-4 leading-tight">Welcome Back</h2>
                    <p className="text-blue-200 text-lg">
                        Sign in to continue analyzing scans with CogniRad++ AI engine.
                    </p>
                    <div className="mt-10 grid grid-cols-3 gap-4 text-center">
                        {[
                            { v: '14+', l: 'Conditions' },
                            { v: '95%', l: 'Accuracy' },
                            { v: '<3s', l: 'Speed' },
                        ].map((s) => (
                            <div key={s.l} className="p-3 rounded-xl bg-white/10 backdrop-blur">
                                <div className="text-xl font-bold text-white">{s.v}</div>
                                <div className="text-[10px] text-blue-200 uppercase tracking-wider mt-0.5">{s.l}</div>
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
                    {/* Logo (mobile) */}
                    <div className="lg:hidden flex items-center gap-2 mb-8">
                        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center text-white">
                            <Brain size={20} />
                        </div>
                        <span className="text-lg font-bold">CogniRad++</span>
                    </div>

                    <h1 className="text-3xl font-extrabold mb-1">Sign In</h1>
                    <p className="text-secondary text-sm mb-8">
                        Don't have an account?{' '}
                        <Link to="/register" className="text-primary-500 font-medium no-underline hover:underline">
                            Create one
                        </Link>
                    </p>

                    {error && (
                        <div className="flex items-center gap-2 p-3 mb-6 rounded-xl bg-red-500/10 border border-red-500/20 text-red-500 text-sm">
                            <AlertCircle size={16} /> {error}
                        </div>
                    )}

                    {/* Google button */}
                    <button
                        type="button"
                        onClick={() => {
                            // Trigger @react-oauth/google programmatically if available, else show message
                            if (window.google && window.google.accounts) {
                                window.google.accounts.id.prompt();
                            } else {
                                setError('Google Sign‑In SDK not loaded. Please add your Google Client ID to .env');
                            }
                        }}
                        className="btn w-full justify-center gap-3 py-3 mb-6 font-medium text-sm"
                    >
                        <svg width="18" height="18" viewBox="0 0 18 18"><path d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844a4.14 4.14 0 0 1-1.796 2.716v2.259h2.908c1.702-1.567 2.684-3.875 2.684-6.615z" fill="#4285F4" /><path d="M9 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z" fill="#34A853" /><path d="M3.964 10.71A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.71V4.958H.957A8.997 8.997 0 0 0 0 9c0 1.452.348 2.827.957 4.042l3.007-2.332z" fill="#FBBC05" /><path d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.958L3.964 6.29C4.672 4.163 6.656 2.58 9 2.58z" fill="#EA4335" /></svg>
                        Continue with Google
                    </button>

                    <div className="flex items-center gap-4 mb-6">
                        <div className="flex-1 h-px bg-border" />
                        <span className="text-xs text-tertiary font-medium">OR</span>
                        <div className="flex-1 h-px bg-border" />
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-5">
                        <div>
                            <label className="text-sm font-medium mb-1.5 block">Email</label>
                            <div className="relative">
                                <Mail size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-tertiary" />
                                <input
                                    id="login-email"
                                    type="email"
                                    required
                                    placeholder="you@hospital.org"
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
                                    id="login-password"
                                    type={showPw ? 'text' : 'password'}
                                    required
                                    placeholder="••••••••"
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

                        <button
                            type="submit"
                            disabled={loading}
                            className="btn btn-primary w-full py-3 text-sm font-semibold"
                        >
                            {loading ? 'Signing in…' : 'Sign In'} {!loading && <ArrowRight size={16} />}
                        </button>
                    </form>
                </motion.div>
            </div>
        </div>
    );
}
