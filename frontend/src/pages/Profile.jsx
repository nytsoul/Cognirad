import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    User, Mail, Phone, Building2, Stethoscope, Camera,
    Save, AlertCircle, CheckCircle2, Lock, Brain
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function Profile() {
    const { user, updateProfile, changePassword } = useAuth();

    const [form, setForm] = useState({
        name: '',
        email: '',
        phone: '',
        department: '',
        specialization: '',
        bio: '',
        role: '',
    });

    const [pwForm, setPwForm] = useState({ current_password: '', new_password: '', confirm: '' });
    const [msg, setMsg] = useState({ type: '', text: '' });
    const [pwMsg, setPwMsg] = useState({ type: '', text: '' });
    const [saving, setSaving] = useState(false);
    const [savingPw, setSavingPw] = useState(false);

    useEffect(() => {
        if (user) {
            setForm({
                name: user.name || '',
                email: user.email || '',
                phone: user.phone || '',
                department: user.department || '',
                specialization: user.specialization || '',
                bio: user.bio || '',
                role: user.role || 'radiologist',
            });
        }
    }, [user]);

    const handleSave = async (e) => {
        e.preventDefault();
        setMsg({ type: '', text: '' });
        setSaving(true);
        try {
            await updateProfile({
                name: form.name,
                phone: form.phone,
                department: form.department,
                specialization: form.specialization,
                bio: form.bio,
            });
            setMsg({ type: 'success', text: 'Profile updated successfully!' });
        } catch (err) {
            setMsg({ type: 'error', text: err.message });
        } finally {
            setSaving(false);
        }
    };

    const handlePasswordChange = async (e) => {
        e.preventDefault();
        setPwMsg({ type: '', text: '' });
        if (pwForm.new_password !== pwForm.confirm) {
            setPwMsg({ type: 'error', text: 'Passwords do not match' });
            return;
        }
        if (pwForm.new_password.length < 6) {
            setPwMsg({ type: 'error', text: 'Password must be at least 6 characters' });
            return;
        }
        setSavingPw(true);
        try {
            await changePassword(pwForm.current_password, pwForm.new_password);
            setPwMsg({ type: 'success', text: 'Password changed successfully!' });
            setPwForm({ current_password: '', new_password: '', confirm: '' });
        } catch (err) {
            setPwMsg({ type: 'error', text: err.message });
        } finally {
            setSavingPw(false);
        }
    };

    const initials = (user?.name || 'U').split(' ').map((w) => w[0]).join('').slice(0, 2).toUpperCase();

    return (
        <div className="max-w-4xl mx-auto space-y-8">
            {/* Header */}
            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="flex items-center gap-4">
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-primary-500 to-violet-500 flex items-center justify-center text-white text-2xl font-bold shadow-lg shadow-primary-500/20">
                    {initials}
                </div>
                <div>
                    <h1 className="text-2xl font-bold mb-0">{user?.name || 'User'}</h1>
                    <p className="text-secondary text-sm mb-0">{user?.email} · <span className="capitalize">{user?.role}</span></p>
                </div>
            </motion.div>

            {/* Profile Form */}
            <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-surface border border-border rounded-2xl p-6 md:p-8"
            >
                <h2 className="text-lg font-bold mb-6 flex items-center gap-2"><User size={20} className="text-primary-500" /> Profile Information</h2>

                {msg.text && (
                    <div className={`flex items-center gap-2 p-3 mb-6 rounded-xl text-sm ${msg.type === 'success' ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-600' : 'bg-red-500/10 border border-red-500/20 text-red-500'
                        }`}>
                        {msg.type === 'success' ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />} {msg.text}
                    </div>
                )}

                <form onSubmit={handleSave} className="space-y-5">
                    <div className="grid md:grid-cols-2 gap-5">
                        <div>
                            <label className="text-sm font-medium mb-1.5 block">Full Name</label>
                            <div className="relative">
                                <User size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-tertiary" />
                                <input
                                    type="text"
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
                                <input type="email" value={form.email} disabled className="pl-10 opacity-60 cursor-not-allowed" />
                            </div>
                        </div>
                    </div>

                    <div className="grid md:grid-cols-2 gap-5">
                        <div>
                            <label className="text-sm font-medium mb-1.5 block">Phone</label>
                            <div className="relative">
                                <Phone size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-tertiary" />
                                <input
                                    type="tel"
                                    value={form.phone}
                                    onChange={(e) => setForm({ ...form, phone: e.target.value })}
                                    placeholder="+1 (555) 000-0000"
                                    className="pl-10"
                                />
                            </div>
                        </div>
                        <div>
                            <label className="text-sm font-medium mb-1.5 block">Department</label>
                            <div className="relative">
                                <Building2 size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-tertiary" />
                                <input
                                    type="text"
                                    value={form.department}
                                    onChange={(e) => setForm({ ...form, department: e.target.value })}
                                    placeholder="Radiology"
                                    className="pl-10"
                                />
                            </div>
                        </div>
                    </div>

                    <div>
                        <label className="text-sm font-medium mb-1.5 block">Specialization</label>
                        <div className="relative">
                            <Stethoscope size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-tertiary" />
                            <input
                                type="text"
                                value={form.specialization}
                                onChange={(e) => setForm({ ...form, specialization: e.target.value })}
                                placeholder="Neuroradiology, Chest imaging…"
                                className="pl-10"
                            />
                        </div>
                    </div>

                    <div>
                        <label className="text-sm font-medium mb-1.5 block">Bio</label>
                        <textarea
                            value={form.bio}
                            onChange={(e) => setForm({ ...form, bio: e.target.value })}
                            placeholder="A short description about yourself…"
                            rows={3}
                            style={{ fontFamily: 'inherit', fontSize: '0.875rem' }}
                        />
                    </div>

                    <div className="flex justify-end">
                        <button type="submit" disabled={saving} className="btn btn-primary gap-2">
                            <Save size={16} /> {saving ? 'Saving…' : 'Save Changes'}
                        </button>
                    </div>
                </form>
            </motion.div>

            {/* Change Password */}
            <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-surface border border-border rounded-2xl p-6 md:p-8"
            >
                <h2 className="text-lg font-bold mb-6 flex items-center gap-2"><Lock size={20} className="text-primary-500" /> Change Password</h2>

                {pwMsg.text && (
                    <div className={`flex items-center gap-2 p-3 mb-6 rounded-xl text-sm ${pwMsg.type === 'success' ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-600' : 'bg-red-500/10 border border-red-500/20 text-red-500'
                        }`}>
                        {pwMsg.type === 'success' ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />} {pwMsg.text}
                    </div>
                )}

                <form onSubmit={handlePasswordChange} className="space-y-5">
                    <div>
                        <label className="text-sm font-medium mb-1.5 block">Current Password</label>
                        <input
                            type="password"
                            value={pwForm.current_password}
                            onChange={(e) => setPwForm({ ...pwForm, current_password: e.target.value })}
                            placeholder="Enter current password"
                        />
                    </div>
                    <div className="grid md:grid-cols-2 gap-5">
                        <div>
                            <label className="text-sm font-medium mb-1.5 block">New Password</label>
                            <input
                                type="password"
                                value={pwForm.new_password}
                                onChange={(e) => setPwForm({ ...pwForm, new_password: e.target.value })}
                                placeholder="Min. 6 characters"
                            />
                        </div>
                        <div>
                            <label className="text-sm font-medium mb-1.5 block">Confirm New Password</label>
                            <input
                                type="password"
                                value={pwForm.confirm}
                                onChange={(e) => setPwForm({ ...pwForm, confirm: e.target.value })}
                                placeholder="Repeat new password"
                            />
                        </div>
                    </div>
                    <div className="flex justify-end">
                        <button type="submit" disabled={savingPw} className="btn btn-primary gap-2">
                            <Lock size={16} /> {savingPw ? 'Changing…' : 'Change Password'}
                        </button>
                    </div>
                </form>
            </motion.div>
        </div>
    );
}
