import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
    Settings as SettingsIcon, Moon, Sun, Bell, BellOff,
    Globe, Monitor, Palette, Shield, Save, CheckCircle2
} from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

export default function Settings() {
    const { theme, toggleTheme } = useTheme();
    const [notifications, setNotifications] = useState(true);
    const [language, setLanguage] = useState('en');
    const [saved, setSaved] = useState(false);

    const handleSave = () => {
        setSaved(true);
        setTimeout(() => setSaved(false), 2500);
    };

    return (
        <div className="max-w-3xl mx-auto space-y-8">
            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
                <h1 className="text-2xl font-bold mb-1">Settings</h1>
                <p className="text-secondary text-sm mb-0">Manage your preferences and application configuration.</p>
            </motion.div>

            {/* Appearance */}
            <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.08 }}
                className="bg-surface border border-border rounded-2xl p-6"
            >
                <h2 className="text-lg font-bold mb-5 flex items-center gap-2">
                    <Palette size={20} className="text-primary-500" /> Appearance
                </h2>

                <div className="flex items-center justify-between py-3">
                    <div>
                        <div className="text-sm font-medium">Theme</div>
                        <div className="text-xs text-secondary mt-0.5">Switch between light and dark mode</div>
                    </div>
                    <button
                        onClick={toggleTheme}
                        className="flex items-center gap-2 px-4 py-2 rounded-xl bg-surface-secondary border border-border text-sm font-medium hover:border-primary-500/30 transition-all"
                    >
                        {theme === 'dark' ? (
                            <><Moon size={16} className="text-violet-400" /> Dark</>
                        ) : (
                            <><Sun size={16} className="text-amber-500" /> Light</>
                        )}
                    </button>
                </div>
            </motion.div>

            {/* Notifications */}
            <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.16 }}
                className="bg-surface border border-border rounded-2xl p-6"
            >
                <h2 className="text-lg font-bold mb-5 flex items-center gap-2">
                    <Bell size={20} className="text-primary-500" /> Notifications
                </h2>

                <div className="space-y-4">
                    {[
                        { label: 'Report ready alerts', desc: 'Get notified when an analysis is complete', key: 'reports' },
                        { label: 'System updates', desc: 'Updates about platform changes', key: 'system' },
                        { label: 'Weekly digest', desc: 'Summary of weekly scan statistics', key: 'digest' },
                    ].map((item) => (
                        <div key={item.key} className="flex items-center justify-between py-2">
                            <div>
                                <div className="text-sm font-medium">{item.label}</div>
                                <div className="text-xs text-secondary mt-0.5">{item.desc}</div>
                            </div>
                            <button
                                onClick={() => setNotifications(!notifications)}
                                className={`relative w-12 h-7 rounded-full transition-colors duration-200 ${notifications ? 'bg-primary-500' : 'bg-border'
                                    }`}
                            >
                                <div
                                    className={`absolute top-0.5 w-6 h-6 rounded-full bg-white shadow transition-transform duration-200 ${notifications ? 'translate-x-5' : 'translate-x-0.5'
                                        }`}
                                />
                            </button>
                        </div>
                    ))}
                </div>
            </motion.div>

            {/* Language */}
            <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.24 }}
                className="bg-surface border border-border rounded-2xl p-6"
            >
                <h2 className="text-lg font-bold mb-5 flex items-center gap-2">
                    <Globe size={20} className="text-primary-500" /> Language & Region
                </h2>

                <div className="flex items-center justify-between py-3">
                    <div>
                        <div className="text-sm font-medium">Language</div>
                        <div className="text-xs text-secondary mt-0.5">Choose your preferred language</div>
                    </div>
                    <select
                        value={language}
                        onChange={(e) => setLanguage(e.target.value)}
                        className="w-40 text-sm"
                    >
                        <option value="en">English</option>
                        <option value="es">Español</option>
                        <option value="fr">Français</option>
                        <option value="de">Deutsch</option>
                        <option value="hi">हिन्दी</option>
                    </select>
                </div>
            </motion.div>

            {/* Privacy */}
            <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.32 }}
                className="bg-surface border border-border rounded-2xl p-6"
            >
                <h2 className="text-lg font-bold mb-5 flex items-center gap-2">
                    <Shield size={20} className="text-primary-500" /> Privacy & Security
                </h2>

                <div className="space-y-4">
                    <div className="flex items-center justify-between py-2">
                        <div>
                            <div className="text-sm font-medium">Two-factor authentication</div>
                            <div className="text-xs text-secondary mt-0.5">Add extra security to your account</div>
                        </div>
                        <button className="btn btn-ghost text-xs px-3 py-1.5">Enable</button>
                    </div>
                    <div className="flex items-center justify-between py-2">
                        <div>
                            <div className="text-sm font-medium">Active sessions</div>
                            <div className="text-xs text-secondary mt-0.5">Manage signed-in devices</div>
                        </div>
                        <button className="btn btn-ghost text-xs px-3 py-1.5">View</button>
                    </div>
                </div>
            </motion.div>

            {/* Save */}
            <div className="flex justify-end pb-8">
                <button onClick={handleSave} className="btn btn-primary gap-2">
                    {saved ? <><CheckCircle2 size={16} /> Saved!</> : <><Save size={16} /> Save Settings</>}
                </button>
            </div>
        </div>
    );
}
