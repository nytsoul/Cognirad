import React, { useState } from 'react';
import { Outlet, useLocation, Link, useNavigate } from 'react-router-dom';
import {
  Brain, Home, Upload, BarChart3, FileText, Microscope, Users,
  Settings, Bell, User, Search, Menu, X, Zap, Shield, Moon, Sun, LogOut
} from 'lucide-react';
import { motion } from 'framer-motion';
import { useTheme } from '../context/ThemeContext';
import { useAuth } from '../context/AuthContext';

function Layout() {
  const location = useLocation();
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const { theme, toggleTheme } = useTheme();
  const { user, logout } = useAuth();

  const navigation = [
    { icon: Home, label: 'Dashboard', href: '/dashboard' },
    { icon: Upload, label: 'Upload Scan', href: '/upload' },
    { icon: BarChart3, label: 'Patient Reports', href: '/patient-reports' },
    { icon: User, label: 'Profile', href: '/profile' },
    { icon: Settings, label: 'Settings', href: '/settings' },
  ];

  const isActive = (path) => location.pathname === path;

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const initials = (user?.name || 'U')
    .split(' ')
    .map((w) => w[0])
    .join('')
    .slice(0, 2)
    .toUpperCase();

  return (
    <div className="min-h-screen bg-background text-primary transition-colors duration-300">
      {/* Sidebar */}
      <motion.aside
        initial={{ x: -300 }}
        animate={{ x: 0 }}
        className={`fixed left-0 top-0 h-screen bg-surface border-r border-border transition-all duration-300 z-40 ${sidebarOpen ? 'w-64' : 'w-20'
          }`}
      >
        <div className="p-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-primary-600 rounded-xl flex items-center justify-center text-white">
              <Brain size={24} />
            </div>
            {sidebarOpen && (
              <div>
                <h1 className="text-lg font-bold text-primary">CogniRad++</h1>
                <span className="text-[9px] text-secondary font-bold uppercase tracking-widest">AI Radiology</span>
              </div>
            )}
          </div>
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-1 hover:bg-surface-secondary rounded-lg transition-colors"
          >
            {sidebarOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>

        {/* Nav Items */}
        <nav className="mt-8 px-3 space-y-2">
          {navigation.map((item) => (
            <Link
              key={item.href}
              to={item.href}
              className={`flex items-center gap-4 px-4 py-3 rounded-xl transition-all no-underline hover:no-underline ${isActive(item.href)
                  ? 'bg-primary-600 text-white'
                  : 'text-secondary hover:bg-surface-secondary'
                }`}
              title={!sidebarOpen ? item.label : ''}
            >
              <item.icon size={20} />
              {sidebarOpen && <span className="text-sm font-medium">{item.label}</span>}
            </Link>
          ))}
        </nav>

        {/* Bottom – Logout */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-surface to-transparent p-6 space-y-3">
          <button
            onClick={handleLogout}
            className={`flex items-center gap-4 w-full px-4 py-3 rounded-xl text-red-500 hover:bg-red-500/10 transition-all ${sidebarOpen ? '' : 'justify-center'
              }`}
            title="Logout"
          >
            <LogOut size={20} />
            {sidebarOpen && <span className="text-sm font-medium">Logout</span>}
          </button>
          {sidebarOpen && (
            <div className="text-center text-[9px] text-secondary font-bold">
              <div className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-success/10 dark:bg-success/10 border border-success/20 dark:border-success/20 rounded-full">
                <div className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />
                <span className="uppercase tracking-widest">Online</span>
              </div>
            </div>
          )}
        </div>
      </motion.aside>

      {/* Top Bar */}
      <header className="fixed top-0 right-0 left-0 h-20 bg-surface/80 backdrop-blur border-b border-border z-30 transition-colors duration-300"
        style={{ marginLeft: sidebarOpen ? '256px' : '80px' }}>
        <div className="h-full flex items-center justify-between px-8">
          {/* Search */}
          <div className="flex-1 max-w-xl hidden lg:flex items-center gap-3 px-4 py-2.5 bg-surface-secondary border border-border rounded-xl transition-colors duration-300">
            <Search size={16} className="text-secondary" />
            <input
              type="text"
              placeholder="Search patient ID or scan..."
              className="bg-transparent outline-none text-sm w-full placeholder:text-tertiary text-primary transition-colors duration-300"
            />
          </div>

          {/* Right Controls */}
          <div className="flex items-center gap-4 ml-auto">
            <button
              onClick={toggleTheme}
              className="p-2.5 hover:bg-surface-secondary rounded-xl transition-colors"
              title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            >
              {theme === 'dark' ? (
                <Sun size={18} className="text-secondary" />
              ) : (
                <Moon size={18} className="text-secondary" />
              )}
            </button>
            <button className="p-2.5 hover:bg-surface-secondary rounded-xl transition-colors">
              <Bell size={18} className="text-secondary" />
            </button>
            <Link to="/settings" className="p-2.5 hover:bg-surface-secondary rounded-xl transition-colors no-underline">
              <Settings size={18} className="text-secondary" />
            </Link>
            <Link to="/profile" className="pl-4 border-l border-border flex items-center gap-3 transition-colors duration-300 no-underline hover:no-underline">
              <div className="hidden sm:block text-right">
                <div className="text-xs font-bold text-primary">{user?.name || 'User'}</div>
                <div className="text-[9px] text-secondary capitalize">{user?.role || 'Radiologist'}</div>
              </div>
              <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-accent rounded-full flex items-center justify-center text-white font-bold text-sm">
                {initials}
              </div>
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main
        className={`transition-all duration-300 ${sidebarOpen ? 'ml-64' : 'ml-20'} mt-20`}
      >
        <div className="p-8 bg-background text-primary transition-colors duration-300">
          <Outlet />
        </div>
      </main>
    </div>
  );
}

export default Layout;
