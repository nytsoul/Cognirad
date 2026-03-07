import React, { createContext, useContext, useEffect, useState } from 'react';

const AuthContext = createContext(null);

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => localStorage.getItem('cognirad-token'));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (token) {
      fetch(`${API_URL}/api/auth/profile`, {
        headers: { Authorization: `Bearer ${token}` },
      })
        .then((r) => (r.ok ? r.json() : Promise.reject()))
        .then((d) => setUser(d.user))
        .catch(() => {
          localStorage.removeItem('cognirad-token');
          setToken(null);
        })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, [token]);

  const saveToken = (t) => {
    localStorage.setItem('cognirad-token', t);
    setToken(t);
  };

  const login = async (email, password) => {
    const r = await fetch(`${API_URL}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Login failed');
    saveToken(d.token);
    setUser(d.user);
    return d.user;
  };

  const loginWithGoogle = async (credential) => {
    const r = await fetch(`${API_URL}/api/auth/google`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: credential }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Google auth failed');
    saveToken(d.token);
    setUser(d.user);
    return d.user;
  };

  const register = async ({ name, email, password, role }) => {
    const r = await fetch(`${API_URL}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email, password, role }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Registration failed');
    saveToken(d.token);
    setUser(d.user);
    return d.user;
  };

  const updateProfile = async (payload) => {
    const r = await fetch(`${API_URL}/api/auth/profile`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(payload),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Update failed');
    setUser(d.user);
    return d.user;
  };

  const changePassword = async (current_password, new_password) => {
    const r = await fetch(`${API_URL}/api/auth/change-password`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ current_password, new_password }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Password change failed');
    return d;
  };

  const logout = () => {
    localStorage.removeItem('cognirad-token');
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        loading,
        login,
        loginWithGoogle,
        register,
        updateProfile,
        changePassword,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
