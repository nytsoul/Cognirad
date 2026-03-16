import React from 'react';
import { Navigate } from 'react-router-dom';
import { Brain } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex flex-col items-center justify-center gap-4">
        <div className="w-14 h-14 bg-primary-600 rounded-2xl flex items-center justify-center text-white">
          <Brain size={28} className="animate-pulse" />
        </div>
        <p className="text-secondary text-sm font-medium">Loading CogniRad++...</p>
      </div>
    );
  }

  if (!user) return <Navigate to="/login" replace />;
  return children;
}
