import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import UploadScan from './pages/UploadScan';
import Analysis from './pages/Analysis';
import Report from './pages/Report';
import EvidenceMap from './pages/EvidenceMap';
import PatientReports from './pages/PatientReports';

function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<UploadScan />} />
            <Route path="/analysis/:caseId" element={<Analysis />} />
            <Route path="/report/:caseId" element={<Report />} />
            <Route path="/evidence/:caseId" element={<EvidenceMap />} />
            <Route path="/patient-reports" element={<PatientReports />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
