import React, { useCallback, useState } from 'react';
import { UploadCloud, File, X, AlertCircle } from 'lucide-react';
import clsx from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';

const UploadZone = ({ onFileSelect, isProcessing }) => {
    const [dragActive, setDragActive] = useState(false);
    const [preview, setPreview] = useState(null);
    const [error, setError] = useState(null);

    const handleDrag = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    }, []);

    const validateFile = (file) => {
        if (!file.type.startsWith('image/')) {
            setError("Please upload an image file (JPEG, PNG).");
            return false;
        }
        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            setError("File size too large (max 10MB).");
            return false;
        }
        setError(null);
        return true;
    };

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const file = e.dataTransfer.files[0];
            if (validateFile(file)) {
                handleFile(file);
            }
        }
    }, [onFileSelect]);

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            if (validateFile(file)) {
                handleFile(file);
            }
        }
    };

    const handleFile = (file) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            setPreview(reader.result);
        };
        reader.readAsDataURL(file);
        onFileSelect(file);
    };

    const clearFile = () => {
        setPreview(null);
        setError(null);
        onFileSelect(null);
    };

    return (
        <div className="w-full max-w-xl mx-auto">
            <AnimatePresence>
                {error && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        className="mb-4 p-3 bg-red-50 text-red-600 rounded-lg flex items-center gap-2 text-sm"
                    >
                        <AlertCircle size={16} />
                        {error}
                    </motion.div>
                )}
            </AnimatePresence>

            <div
                className={clsx(
                    "relative border-2 border-dashed rounded-2xl p-10 transition-all duration-300 ease-in-out flex flex-col items-center justify-center min-h-[350px] group",
                    dragActive ? "border-primary-500 bg-primary-500/5 scale-[1.02] shadow-glow" : "border-border hover:border-primary-500/50 bg-surface/50 backdrop-blur-sm",
                    isProcessing && "opacity-50 pointer-events-none"
                )}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
            >
                <input
                    type="file"
                    id="file-upload"
                    className="hidden"
                    accept="image/*"
                    onChange={handleChange}
                    disabled={isProcessing}
                />

                {preview ? (
                    <div className="w-full h-full flex flex-col items-center">
                        <div className="relative w-full max-h-[450px] overflow-hidden rounded-xl shadow-glass border border-white/5">
                            <img src={preview} alt="Preview" className="w-full h-full object-contain" />
                            <button
                                onClick={(e) => { e.stopPropagation(); clearFile(); }}
                                className="absolute top-4 right-4 p-2 bg-black/60 hover:bg-black/80 backdrop-blur-md rounded-full text-white/70 hover:text-danger transition-colors border border-white/10"
                            >
                                <X size={20} />
                            </button>
                        </div>
                        <div className="mt-6 text-[10px] uppercase font-black tracking-[0.2em] text-primary-400 flex items-center gap-3">
                            <div className="w-1.5 h-1.5 rounded-full bg-primary-500 animate-pulse shadow-glow" />
                            Visual Data Locked
                        </div>
                    </div>
                ) : (
                    <label htmlFor="file-upload" className="flex flex-col items-center cursor-pointer w-full h-full justify-center p-4">
                        <div className={clsx(
                            "p-5 rounded-2xl mb-6 transition-all duration-300",
                            dragActive ? "bg-primary-500 text-white shadow-glow" : "bg-white/5 text-slate-500 group-hover:bg-primary-500/10 group-hover:text-primary-400"
                        )}>
                            <UploadCloud size={48} />
                        </div>
                        <p className="text-2xl font-black text-white mb-2 tracking-tight">
                            {dragActive ? "Drop Digital Radiograph" : "Upload Case Data"}
                        </p>
                        <p className="text-sm text-slate-500 text-center max-w-xs leading-relaxed">
                            Drag and drop high-res X-ray imagery or <span className="text-primary-500 font-bold">browse workstation</span>
                        </p>
                        <div className="mt-8 px-6 py-2.5 bg-white/5 border border-white/10 rounded-xl text-xs font-black uppercase tracking-widest text-slate-400 hover:bg-white/10 hover:text-white transition-all">
                            Select DICOM/Image
                        </div>
                    </label>
                )}
            </div>
        </div>
    );
};

export default UploadZone;
