import React, { useState } from 'react';
import { Eye, EyeOff, Maximize2, Activity } from 'lucide-react';
import clsx from 'clsx';

const ImageViewer = ({ imageUrl, attentionMap, isLoading, perceptionLayers = [] }) => {
    const [showAttention, setShowAttention] = useState(false);
    const [showPerception, setShowPerception] = useState(true);
    const [isZoomed, setIsZoomed] = useState(false);
    const [hoveredLayer, setHoveredLayer] = useState(null);

    if (!imageUrl) return null;

    return (
        <div className={clsx(
            "relative rounded-xl overflow-hidden bg-black aspect-[3/4] group transition-all duration-300",
            isZoomed ? "fixed inset-4 z-50 shadow-2xl" : "w-full shadow-md"
        )}>
            {/* Controls Overlay */}
            <div className="absolute top-4 right-4 z-20 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                {perceptionLayers.length > 0 && (
                    <button
                        onClick={() => setShowPerception(!showPerception)}
                        className={clsx(
                            "p-2 rounded-lg backdrop-blur-sm transition-colors",
                            showPerception ? "bg-primary-500 text-white" : "bg-black/60 text-white hover:bg-black/80"
                        )}
                        title="Toggle Radiological Perception (PRO-FA)"
                    >
                        <Activity size={20} />
                    </button>
                )}
                {attentionMap && (
                    <button
                        onClick={() => setShowAttention(!showAttention)}
                        className={clsx(
                            "p-2 rounded-lg backdrop-blur-sm transition-colors",
                            showAttention ? "bg-amber-500 text-white" : "bg-black/60 text-white hover:bg-black/80"
                        )}
                        title="Toggle Attention Map"
                    >
                        {showAttention ? <EyeOff size={20} /> : <Eye size={20} />}
                    </button>
                )}
                <button
                    onClick={() => setIsZoomed(!isZoomed)}
                    className="p-2 bg-black/60 hover:bg-black/80 text-white rounded-lg backdrop-blur-sm transition-colors"
                    title={isZoomed ? "Minimize" : "Maximize"}
                >
                    <Maximize2 size={20} />
                </button>
            </div>

            {/* Content Container (to maintain aspect ratio relative to image) */}
            <div className="relative w-full h-full flex items-center justify-center">
                {/* Main Image */}
                <img
                    src={imageUrl}
                    alt="Chest X-ray"
                    className="max-w-full max-h-full object-contain"
                />

                {/* Perception Layers (SVG Overlay) */}
                {showPerception && perceptionLayers.length > 0 && (
                    <svg
                        viewBox="0 0 400 400"
                        className="absolute inset-0 w-full h-full pointer-events-auto"
                        preserveAspectRatio="xMidYMid meet"
                    >
                        {perceptionLayers.map((layer, idx) => (
                            <path
                                key={layer.name}
                                d={layer.path}
                                fill={hoveredLayer === layer.name ? "rgba(14, 165, 233, 0.2)" : "transparent"}
                                stroke={hoveredLayer === layer.name ? "#0ea5e9" : "rgba(14, 165, 233, 0.4)"}
                                strokeWidth="2"
                                className="transition-all duration-200 cursor-help"
                                onMouseEnter={() => setHoveredLayer(layer.name)}
                                onMouseLeave={() => setHoveredLayer(null)}
                            />
                        ))}
                    </svg>
                )}

                {/* Attention Map Overlay */}
                {showAttention && attentionMap && (
                    <div className="absolute inset-0 pointer-events-none mix-blend-overlay opacity-60">
                        <img
                            src={attentionMap}
                            alt="Attention Heatmap"
                            className="w-full h-full object-contain"
                        />
                    </div>
                )}
            </div>

            {/* Layer Info Tooltip */}
            {hoveredLayer && (
                <div className="absolute bottom-16 left-4 bg-slate-900/90 text-white px-3 py-1.5 rounded-lg text-xs font-bold border border-white/10 backdrop-blur-sm z-30">
                    PRO-FA: {hoveredLayer} Concepts Detected
                </div>
            )}

            {/* Loading Overlay */}
            {isLoading && (
                <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px] flex items-center justify-center z-10">
                    <div className="flex flex-col items-center">
                        <div className="w-10 h-10 border-4 border-white/30 border-t-white rounded-full animate-spin mb-3"></div>
                        <span className="text-white font-medium tracking-wide text-sm">Cognitive Processing...</span>
                    </div>
                </div>
            )}

            {/* Label */}
            <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent p-4 pt-12">
                <div className="flex items-center justify-between">
                    <div>
                        <span className="text-white/90 text-sm font-medium">Input Radiograph</span>
                        {showAttention && <span className="ml-2 text-amber-300 text-[10px] font-bold uppercase tracking-wider bg-amber-950/50 px-1.5 py-0.5 rounded leading-none">• Attention active</span>}
                        {showPerception && <span className="ml-2 text-primary-300 text-[10px] font-bold uppercase tracking-wider bg-primary-950/50 px-1.5 py-0.5 rounded leading-none">• PRO-FA active</span>}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ImageViewer;
