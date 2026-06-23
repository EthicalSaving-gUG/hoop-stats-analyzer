import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { UploadCloud, Play, Square, Activity, Cpu } from 'lucide-react';
import * as ort from 'onnxruntime-web';
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';

const ClientAIBenchmark = ({ onBack }) => {
    const [status, setStatus] = useState('models_unloaded'); // models_unloaded, loading, ready, running
    const [loadTime, setLoadTime] = useState(0);
    const [fps, setFps] = useState(0);
    const [inferenceTime, setInferenceTime] = useState(0);
    
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const yoloSession = useRef(null);
    const poseLandmarker = useRef(null);
    const lastTimeRef = useRef(0);
    const framesRef = useRef(0);
    
    const [videoUrl, setVideoUrl] = useState(null);

    useEffect(() => {
        return () => {
            if (yoloSession.current) {
                // Cannot strictly 'close' in all ORT versions, just let it GC
            }
            if (poseLandmarker.current) {
                poseLandmarker.current.close();
            }
        };
    }, []);

    const loadModels = async () => {
        setStatus('loading');
        const startTime = performance.now();

        try {
            // 1. Load YOLOv8 ONNX
            console.log("Loading YOLO ONNX model...");
            yoloSession.current = await ort.InferenceSession.create('/models/yolov8n.onnx', {
                executionProviders: ['wasm']
            });

            // 2. Load MediaPipe Pose
            console.log("Loading MediaPipe Pose...");
            const vision = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
            );
            
            poseLandmarker.current = await PoseLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: '/models/pose_landmarker.task',
                    delegate: "GPU"
                },
                runningMode: "VIDEO",
                numPoses: 2
            });

            const endTime = performance.now();
            setLoadTime(((endTime - startTime) / 1000).toFixed(2));
            setStatus('ready');
            
        } catch (err) {
            console.error("Failed to load models:", err);
            setStatus('error');
        }
    };

    const handleFileSelect = (e) => {
        if (e.target.files && e.target.files[0]) {
            const url = URL.createObjectURL(e.target.files[0]);
            setVideoUrl(url);
        }
    };

    // Preprocessing for YOLO (640x640)
    const preprocess = (canvas) => {
        // Draw image stretched to 640x640
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 640;
        tempCanvas.height = 640;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(canvas, 0, 0, 640, 640);
        
        const imageData = tempCtx.getImageData(0, 0, 640, 640).data;
        const float32Data = new Float32Array(3 * 640 * 640);

        // Convert RGBA to RGB and HWC to CHW format, normalize to [0,1]
        for (let i = 0, j = 0; i < imageData.length; i += 4, j++) {
            float32Data[j] = imageData[i] / 255.0; // R
            float32Data[j + 640 * 640] = imageData[i + 1] / 255.0; // G
            float32Data[j + 2 * 640 * 640] = imageData[i + 2] / 255.0; // B
        }
        return new ort.Tensor('float32', float32Data, [1, 3, 640, 640]);
    };

    const processFrame = async (now) => {
        if (!videoRef.current || !canvasRef.current || status !== 'running') return;
        
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        
        // Match canvas to video dimensions
        if (canvas.width !== video.videoWidth) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }

        const t0 = performance.now();

        // 1. Draw current video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        try {
            // 2. YOLO Inference
            const tensor = preprocess(canvas);
            const feeds = {};
            feeds[yoloSession.current.inputNames[0]] = tensor;
            const yoloResults = await yoloSession.current.run(feeds);
            const output = yoloResults[yoloSession.current.outputNames[0]].data;
            
            // Simple YOLO parsing for PoC (output shape: [1, 84, 8400])
            // Just draw anything with confidence > 0.5 (no NMS to keep it fast/simple)
            const numClasses = 80;
            const numBoxes = 8400;
            ctx.strokeStyle = 'orange';
            ctx.lineWidth = 2;
            
            for (let i = 0; i < numBoxes; i++) {
                let maxClassConf = 0;
                for (let c = 0; c < numClasses; c++) {
                    const conf = output[(4 + c) * numBoxes + i];
                    if (conf > maxClassConf) {
                        maxClassConf = conf;
                    }
                }
                
                if (maxClassConf > 0.4) {
                    const cx = output[0 * numBoxes + i] * (canvas.width / 640);
                    const cy = output[1 * numBoxes + i] * (canvas.height / 640);
                    const w = output[2 * numBoxes + i] * (canvas.width / 640);
                    const h = output[3 * numBoxes + i] * (canvas.height / 640);
                    
                    ctx.strokeRect(cx - w/2, cy - h/2, w, h);
                }
            }

            // 3. MediaPipe Pose Inference
            const poseResult = await poseLandmarker.current.detectForVideo(video, performance.now());
            
            if (poseResult.landmarks) {
                ctx.fillStyle = '#10b981'; // emerald-500
                for (const pose of poseResult.landmarks) {
                    for (const lm of pose) {
                        ctx.beginPath();
                        ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 3, 0, 2 * Math.PI);
                        ctx.fill();
                    }
                }
            }

        } catch (e) {
            console.error("Inference error:", e);
        }

        const t1 = performance.now();
        setInferenceTime((t1 - t0).toFixed(1));

        // FPS Calculation
        framesRef.current++;
        if (now - lastTimeRef.current >= 1000) {
            setFps(framesRef.current);
            framesRef.current = 0;
            lastTimeRef.current = now;
        }

        if (status === 'running' && !video.paused) {
            video.requestVideoFrameCallback(processFrame);
        }
    };

    const toggleProcessing = () => {
        if (!videoRef.current) return;
        
        if (status === 'running') {
            videoRef.current.pause();
            setStatus('ready');
        } else {
            videoRef.current.play();
            setStatus('running');
            lastTimeRef.current = performance.now();
            framesRef.current = 0;
            videoRef.current.requestVideoFrameCallback(processFrame);
        }
    };

    return (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="w-full max-w-6xl space-y-6 relative z-10">
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h2 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">Browser-KI Benchmark</h2>
                    <p className="text-slate-400 mt-1">Proof of Concept: Läuft die KI komplett in deinem Browser?</p>
                </div>
                <button onClick={onBack} className="text-sm text-slate-400 hover:text-white px-4 py-2 bg-slate-800 rounded-lg">Beenden</button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                
                {/* Control Panel */}
                <div className="lg:col-span-1 space-y-6">
                    {/* Step 1: Load Models */}
                    <div className="glass p-5">
                        <h3 className="font-semibold text-lg mb-3 flex items-center gap-2"><Cpu className="w-5 h-5 text-indigo-400"/> 1. KI initialisieren</h3>
                        {status === 'models_unloaded' && (
                            <button onClick={loadModels} className="w-full py-3 bg-indigo-600 hover:bg-indigo-700 font-semibold rounded-lg transition">
                                Modelle herunterladen
                            </button>
                        )}
                        {status === 'loading' && (
                            <div className="text-center py-3">
                                <Activity className="w-6 h-6 animate-pulse text-indigo-400 mx-auto mb-2" />
                                <span className="text-sm text-slate-300">Lade ~20MB an KI-Modellen in den Browser (einmalig)...</span>
                            </div>
                        )}
                        {(status === 'ready' || status === 'running') && (
                            <div className="bg-emerald-500/10 border border-emerald-500/30 p-3 rounded-lg flex justify-between items-center">
                                <span className="text-emerald-400 font-medium">Bereit</span>
                                <span className="text-xs text-slate-400 font-mono">Geladen in {loadTime}s</span>
                            </div>
                        )}
                    </div>

                    {/* Step 2: Select Video */}
                    <div className={`glass p-5 transition ${status === 'models_unloaded' ? 'opacity-50 pointer-events-none' : ''}`}>
                        <h3 className="font-semibold text-lg mb-3 flex items-center gap-2"><UploadCloud className="w-5 h-5 text-blue-400"/> 2. Video wählen</h3>
                        <p className="text-xs text-slate-400 mb-3">Kein Server-Upload! Das Video wird nur lokal verarbeitet.</p>
                        <label className="block w-full py-3 text-center border-2 border-dashed border-slate-600 hover:border-blue-500 cursor-pointer rounded-lg transition">
                            <span className="text-sm font-medium">{videoUrl ? 'Anderes Video' : 'Video auswählen'}</span>
                            <input type="file" accept="video/*" className="hidden" onChange={handleFileSelect} />
                        </label>
                    </div>

                    {/* Benchmark Results */}
                    <div className={`glass p-5 transition ${(status !== 'running' && fps === 0) ? 'opacity-50' : ''}`}>
                        <h3 className="font-semibold text-lg mb-4 text-orange-400">Live Metriken</h3>
                        <div className="space-y-4">
                            <div>
                                <div className="text-xs text-slate-400 uppercase tracking-widest mb-1">Bilder pro Sekunde (FPS)</div>
                                <div className="text-4xl font-mono font-bold text-white flex items-baseline gap-2">
                                    {fps} <span className="text-sm font-normal text-slate-500">FPS</span>
                                </div>
                            </div>
                            <div>
                                <div className="text-xs text-slate-400 uppercase tracking-widest mb-1">Inferenzzeit / Frame</div>
                                <div className="text-2xl font-mono font-bold text-slate-200">
                                    {inferenceTime}ms
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Video Player Area */}
                <div className="lg:col-span-3 glass p-4 flex flex-col min-h-[500px]">
                    <div className="relative flex-grow bg-black rounded-lg overflow-hidden border border-slate-700 flex items-center justify-center">
                        {!videoUrl ? (
                            <div className="text-slate-500">Bitte wähle ein Video aus.</div>
                        ) : (
                            <>
                                <video 
                                    ref={videoRef} 
                                    src={videoUrl} 
                                    className="absolute inset-0 w-full h-full object-contain invisible" 
                                    crossOrigin="anonymous"
                                    loop
                                    muted
                                />
                                <canvas 
                                    ref={canvasRef} 
                                    className="absolute inset-0 w-full h-full object-contain"
                                />
                            </>
                        )}
                    </div>
                    
                    {/* Controls */}
                    <div className="mt-4 flex justify-center">
                        <button 
                            onClick={toggleProcessing}
                            disabled={!videoUrl || status === 'loading' || status === 'models_unloaded'}
                            className="px-8 py-3 bg-gradient-to-r from-orange-500 to-rose-500 hover:from-orange-400 hover:to-rose-400 disabled:opacity-50 rounded-xl font-bold flex items-center gap-2 transition"
                        >
                            {status === 'running' ? (
                                <><Square className="w-5 h-5 fill-current" /> Analyse stoppen</>
                            ) : (
                                <><Play className="w-5 h-5 fill-current" /> Analyse starten</>
                            )}
                        </button>
                    </div>
                </div>

            </div>
        </motion.div>
    );
};

export default ClientAIBenchmark;
