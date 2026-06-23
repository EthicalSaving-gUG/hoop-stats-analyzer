import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, Timer, Clock, Users, Trophy, Mic, MicOff, Video, RefreshCw } from 'lucide-react';
import axios from 'axios';
import ShotChart from './ShotChart';
import { API_BASE } from '../config';

const BiometricsCanvas = ({ poseModel, mesh, skeleton }) => {
    const canvasRef = React.useRef(null);
    const angleRef = React.useRef(0);

    React.useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        let animationFrame;

        const draw = () => {
            if (!ctx || !canvas) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw a futuristic grid & brackets background
            const radGrad = ctx.createRadialGradient(
                canvas.width / 2, canvas.height / 2, 5,
                canvas.width / 2, canvas.height / 2, 50
            );
            radGrad.addColorStop(0, 'rgba(99, 102, 241, 0.08)');
            radGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
            ctx.fillStyle = radGrad;
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // subtle coordinate grid
            ctx.strokeStyle = 'rgba(99, 102, 241, 0.04)';
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(0, canvas.height / 2);
            ctx.lineTo(canvas.width, canvas.height / 2);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(canvas.width / 2, 0);
            ctx.lineTo(canvas.width / 2, canvas.height);
            ctx.stroke();

            // Subtle corner brackets
            const bracketSize = 6;
            const margin = 4;
            ctx.strokeStyle = 'rgba(99, 102, 241, 0.2)';
            ctx.lineWidth = 1;

            ctx.beginPath();
            ctx.moveTo(margin + bracketSize, margin); ctx.lineTo(margin, margin); ctx.lineTo(margin, margin + bracketSize);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(canvas.width - margin - bracketSize, margin); ctx.lineTo(canvas.width - margin, margin); ctx.lineTo(canvas.width - margin, margin + bracketSize);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(margin + bracketSize, canvas.height - margin); ctx.lineTo(margin, canvas.height - margin); ctx.lineTo(margin, canvas.height - margin - bracketSize);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(canvas.width - margin - bracketSize, canvas.height - margin); ctx.lineTo(canvas.width - margin, canvas.height - margin); ctx.lineTo(canvas.width - margin, canvas.height - margin - bracketSize);
            ctx.stroke();

            // Update rotation angle slightly over time
            angleRef.current += 0.015;
            const cosA = Math.cos(angleRef.current);
            const sinA = Math.sin(angleRef.current);

            // Handle rendering based on active data
            if (poseModel === 'densepose' && mesh && mesh.length > 0) {
                // Compute centroid to center the mesh in the canvas
                let sumX = 0, sumY = 0, count = mesh.length;
                mesh.forEach(pt => {
                    sumX += pt[0];
                    sumY += pt[1];
                });
                const avgX = sumX / count;
                const avgY = sumY / count;

                // Part colors matching standard DensePose segments
                const partColors = {
                    1: '#3b82f6', // blue-500
                    2: '#60a5fa', // blue-400
                    3: '#10b981', // emerald-500
                    4: '#34d399', // emerald-400
                    5: '#f59e0b', // amber-500
                    6: '#fbbf24', // amber-400
                    7: '#ec4899', // pink-500
                    8: '#f472b6', // pink-400
                    9: '#8b5cf6', // violet-500
                    10: '#a78bfa', // violet-400
                    11: '#f43f5e', // rose-500
                };

                // Project 3D points [x, y, z] to 2D canvas coordinates
                mesh.forEach(pt => {
                    const [rawX, rawY, rawZ, partId] = pt;

                    // Centering
                    const cx = rawX - avgX;
                    const cy = rawY - avgY;
                    const cz = rawZ;

                    // Rotate around Y-axis for 3D depth perspective
                    const rotX = cx * cosA - cz * sinA;
                    const rotZ = cx * sinA + cz * cosA;

                    // Orthographic projection with scaling
                    const scale = 220; // scaling factor
                    const px = canvas.width / 2 + rotX * scale;
                    const py = canvas.height / 2 + cy * scale;

                    // Calculate depth-based opacity and size
                    const depthAlpha = Math.max(0.3, Math.min(1, 0.8 + rotZ * 4));
                    ctx.globalAlpha = depthAlpha;

                    // Select part color
                    ctx.fillStyle = partColors[partId] || '#4f46e5'; // indigo-600 default

                    // Draw mesh node
                    ctx.beginPath();
                    ctx.arc(px, py, 1.2, 0, 2 * Math.PI);
                    ctx.fill();
                });
            } else if (poseModel === 'mediapipe' && skeleton && skeleton.landmarks && skeleton.landmarks.length > 0) {
                // Center the skeleton in the canvas
                let sumX = 0, sumY = 0, count = skeleton.landmarks.length;
                skeleton.landmarks.forEach(lm => {
                    sumX += lm[0];
                    sumY += lm[1];
                });
                const avgX = sumX / count;
                const avgY = sumY / count;

                // Project all 3D landmarks
                const projected = skeleton.landmarks.map((lm, idx) => {
                    const cx = lm[0] - avgX;
                    const cy = lm[1] - avgY;
                    const cz = lm[2];

                    // Rotate around Y axis
                    const rotX = cx * cosA - cz * sinA;
                    const rotZ = cx * sinA + cz * cosA;

                    // Project
                    const scale = 220; // scaling factor
                    const px = canvas.width / 2 + rotX * scale;
                    const py = canvas.height / 2 + cy * scale;

                    return { px, py, rotZ, idx };
                });

                // Draw skeleton connection lines
                ctx.strokeStyle = 'rgba(99, 102, 241, 0.45)';
                ctx.lineWidth = 1.5;
                skeleton.connections.forEach(([startIdx, endIdx]) => {
                    const startPt = projected[startIdx];
                    const endPt = projected[endIdx];
                    if (startPt && endPt) {
                        ctx.beginPath();
                        ctx.moveTo(startPt.px, startPt.py);
                        ctx.lineTo(endPt.px, endPt.py);
                        ctx.stroke();
                    }
                });

                // Draw glowing joint nodes
                projected.forEach(({ px, py, rotZ }) => {
                    const depthAlpha = Math.max(0.4, Math.min(1, 0.7 + rotZ * 5));
                    ctx.globalAlpha = depthAlpha;

                    // Fill glowing cyan dot
                    ctx.fillStyle = '#22d3ee'; // cyan-400
                    ctx.beginPath();
                    ctx.arc(px, py, 3, 0, 2 * Math.PI);
                    ctx.fill();

                    // Outer glow circle
                    ctx.strokeStyle = 'rgba(34, 211, 238, 0.25)';
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.arc(px, py, 5.5, 0, 2 * Math.PI);
                    ctx.stroke();
                });
            } else {
                // Draw a placeholder wireframe outline
                ctx.strokeStyle = 'rgba(99, 102, 241, 0.12)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(canvas.width / 2, canvas.height / 2, 40, 0, 2 * Math.PI);
                ctx.stroke();

                ctx.fillStyle = 'rgba(99, 102, 241, 0.25)';
                ctx.font = '8px monospace';
                ctx.fillText("WAITING FOR AI", canvas.width / 2 - 38, canvas.height / 2 + 3);
            }

            ctx.globalAlpha = 1.0;
        };

        const renderLoop = () => {
            draw();
            animationFrame = requestAnimationFrame(renderLoop);
        };

        renderLoop();
        return () => cancelAnimationFrame(animationFrame);
    }, [mesh, skeleton, poseModel]);

    return (
        <div className="relative w-28 h-28 bg-slate-950/60 rounded-xl border border-slate-800/80 overflow-hidden flex items-center justify-center shrink-0">
            <canvas ref={canvasRef} width={112} height={112} className="w-full h-full" />
            <div className="absolute bottom-1 right-1.5 text-[7px] font-mono text-slate-500 flex items-center gap-1 select-none pointer-events-none">
                <span className="w-1 h-1 rounded-full bg-indigo-500 animate-ping" />
                {poseModel === 'densepose' ? '3D MESH' : '3D SKELETON'}
            </div>
        </div>
    );
};

const LiveDashboard = ({ onStop }) => {
    const [state, setState] = useState(null);
    const [isListening, setIsListening] = useState(false);
    const [speechSupported, setSpeechSupported] = useState(false);
    const [lastTranscript, setLastTranscript] = useState('');
    const [voiceFeedback, setVoiceFeedback] = useState('');
    const [recognition, setRecognition] = useState(null);
    const [highlights, setHighlights] = useState([]);
    const [loadingHighlights, setLoadingHighlights] = useState(false);

    useEffect(() => {
        const interval = setInterval(async () => {
            try {
                const res = await axios.get(`${API_BASE}/game/state`);
                if (res.data.status !== 'no_game') {
                    setState(res.data);
                }
            } catch (e) { /* ignore polling errors */ }
        }, 500);
        return () => clearInterval(interval);
    }, []);

    // Fetch highlights
    const fetchHighlights = async () => {
        setLoadingHighlights(true);
        try {
            const res = await axios.get(`${API_BASE}/game/highlights`);
            if (res.data.status === 'success') {
                setHighlights(res.data.highlights);
            }
        } catch (e) {
            console.error("Failed to load highlights", e);
        } finally {
            setLoadingHighlights(false);
        }
    };

    useEffect(() => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
            setSpeechSupported(true);
            const rec = new SpeechRecognition();
            rec.continuous = true;
            rec.interimResults = false;
            rec.lang = 'de-DE';

            rec.onresult = async (event) => {
                const cmd = event.results[event.results.length - 1][0].transcript.trim().toLowerCase();
                setLastTranscript(cmd);

                // Parse the speech command to map it to backend commands
                let mappedCmd = null;
                let player = null;

                if (cmd.includes('foul heim') || cmd.includes('foul schwarz')) {
                    mappedCmd = 'foul_home';
                    const match = cmd.match(/\d+/);
                    if (match) player = match[0];
                } else if (cmd.includes('foul gast') || cmd.includes('foul gelb')) {
                    mappedCmd = 'foul_away';
                    const match = cmd.match(/\d+/);
                    if (match) player = match[0];
                } else if (cmd.includes('uhr stoppen') || cmd.includes('uhr starten') || cmd.includes('uhr stoppe') || cmd.includes('uhr starte') || cmd.includes('uhr toggle') || cmd.includes('zeit stop') || cmd.includes('zeit start')) {
                    mappedCmd = 'toggle_clock';
                } else if (cmd.includes('auszeit') || cmd.includes('time-out') || cmd.includes('timeout')) {
                    mappedCmd = 'timeout';
                } else if (cmd.includes('wechsel') || cmd.includes('auswechsel') || cmd.includes('substitution')) {
                    mappedCmd = 'sub';
                }

                if (mappedCmd) {
                    setVoiceFeedback(`Sende Befehl: "${cmd}"...`);
                    try {
                        const res = await axios.post(`${API_BASE}/game/voice-command`, { command: mappedCmd, player: player });
                        if (res.data.status === 'success') {
                            setVoiceFeedback(`Ausgeführt: ${cmd}`);
                        } else {
                            setVoiceFeedback(`Fehler beim Ausführen: ${cmd}`);
                        }
                    } catch (e) {
                        setVoiceFeedback(`Fehler bei Übertragung`);
                    }
                } else {
                    setVoiceFeedback(`Befehl nicht erkannt: "${cmd}"`);
                }
                setTimeout(() => setVoiceFeedback(''), 4000);
            };

            rec.onerror = (event) => {
                console.error("Speech recognition error", event.error);
            };

            rec.onend = () => {
                if (isListening) {
                    try { rec.start(); } catch (e) { /* ignore recorder restart errors */ }
                }
            };

            setRecognition(rec);
        }
    }, [isListening]);

    const toggleListening = () => {
        if (!recognition) return;
        if (isListening) {
            recognition.stop();
            setIsListening(false);
            setVoiceFeedback('');
        } else {
            setVoiceFeedback('Mikrofon aktiv...');
            recognition.start();
            setIsListening(true);
        }
    };

    const formatTime = (seconds) => {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    };

    if (!state) {
        return (
            <div className="text-center py-20">
                <div className="text-4xl mb-4 animate-bounce">🏀</div>
                <p className="text-slate-400">Verbindung wird hergestellt...</p>
            </div>
        );
    }

    const events = state.recent_events || [];
    const violations = events.filter(e => e.type.includes('VIOLATION') || e.type.includes('ALERT') || e.type.includes('DISQUALIFIED'));
    const liveShots = events
        .filter(e => e.type === 'SHOT_MAKE' || e.type === 'SHOT_MISS')
        .map(e => ({
            coord_m: e.data ? e.data.coord_m : null,
            status: e.type === 'SHOT_MAKE' ? 'made' : 'missed',
            shooter_id: e.data ? e.data.shooter_id : null,
            zone: e.data ? e.data.zone : 'Unknown',
            expected_points: e.data ? e.data.expected_points : null,
            shot_quality: e.data ? e.data.shot_quality : null,
            defender_dist: e.data ? e.data.defender_dist : null
        }))
        .filter(shot => shot.coord_m && Array.isArray(shot.coord_m) && shot.coord_m.length === 2);

    return (
        <div className="w-full max-w-6xl space-y-6">
            {/* Scoreboard */}
            <div className="glass p-6">
                <div className="flex items-center justify-between">
                    <div className="text-center flex-1">
                        <p className="text-slate-400 text-sm font-medium uppercase tracking-wider mb-1">{state.team_home || 'Home'}</p>
                        <p className="text-6xl font-bold text-blue-400">{state.score_home}</p>
                    </div>

                    <div className="text-center px-8">
                        <p className="text-slate-500 text-xs uppercase tracking-widest mb-2">Viertel {state.quarter}</p>
                        <div className={`text-4xl font-mono font-bold ${state.game_clock_running ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {formatTime(state.game_clock)}
                        </div>
                        <div className="mt-2 flex items-center justify-center gap-2">
                            <Timer className="w-4 h-4 text-orange-400" />
                            <span className={`text-2xl font-mono font-bold ${state.shot_clock <= 5 ? 'text-rose-500 animate-pulse' : 'text-orange-400'}`}>
                                {Math.ceil(state.shot_clock)}
                            </span>
                        </div>
                    </div>

                    <div className="text-center flex-1">
                        <p className="text-slate-400 text-sm font-medium uppercase tracking-wider mb-1">{state.team_away || 'Away'}</p>
                        <p className="text-6xl font-bold text-emerald-400">{state.score_away}</p>
                    </div>
                </div>

                {state.possession && (
                    <div className="text-center mt-4">
                        <span className="text-xs uppercase tracking-widest text-orange-400 font-semibold">
                            Ballbesitz: {state.possession === 'home' ? state.team_home : state.team_away}
                        </span>
                    </div>
                )}
            </div>

            {/* 🗣️ Voice HUD Panel */}
            <div className="glass p-5 border border-indigo-500/20 shadow-[0_0_15px_rgba(99,102,241,0.05)] rounded-2xl">
                <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                    <div className="flex items-center gap-4">
                        <motion.button
                            onClick={toggleListening}
                            disabled={!speechSupported}
                            className={`w-14 h-14 rounded-full flex items-center justify-center border-2 transition-all relative ${
                                isListening 
                                ? 'bg-red-500/10 border-red-500 shadow-[0_0_20px_rgba(239,68,68,0.4)] text-red-500 hover:bg-red-500/25' 
                                : 'bg-slate-900 border-slate-700 text-slate-400 hover:border-slate-500 hover:text-slate-300 disabled:opacity-40'
                            }`}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                        >
                            {isListening ? (
                                <>
                                    <Mic className="w-6 h-6 animate-pulse" />
                                    <span className="absolute -top-1 -right-1 flex h-3 w-3">
                                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                                        <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                                    </span>
                                </>
                            ) : (
                                <MicOff className="w-6 h-6" />
                            )}
                        </motion.button>
                        <div>
                            <h3 className="text-lg font-bold text-slate-200 flex items-center gap-2">
                                🗣️ Voice HUD (Schiedsrichter-Sprachsteuerung)
                                {isListening && (
                                    <span className="text-xs bg-red-500/15 text-red-400 border border-red-500/20 px-2 py-0.5 rounded-full animate-pulse">
                                        LIVE
                                    </span>
                                )}
                            </h3>
                            <p className="text-xs text-slate-400">
                                {speechSupported 
                                    ? 'Sagen Sie z. B. "Foul Heim 4", "Foul Gelb 12", "Auswechselung", "Auszeit" oder "Uhr stoppen".' 
                                    : 'Die Web Speech API wird in Ihrem Browser leider nicht unterstützt (Empfohlen: Chrome/Edge).'
                                }
                            </p>
                        </div>
                    </div>

                    <div className="flex-1 max-w-md w-full">
                        <AnimatePresence mode="wait">
                            {voiceFeedback ? (
                                <motion.div
                                    key="feedback"
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -10 }}
                                    className={`px-4 py-2.5 rounded-xl border text-sm font-mono text-center shadow-lg ${
                                        voiceFeedback.includes('erkannt:') 
                                        ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' 
                                        : 'bg-indigo-500/10 border-indigo-500/30 text-indigo-400'
                                    }`}
                                >
                                    {voiceFeedback}
                                </motion.div>
                            ) : lastTranscript ? (
                                <motion.div
                                    key="transcript"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 0.7 }}
                                    className="bg-slate-900/60 border border-slate-800 px-4 py-2.5 rounded-xl text-xs font-mono text-slate-400 text-center truncate"
                                >
                                    Letzte Eingabe: "{lastTranscript}"
                                </motion.div>
                            ) : (
                                <div className="border border-dashed border-slate-800 px-4 py-2.5 rounded-xl text-xs font-mono text-slate-600 text-center select-none">
                                    Warten auf Sprachbefehl...
                                </div>
                            )}
                        </AnimatePresence>
                    </div>
                </div>
            </div>

            {/* Info Row */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <InfoCard icon={<Users className="w-5 h-5 text-blue-400" />} label="Teamfouls Home" value={state.team_fouls_home?.[state.quarter] || 0} />
                <InfoCard icon={<Users className="w-5 h-5 text-emerald-400" />} label="Teamfouls Away" value={state.team_fouls_away?.[state.quarter] || 0} />
                <InfoCard icon={<Trophy className="w-5 h-5 text-orange-400" />} label="Regelwerk" value={state.ruleset === 'fiba' ? 'FIBA' : 'NBA'} />
                <InfoCard icon={<Clock className="w-5 h-5 text-purple-400" />} label="Events" value={events.length} />
            </div>

            {/* Kamera-Status */}
            {state.cameras && Object.keys(state.cameras).length > 0 && (
                <div className="glass p-5 border-l-4 border-emerald-500">
                    <h3 className="text-md font-bold text-slate-200 mb-3 flex items-center gap-2">
                        <Video className="w-5 h-5 text-emerald-400" /> 🎥 Kamera-Status (Live-Verbindungen)
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {Object.entries(state.cameras).map(([camId, camInfo]) => (
                            <div key={camId} className="bg-slate-800/30 border border-slate-700/40 p-3.5 rounded-xl flex items-center justify-between gap-4">
                                <div className="flex items-center gap-3 min-w-0">
                                    <div className={`w-3 h-3 rounded-full shrink-0 ${camInfo.connected ? 'bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)] animate-pulse' : 'bg-rose-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]'}`} />
                                    <div className="min-w-0">
                                        <p className="text-slate-200 font-semibold text-sm truncate">{camId.toUpperCase()}</p>
                                        <p className="text-slate-500 text-[10px] font-mono truncate" title={camInfo.source}>{camInfo.source}</p>
                                    </div>
                                </div>
                                <div className="text-right shrink-0">
                                    <p className="text-slate-300 text-xs font-medium font-mono">
                                        {camInfo.resolution && camInfo.resolution[0] > 0 ? `${camInfo.resolution[0]}x${camInfo.resolution[1]}` : '—'}
                                    </p>
                                    <p className="text-slate-400 text-[10px] font-mono font-semibold">
                                        {camInfo.fps && camInfo.fps > 0 ? `${camInfo.fps.toFixed(1)} FPS` : '—'}
                                    </p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Biomechanics AI Edge */}
            {state.biomechanics && (
                <div className="glass p-5 flex flex-col md:flex-row items-center justify-between border-l-4 border-indigo-500 gap-6">
                    <div className="flex items-center gap-4 w-full md:w-auto">
                        <div className="w-12 h-12 rounded-full bg-indigo-500/20 flex items-center justify-center text-2xl shrink-0">
                            {state.biomechanics.pose_model === 'densepose' ? '🌀' : '🧍‍♂️'}
                        </div>
                        <div>
                            <h3 className="text-lg font-bold text-indigo-400">
                                {state.biomechanics.pose_model === 'densepose' ? 'Biomechanics AI (DensePose)' : 'Biomechanics AI (MediaPipe)'}
                            </h3>
                            <p className="text-xs text-slate-400">
                                {state.biomechanics.pose_model === 'densepose' 
                                    ? `Live 3D body surface mesh (${state.biomechanics.is_simulated ? 'Simulated Fallback' : 'Real-Time GPU'})`
                                    : 'Live 3D skeletal body landmarks (Lightweight CPU)'
                                }
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center gap-6 justify-around w-full md:w-auto flex-wrap">
                        <div className="text-center">
                            <p className="text-xs text-slate-500 font-bold uppercase tracking-wider mb-1">Status</p>
                            <p className={`font-semibold text-sm ${state.biomechanics.status === 'Idle' ? 'text-slate-400' : 'text-emerald-400 animate-pulse'}`}>
                                {state.biomechanics.status}
                            </p>
                        </div>
                        <div className="text-center">
                            <p className="text-xs text-slate-500 font-bold uppercase tracking-wider mb-1">Release Angle</p>
                            <p className="text-xl font-bold font-mono text-indigo-300">
                                {state.biomechanics.elbow_angle}°
                            </p>
                        </div>
                        <div className="text-center">
                            <p className="text-xs text-slate-500 font-bold uppercase tracking-wider mb-1">Knee Flexion</p>
                            <p className="text-xl font-bold font-mono text-indigo-300">
                                {state.biomechanics.knee_angle}°
                            </p>
                        </div>
                        <BiometricsCanvas 
                            poseModel={state.biomechanics.pose_model} 
                            mesh={state.biomechanics.surface_mesh} 
                            skeleton={state.biomechanics.skeleton} 
                        />
                    </div>
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Event Feed */}
                <div className="glass p-5">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">📋 Event-Protokoll</h3>
                    <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2">
                        {events.slice().reverse().map((evt, idx) => (
                            <div key={idx} className="bg-slate-800/40 p-3 rounded-lg border border-slate-700/50 text-sm flex flex-col gap-1">
                                <div className="flex justify-between items-center">
                                    <span className="text-slate-300 font-medium">{evt.type}</span>
                                    <span className="text-slate-500 text-xs">{evt.agent}</span>
                                </div>
                                {evt.data && Object.keys(evt.data).length > 0 && (
                                    <span className="text-slate-400 text-xs truncate bg-slate-950/30 px-2 py-1 rounded font-mono">{JSON.stringify(evt.data)}</span>
                                )}
                            </div>
                        ))}
                        {events.length === 0 && <p className="text-slate-500 text-sm text-center py-4">Noch keine Events...</p>}
                    </div>
                </div>

                {/* Live-Wurfkarte */}
                <div className="glass p-5 flex flex-col items-center justify-center min-h-[350px]">
                    <ShotChart shots={liveShots} title="Live-Wurfkarte" />
                </div>

                {/* Violations / Alerts */}
                <div className="glass p-5">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <AlertTriangle className="w-5 h-5 text-rose-400" /> Schiedsrichter-Meldungen
                    </h3>
                    <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2">
                        {violations.slice().reverse().map((evt, idx) => (
                            <motion.div
                                key={idx}
                                initial={{ x: 20, opacity: 0 }}
                                animate={{ x: 0, opacity: 1 }}
                                className="bg-rose-500/10 border border-rose-500/30 p-3 rounded-lg text-sm flex flex-col gap-1"
                            >
                                <p className="text-rose-400 font-semibold">{evt.type}</p>
                                <p className="text-slate-400 text-xs bg-slate-950/30 px-2 py-1 rounded font-mono truncate">{JSON.stringify(evt.data)}</p>
                            </motion.div>
                        ))}
                        {violations.length === 0 && (
                            <p className="text-slate-500 text-sm text-center py-4">Keine Verstöße erkannt ✅</p>
                        )}
                    </div>
                </div>
            </div>

            {/* 🎬 Highlight-Reel Panel */}
            <div className="glass p-5 border border-orange-500/10 shadow-[0_0_15px_rgba(234,88,12,0.03)] rounded-2xl">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold flex items-center gap-2 text-orange-400">
                        <Video className="w-5 h-5" /> 🎬 Automatisches Highlight-Reel (AI Video Clipper)
                    </h3>
                    <motion.button
                        onClick={fetchHighlights}
                        disabled={loadingHighlights}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        className="px-4 py-1.5 bg-orange-500/10 border border-orange-500/30 text-orange-400 rounded-lg text-xs font-semibold flex items-center gap-2 hover:bg-orange-500/20 disabled:opacity-50 transition-all"
                    >
                        {loadingHighlights ? (
                            <>
                                <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Generieren...
                            </>
                        ) : (
                            <>
                                <RefreshCw className="w-3.5 h-3.5" /> Highlights laden
                            </>
                        )}
                    </motion.button>
                </div>

                {highlights.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {highlights.map((hl, idx) => (
                            <div key={idx} className="bg-slate-800/40 border border-slate-700/50 p-3 rounded-xl flex flex-col gap-2">
                                <div className="flex justify-between items-center text-xs">
                                    <span className="text-orange-400 font-semibold">{hl.type}</span>
                                    <span className="text-slate-500 font-mono">Frame {hl.frame_ref}</span>
                                </div>
                                <p className="text-slate-300 text-xs line-clamp-2 min-h-[32px]">{hl.desc}</p>
                                <video src={`${API_BASE}${hl.video_url}`} controls className="w-full rounded-lg bg-slate-950" />
                            </div>
                        ))}
                    </div>
                ) : (
                    <p className="text-slate-500 text-sm text-center py-6">Noch keine Highlights generiert. Klicken Sie auf "Highlights laden", um das Spielvideo zu scannen.</p>
                )}
            </div>

            {/* Stop Button */}
            <div className="text-center mt-6">
                <button
                    onClick={onStop}
                    className="px-8 py-3 bg-rose-600 hover:bg-rose-700 text-white font-semibold rounded-xl transition-colors shadow-[0_4px_12px_rgba(225,29,72,0.25)]"
                >
                    Spiel beenden
                </button>
            </div>
        </div>
    );
};

function InfoCard({ icon, label, value }) {
    return (
        <div className="glass p-4 flex items-center gap-3">
            {icon}
            <div>
                <p className="text-slate-400 text-xs font-medium">{label}</p>
                <p className="text-xl font-bold text-slate-100">{value}</p>
            </div>
        </div>
    );
}

export default LiveDashboard;
