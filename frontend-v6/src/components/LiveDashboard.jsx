import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, Timer, Clock, Users, Trophy } from 'lucide-react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

const LiveDashboard = ({ onStop }) => {
    const [state, setState] = useState(null);

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

            {/* Info Row */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <InfoCard icon={<Users className="w-5 h-5 text-blue-400" />} label="Teamfouls Home" value={state.team_fouls_home?.[state.quarter] || 0} />
                <InfoCard icon={<Users className="w-5 h-5 text-emerald-400" />} label="Teamfouls Away" value={state.team_fouls_away?.[state.quarter] || 0} />
                <InfoCard icon={<Trophy className="w-5 h-5 text-orange-400" />} label="Regelwerk" value={state.ruleset === 'fiba' ? 'FIBA' : 'NBA'} />
                <InfoCard icon={<Clock className="w-5 h-5 text-purple-400" />} label="Events" value={events.length} />
            </div>

            {/* Biomechanics AI Edge */}
            {state.biomechanics && (
                <div className="glass p-5 flex items-center justify-between border-l-4 border-indigo-500">
                    <div className="flex items-center gap-4">
                        <div className="w-12 h-12 rounded-full bg-indigo-500/20 flex items-center justify-center text-2xl">
                            🧍‍♂️
                        </div>
                        <div>
                            <h3 className="text-lg font-bold text-indigo-400">Biomechanics AI Edge</h3>
                            <p className="text-sm text-slate-400">Live Shooter Analysis via MediaPipe</p>
                        </div>
                    </div>
                    <div className="flex gap-8 text-center">
                        <div>
                            <p className="text-xs text-slate-500 font-bold uppercase tracking-wider mb-1">Status</p>
                            <p className={`font-semibold ${state.biomechanics.status === 'Idle' ? 'text-slate-400' : 'text-emerald-400'}`}>
                                {state.biomechanics.status}
                            </p>
                        </div>
                        <div>
                            <p className="text-xs text-slate-500 font-bold uppercase tracking-wider mb-1">Release Angle</p>
                            <p className="text-xl font-bold font-mono text-indigo-300">
                                {state.biomechanics.elbow_angle}°
                            </p>
                        </div>
                        <div>
                            <p className="text-xs text-slate-500 font-bold uppercase tracking-wider mb-1">Knee Flexion</p>
                            <p className="text-xl font-bold font-mono text-indigo-300">
                                {state.biomechanics.knee_angle}°
                            </p>
                        </div>
                    </div>
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Event Feed */}
                <div className="glass p-5">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">📋 Event-Protokoll</h3>
                    <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2">
                        {events.slice().reverse().map((evt, idx) => (
                            <div key={idx} className="bg-slate-800/40 p-3 rounded-lg border border-slate-700/50 text-sm flex justify-between items-center">
                                <div>
                                    <span className="text-slate-300 font-medium">{evt.type}</span>
                                    <span className="text-slate-500 ml-2 text-xs">{evt.agent}</span>
                                </div>
                                {evt.data && Object.keys(evt.data).length > 0 && (
                                    <span className="text-slate-400 text-xs">{JSON.stringify(evt.data)}</span>
                                )}
                            </div>
                        ))}
                        {events.length === 0 && <p className="text-slate-500 text-sm text-center py-4">Noch keine Events...</p>}
                    </div>
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
                                className="bg-rose-500/10 border border-rose-500/30 p-3 rounded-lg text-sm"
                            >
                                <p className="text-rose-400 font-semibold">{evt.type}</p>
                                <p className="text-slate-400 text-xs mt-1">{JSON.stringify(evt.data)}</p>
                            </motion.div>
                        ))}
                        {violations.length === 0 && (
                            <p className="text-slate-500 text-sm text-center py-4">Keine Verstöße erkannt ✅</p>
                        )}
                    </div>
                </div>
            </div>

            {/* Stop Button */}
            <div className="text-center">
                <button
                    onClick={onStop}
                    className="px-8 py-3 bg-rose-600 hover:bg-rose-700 text-white font-semibold rounded-xl transition-colors"
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
