import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Download, Undo2, Plus, Minus, ChevronDown, ChevronUp } from 'lucide-react';
import axios from 'axios';
import VideoPlayer from './VideoPlayer';

const API_BASE = 'http://localhost:8000';

const STAT_FIELDS = [
    { key: 'points_2pt', label: '2PT', short: '2P' },
    { key: 'points_3pt', label: '3PT', short: '3P' },
    { key: 'points_ft', label: 'FT', short: 'FT' },
    { key: 'fouls', label: 'Fouls', short: 'PF' },
    { key: 'rebounds_def', label: 'Def Reb', short: 'DR' },
    { key: 'rebounds_off', label: 'Off Reb', short: 'OR' },
    { key: 'assists', label: 'Assists', short: 'AST' },
    { key: 'steals', label: 'Steals', short: 'STL' },
    { key: 'blocks', label: 'Blocks', short: 'BLK' },
    { key: 'turnovers', label: 'Turnover', short: 'TO' },
    { key: 'minutes', label: 'Minuten', short: 'MIN' }
];

const ManualDashboard = ({ onExit }) => {
    const [state, setState] = useState(null);
    const [activeTab, setActiveTab] = useState('home');
    const [expandedPlayer, setExpandedPlayer] = useState(null);
    const [actionHistory, setActionHistory] = useState([]);

    useEffect(() => {
        fetchState();
    }, []);

    const fetchState = async () => {
        try {
            const res = await axios.get(`${API_BASE}/manual/state`);
            setState(res.data);
        } catch (err) {
            console.error("Failed to fetch state", err);
        }
    };

    const updateStat = async (playerId, field, value) => {
        // Optimistic UI update
        const previousState = { ...state, players: state.players.map(p => ({...p})) };
        
        setState(prev => ({
            ...prev,
            players: prev.players.map(p => {
                if (p.id === playerId) {
                    return { ...p, [field]: Math.max(0, p[field] + value) };
                }
                return p;
            })
        }));

        // Log action for undo
        setActionHistory(prev => [...prev, { playerId, field, value: -value, previousState }]);

        // Server update
        try {
            await axios.post(`${API_BASE}/manual/update-player`, {
                player_id: playerId,
                field: field,
                value: value
            });
        } catch (err) {
            console.error("Failed to update stat", err);
            // Revert on error
            setState(previousState);
        }
    };

    const handleUndo = async () => {
        if (actionHistory.length === 0) return;
        
        const lastAction = actionHistory[actionHistory.length - 1];
        setActionHistory(prev => prev.slice(0, -1));
        
        // Restore previous state visually immediately
        setState(lastAction.previousState);

        // Send reverse update to server
        try {
            await axios.post(`${API_BASE}/manual/update-player`, {
                player_id: lastAction.playerId,
                field: lastAction.field,
                value: lastAction.value // value was already negated when pushed to history
            });
        } catch (err) {
            console.error("Undo failed on server", err);
            fetchState(); // re-sync
        }
    };

    const handleExport = async () => {
        try {
            const res = await axios.get(`${API_BASE}/manual/export`);
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(res.data, null, 2));
            const downloadAnchorNode = document.createElement('a');
            downloadAnchorNode.setAttribute("href", dataStr);
            downloadAnchorNode.setAttribute("download", `hoop_stats_manual_${new Date().toISOString().slice(0,10)}.json`);
            document.body.appendChild(downloadAnchorNode);
            downloadAnchorNode.click();
            downloadAnchorNode.remove();
        } catch (err) {
            console.error("Export failed", err);
        }
    };

    if (!state) return <div className="text-center p-20 animate-pulse text-xl">Lade Dashboard...</div>;

    const teamHomePoints = state.players.filter(p => p.team === 'home').reduce((acc, p) => acc + (p.points_2pt * 2) + (p.points_3pt * 3) + p.points_ft, 0);
    const teamAwayPoints = state.players.filter(p => p.team === 'away').reduce((acc, p) => acc + (p.points_2pt * 2) + (p.points_3pt * 3) + p.points_ft, 0);

    const playersList = state.players.filter(p => p.team === activeTab);

    return (
        <div className="w-full h-[90vh] flex flex-col md:flex-row gap-6 z-10">
            {/* Left Side: Video (60%) */}
            <div className="md:w-3/5 h-full flex flex-col gap-4">
                <div className="glass p-4 flex justify-between items-center shrink-0">
                    <button onClick={onExit} className="text-sm text-slate-400 hover:text-white px-3 py-1 rounded bg-slate-800">Exit</button>
                    <div className="flex gap-8 text-center font-bold">
                        <div className="text-blue-400">
                            <div className="text-xs uppercase tracking-widest">{state.team_home}</div>
                            <div className="text-3xl">{teamHomePoints}</div>
                        </div>
                        <div className="text-slate-500 text-3xl">-</div>
                        <div className="text-emerald-400">
                            <div className="text-xs uppercase tracking-widest">{state.team_away}</div>
                            <div className="text-3xl">{teamAwayPoints}</div>
                        </div>
                    </div>
                    <button onClick={handleExport} className="text-sm text-slate-300 hover:text-white px-3 py-1.5 rounded bg-slate-800 flex items-center gap-2 border border-slate-700">
                        <Download className="w-4 h-4" /> Export
                    </button>
                </div>
                
                <div className="flex-grow min-h-[300px]">
                    <VideoPlayer videoUrl={state.video_url} />
                </div>
            </div>

            {/* Right Side: Stats Dashboard (40%) */}
            <div className="md:w-2/5 h-full flex flex-col glass overflow-hidden border border-slate-700/50 rounded-xl">
                {/* Tabs */}
                <div className="flex border-b border-slate-700/50 shrink-0">
                    <button 
                        onClick={() => setActiveTab('home')}
                        className={`flex-1 py-4 font-semibold transition-colors border-b-2 ${activeTab === 'home' ? 'border-blue-500 text-blue-400 bg-blue-500/5' : 'border-transparent text-slate-400 hover:bg-slate-800/50'}`}
                    >
                        {state.team_home}
                    </button>
                    <button 
                        onClick={() => setActiveTab('away')}
                        className={`flex-1 py-4 font-semibold transition-colors border-b-2 ${activeTab === 'away' ? 'border-emerald-500 text-emerald-400 bg-emerald-500/5' : 'border-transparent text-slate-400 hover:bg-slate-800/50'}`}
                    >
                        {state.team_away}
                    </button>
                </div>

                {/* Toolbar */}
                <div className="p-3 bg-slate-800/50 border-b border-slate-700/50 flex justify-end shrink-0">
                    <button 
                        onClick={handleUndo} 
                        disabled={actionHistory.length === 0}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded bg-slate-700 hover:bg-slate-600 text-slate-300 disabled:opacity-30 transition-colors"
                    >
                        <Undo2 className="w-4 h-4" /> Undo
                    </button>
                </div>

                {/* Player List */}
                <div className="flex-grow overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
                    {playersList.length === 0 ? (
                        <div className="text-center text-slate-500 py-10">Keine Spieler angelegt.</div>
                    ) : (
                        playersList.map(player => {
                            const isExpanded = expandedPlayer === player.id;
                            const totalPts = (player.points_2pt * 2) + (player.points_3pt * 3) + player.points_ft;
                            
                            return (
                                <div key={player.id} className={`bg-slate-800/80 rounded-xl border transition-colors ${isExpanded ? 'border-orange-500/50' : 'border-slate-700'}`}>
                                    {/* Header / Quick Actions */}
                                    <div className="p-3 flex items-center justify-between gap-2">
                                        <div 
                                            className="flex items-center gap-3 cursor-pointer flex-1"
                                            onClick={() => setExpandedPlayer(isExpanded ? null : player.id)}
                                        >
                                            <div className={`w-8 h-8 rounded flex items-center justify-center font-bold text-slate-900 ${activeTab === 'home' ? 'bg-blue-400' : 'bg-emerald-400'}`}>
                                                {player.jersey}
                                            </div>
                                            <div>
                                                <div className="font-semibold text-sm">{player.name || 'Unbekannt'}</div>
                                                <div className="text-xs text-slate-400">{totalPts} Pts, {player.fouls} PF</div>
                                            </div>
                                            {isExpanded ? <ChevronUp className="w-4 h-4 text-slate-500 ml-auto" /> : <ChevronDown className="w-4 h-4 text-slate-500 ml-auto" />}
                                        </div>
                                        
                                        <div className="flex gap-1">
                                            <button onClick={() => updateStat(player.id, 'points_2pt', 1)} className="px-2 py-1.5 bg-slate-700 hover:bg-orange-500 hover:text-white text-xs font-bold rounded text-slate-300 transition-colors">+2</button>
                                            <button onClick={() => updateStat(player.id, 'points_3pt', 1)} className="px-2 py-1.5 bg-slate-700 hover:bg-orange-500 hover:text-white text-xs font-bold rounded text-slate-300 transition-colors">+3</button>
                                            <button onClick={() => updateStat(player.id, 'rebounds_def', 1)} className="px-2 py-1.5 bg-slate-700 hover:bg-blue-500 hover:text-white text-xs font-bold rounded text-slate-300 transition-colors">+DR</button>
                                        </div>
                                    </div>

                                    {/* Expanded Stats */}
                                    <AnimatePresence>
                                        {isExpanded && (
                                            <motion.div 
                                                initial={{ height: 0, opacity: 0 }} 
                                                animate={{ height: 'auto', opacity: 1 }} 
                                                exit={{ height: 0, opacity: 0 }}
                                                className="overflow-hidden border-t border-slate-700/50"
                                            >
                                                <div className="p-3 grid grid-cols-3 gap-3 bg-slate-900/50">
                                                    {STAT_FIELDS.map(field => (
                                                        <div key={field.key} className="flex flex-col items-center p-2 rounded bg-slate-800">
                                                            <div className="text-[10px] text-slate-400 uppercase tracking-wider mb-1">{field.label}</div>
                                                            <div className="flex items-center gap-3">
                                                                <button onClick={() => updateStat(player.id, field.key, -1)} disabled={player[field.key] === 0} className="w-5 h-5 flex items-center justify-center rounded-full bg-slate-700 hover:bg-slate-600 disabled:opacity-30 text-slate-300"><Minus className="w-3 h-3" /></button>
                                                                <span className="font-mono text-sm w-4 text-center">{player[field.key]}</span>
                                                                <button onClick={() => updateStat(player.id, field.key, 1)} className="w-5 h-5 flex items-center justify-center rounded-full bg-slate-700 hover:bg-slate-600 text-slate-300"><Plus className="w-3 h-3" /></button>
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </div>
                            );
                        })
                    )}
                </div>
            </div>
        </div>
    );
};

export default ManualDashboard;
