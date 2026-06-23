import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { UploadCloud, Users, Play, Plus, X } from 'lucide-react';
import axios from 'axios';
import { API_BASE } from '../config';

const ManualSetup = ({ onStart, onBack }) => {
    const [teamHome, setTeamHome] = useState('Team A');
    const [teamAway, setTeamAway] = useState('Team B');
    const [players, setPlayers] = useState([
        { id: '1', name: 'Spieler 1', jersey: '1', team: 'home' },
        { id: '2', name: 'Spieler 2', jersey: '2', team: 'away' }
    ]);
    const [videoFile, setVideoFile] = useState(null);
    const [isUploading, setIsUploading] = useState(false);

    const addPlayer = (team) => {
        const newId = Date.now().toString();
        setPlayers([...players, { id: newId, name: '', jersey: '', team }]);
    };

    const updatePlayer = (id, field, value) => {
        setPlayers(players.map(p => p.id === id ? { ...p, [field]: value } : p));
    };

    const removePlayer = (id) => {
        setPlayers(players.filter(p => p.id !== id));
    };

    const handleFileUpload = (e) => {
        if (e.target.files && e.target.files[0]) {
            setVideoFile(e.target.files[0]);
        }
    };

    const handleStart = async () => {
        let videoUrl = null;
        if (videoFile) {
            setIsUploading(true);
            const formData = new FormData();
            formData.append('file', videoFile);
            try {
                const res = await axios.post(`${API_BASE}/manual/upload-video`, formData);
                videoUrl = res.data.video_url;
            } catch (err) {
                console.error("Video upload failed", err);
            }
            setIsUploading(false);
        }

        try {
            await axios.post(`${API_BASE}/manual/start`, {
                team_home: teamHome,
                team_away: teamAway,
                players: players.filter(p => p.name.trim() || p.jersey.trim()), // Only send valid players
                video_url: videoUrl
            });
            onStart();
        } catch (err) {
            console.error("Failed to start manual session", err);
        }
    };

    return (
        <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.9, opacity: 0 }}
            className="w-full max-w-5xl z-10 space-y-6"
        >
            <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold">Manueller Kampfbericht - Setup</h2>
                <button onClick={onBack} className="text-sm text-slate-400 hover:text-slate-200">← Zurück</button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Team Home */}
                <div className="glass p-6">
                    <div className="mb-4">
                        <label className="text-sm text-slate-400 mb-1 block font-semibold">Team Home (Links)</label>
                        <input value={teamHome} onChange={(e) => setTeamHome(e.target.value)} placeholder="z.B. Grizzlies"
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-lg font-bold focus:border-blue-500 outline-none text-blue-400" />
                    </div>
                    
                    <div className="space-y-3 mb-4">
                        <h3 className="text-sm text-slate-400 font-semibold border-b border-slate-700 pb-2">Spieler</h3>
                        {players.filter(p => p.team === 'home').map((p) => (
                            <div key={p.id} className="flex gap-2 items-center">
                                <input value={p.jersey} onChange={(e) => updatePlayer(p.id, 'jersey', e.target.value)} placeholder="#"
                                    className="w-12 bg-slate-800 border border-slate-700 rounded text-center px-1 py-1 text-sm focus:border-blue-500 outline-none" />
                                <input value={p.name} onChange={(e) => updatePlayer(p.id, 'name', e.target.value)} placeholder="Spielername"
                                    className="flex-grow bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm focus:border-blue-500 outline-none" />
                                <button onClick={() => removePlayer(p.id)} className="text-slate-500 hover:text-rose-400"><X className="w-4 h-4" /></button>
                            </div>
                        ))}
                    </div>
                    <button onClick={() => addPlayer('home')} className="text-sm text-blue-400 hover:text-blue-300 font-medium flex items-center gap-1">
                        <Plus className="w-4 h-4" /> Spieler hinzufügen
                    </button>
                </div>

                {/* Team Away */}
                <div className="glass p-6">
                    <div className="mb-4">
                        <label className="text-sm text-slate-400 mb-1 block font-semibold">Team Away (Rechts)</label>
                        <input value={teamAway} onChange={(e) => setTeamAway(e.target.value)} placeholder="z.B. Uffenheim"
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-lg font-bold focus:border-emerald-500 outline-none text-emerald-400" />
                    </div>
                    
                    <div className="space-y-3 mb-4">
                        <h3 className="text-sm text-slate-400 font-semibold border-b border-slate-700 pb-2">Spieler</h3>
                        {players.filter(p => p.team === 'away').map((p) => (
                            <div key={p.id} className="flex gap-2 items-center">
                                <input value={p.jersey} onChange={(e) => updatePlayer(p.id, 'jersey', e.target.value)} placeholder="#"
                                    className="w-12 bg-slate-800 border border-slate-700 rounded text-center px-1 py-1 text-sm focus:border-emerald-500 outline-none" />
                                <input value={p.name} onChange={(e) => updatePlayer(p.id, 'name', e.target.value)} placeholder="Spielername"
                                    className="flex-grow bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm focus:border-emerald-500 outline-none" />
                                <button onClick={() => removePlayer(p.id)} className="text-slate-500 hover:text-rose-400"><X className="w-4 h-4" /></button>
                            </div>
                        ))}
                    </div>
                    <button onClick={() => addPlayer('away')} className="text-sm text-emerald-400 hover:text-emerald-300 font-medium flex items-center gap-1">
                        <Plus className="w-4 h-4" /> Spieler hinzufügen
                    </button>
                </div>
            </div>

            {/* Video Upload */}
            <div className="glass p-6 text-center">
                <h3 className="text-lg font-semibold mb-2">Spiel-Video (Optional)</h3>
                <p className="text-slate-400 text-sm mb-4">Lade ein Video hoch, um es parallel zur Statistik-Erfassung anzusehen.</p>
                <div className="flex justify-center">
                    {videoFile ? (
                        <div className="bg-slate-800 px-6 py-3 rounded-xl border border-slate-700 flex items-center gap-3">
                            <span className="text-emerald-400">✓</span>
                            <span className="font-medium">{videoFile.name}</span>
                            <button onClick={() => setVideoFile(null)} className="text-rose-400 hover:text-rose-300 ml-3 text-sm">Entfernen</button>
                        </div>
                    ) : (
                        <label className="bg-slate-800 hover:bg-slate-700 cursor-pointer px-6 py-3 rounded-xl border border-slate-600 transition-colors flex items-center gap-2 font-medium">
                            <UploadCloud className="w-5 h-5" /> Video auswählen
                            <input type="file" accept="video/*" className="hidden" onChange={handleFileUpload} />
                        </label>
                    )}
                </div>
            </div>

            {/* Start Button */}
            <div className="text-center pt-4">
                <button 
                    onClick={handleStart}
                    disabled={isUploading}
                    className="btn-primary px-10 py-4 rounded-xl font-semibold text-lg shadow-lg hover:shadow-blue-500/25 inline-flex items-center gap-3 disabled:opacity-50"
                >
                    {isUploading ? (
                        <span className="animate-pulse">Lädt hoch...</span>
                    ) : (
                        <>
                            <Play className="w-5 h-5" /> Dashboard starten
                        </>
                    )}
                </button>
            </div>
        </motion.div>
    );
};

export default ManualSetup;
