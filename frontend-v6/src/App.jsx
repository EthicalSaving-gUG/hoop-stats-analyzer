import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, FileVideo, Activity, Target, Shield, CheckCircle, XCircle, Play, Video } from 'lucide-react';
import axios from 'axios';
import ShotChart from './components/ShotChart';
import AgentCard from './components/AgentCard';
import LiveDashboard from './components/LiveDashboard';
import ManualSetup from './components/ManualSetup';
import ManualDashboard from './components/ManualDashboard';
import ClientAIBenchmark from './components/ClientAIBenchmark';

const API_BASE = 'http://localhost:8000';

function App() {
  // Top-level: MODE_SELECT -> UPLOAD/LIVE_SETUP -> PROCESSING/LIVE -> RESULTS
  const [appState, setAppState] = useState('MODE_SELECT');
  const [jobId, setJobId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState('');
  const [resultsData, setResultsData] = useState(null);

  // Kampfgericht config
  const [enabledAgents, setEnabledAgents] = useState({
    shot_clock: true, game_clock: true, scoresheet: true, referee: true, pose: true, referee_signals: true
  });
  const [cameras, setCameras] = useState([{ id: 'cam1', source: '' }]);
  const [teamHome, setTeamHome] = useState('Team Schwarz');
  const [teamAway, setTeamAway] = useState('Team Gelb');
  const [ruleset, setRuleset] = useState('fiba');
  const [processingFps, setProcessingFps] = useState(10);
  const [poseModel, setPoseModel] = useState('mediapipe');

  // Poll for file analysis progress
  useEffect(() => {
    let interval;
    if (appState === 'PROCESSING' && jobId) {
      interval = setInterval(async () => {
        try {
          const res = await axios.get(`${API_BASE}/analyze/progress/${jobId}`);
          if (res.data.status === 'error') { clearInterval(interval); return; }
          setProgress(res.data.progress);
          setStatusMsg(res.data.status);
          if (res.data.progress === 100) {
            clearInterval(interval);
            setTimeout(() => fetchResults(res.data.result_file), 1000);
          }
        } catch (e) { console.error(e); }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [appState, jobId]);

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setAppState('PROCESSING');
    setProgress(0);
    setStatusMsg('Uploading Video...');
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await axios.post(`${API_BASE}/analyze/file`, formData);
      setJobId(res.data.job_id);
    } catch (error) {
      setStatusMsg('Upload failed.');
      setAppState('UPLOAD');
    }
  };

  const fetchResults = async (resultFileName) => {
    try {
      const res = await axios.get(`${API_BASE}/results/${resultFileName}`);
      setResultsData(res.data);
      setAppState('RESULTS');
    } catch (e) {
      setStatusMsg('Could not load results.');
    }
  };

  const toggleAgent = (key) => {
    setEnabledAgents(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const addCamera = () => {
    setCameras(prev => [...prev, { id: `cam${prev.length + 1}`, source: '' }]);
  };

  const updateCamera = (index, field, value) => {
    setCameras(prev => prev.map((c, i) => i === index ? { ...c, [field]: value } : c));
  };

  const removeCamera = (index) => {
    setCameras(prev => prev.filter((_, i) => i !== index));
  };

  const startLiveGame = async () => {
    const activeAgents = Object.entries(enabledAgents).filter(([_, v]) => v).map(([k]) => k);
    const validCameras = cameras.filter(c => c.source.trim());
    if (validCameras.length === 0) return;

    try {
      await axios.post(`${API_BASE}/game/start`, {
        cameras: validCameras,
        agents: activeAgents,
        team_home: teamHome || 'Team Schwarz',
        team_away: teamAway || 'Team Gelb',
        ruleset: ruleset,
        processing_fps: processingFps,
        pose_model: poseModel,
      });
      setAppState('LIVE');
    } catch (e) {
      console.error('Failed to start game:', e);
    }
  };

  const stopLiveGame = async () => {
    try {
      await axios.post(`${API_BASE}/game/stop`);
    } catch (e) { /* ignore */ }
    setAppState('MODE_SELECT');
  };

  return (
    <div className="min-h-screen gradient-bg text-slate-100 flex flex-col items-center p-6 relative overflow-hidden">
      <div className="absolute top-[-10%] left-[-10%] w-[500px] h-[500px] rounded-full bg-orange-500/10 blur-[100px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[600px] h-[600px] rounded-full bg-blue-500/10 blur-[120px] pointer-events-none" />

      <motion.div initial={{ y: -20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="text-center z-10 mb-10 mt-6">
        <h1 className="text-5xl font-bold tracking-tight mb-2 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-orange-600">
          Hoop Stats Pro
        </h1>
        <p className="text-slate-400 text-lg font-light tracking-wide">AI-Powered Basketball Analytics & Kampfgericht</p>
      </motion.div>

      <AnimatePresence mode="wait">

        {/* ─── MODE SELECT ─── */}
        {appState === 'MODE_SELECT' && (
          <motion.div key="mode" initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.9, opacity: 0 }}
            className="w-full max-w-6xl z-10 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
          >
            <motion.div whileHover={{ scale: 1.03 }} onClick={() => setAppState('UPLOAD')}
              className="glass p-8 cursor-pointer text-center hover:border-orange-500/30 transition-all border-2 border-transparent"
            >
              <div className="text-5xl mb-4">📹</div>
              <h2 className="text-2xl font-semibold mb-2">Video-Analyse</h2>
              <p className="text-slate-400 text-sm">Lade ein Video hoch und erhalte detaillierte Spielerstatistiken, Shot Charts und Metriken.</p>
            </motion.div>

            <motion.div whileHover={{ scale: 1.03 }} onClick={() => setAppState('LIVE_SETUP')}
              className="glass p-8 cursor-pointer text-center hover:border-orange-500/30 transition-all border-2 border-transparent"
            >
              <div className="text-5xl mb-4">🏟️</div>
              <h2 className="text-2xl font-semibold mb-2">Live Kampfgericht</h2>
              <p className="text-slate-400 text-sm">Starte die Echtzeit-Spielüberwachung mit 24s-Uhr, Spieluhr, Kampfbericht und Schiedsrichter.</p>
            </motion.div>

            <motion.div whileHover={{ scale: 1.03 }} onClick={() => setAppState('MANUAL_SETUP')}
              className="glass p-8 cursor-pointer text-center hover:border-orange-500/30 transition-all border-2 border-transparent"
            >
              <div className="text-5xl mb-4">📝</div>
              <h2 className="text-2xl font-semibold mb-2">Manueller Bericht</h2>
              <p className="text-slate-400 text-sm">Video ansehen und Spielerstatistiken manuell auf einem Dashboard erfassen.</p>
            </motion.div>

            <motion.div whileHover={{ scale: 1.03 }} onClick={() => setAppState('CLIENT_AI_BENCHMARK')}
              className="glass p-8 cursor-pointer text-center hover:border-indigo-500/30 transition-all border-2 border-transparent"
            >
              <div className="text-5xl mb-4">🧪</div>
              <h2 className="text-2xl font-semibold mb-2">Browser-KI Benchmark</h2>
              <p className="text-slate-400 text-sm">Testet die Videoanalyse direkt in deinem Browser (ohne Server-Upload).</p>
            </motion.div>
          </motion.div>
        )}

        {/* ─── LIVE SETUP ─── */}
        {appState === 'LIVE_SETUP' && (
          <motion.div key="setup" initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.9, opacity: 0 }}
            className="w-full max-w-4xl z-10 space-y-6"
          >
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-bold">Kampfgericht konfigurieren</h2>
              <button onClick={() => setAppState('MODE_SELECT')} className="text-sm text-slate-400 hover:text-slate-200">← Zurück</button>
            </div>

            {/* Team Names */}
            <div className="glass p-5 grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm text-slate-400 mb-1 block">Team Home</label>
                <input value={teamHome} onChange={(e) => setTeamHome(e.target.value)} placeholder="z.B. Grizzlies"
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:border-orange-500 outline-none" />
              </div>
              <div>
                <label className="text-sm text-slate-400 mb-1 block">Team Away</label>
                <input value={teamAway} onChange={(e) => setTeamAway(e.target.value)} placeholder="z.B. Team Gelb"
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:border-orange-500 outline-none" />
              </div>
            </div>

            {/* Ruleset Selection */}
            <div>
              <h3 className="text-lg font-semibold mb-3">Regelwerk</h3>
              <div className="grid grid-cols-2 gap-4">
                <motion.div whileHover={{ scale: 1.02 }} onClick={() => setRuleset('fiba')}
                  className={`glass p-5 cursor-pointer text-center transition-all border-2 ${ruleset === 'fiba' ? 'border-blue-500/50 bg-blue-500/5' : 'border-transparent'}`}>
                  <span className="text-3xl block mb-2">🌍</span>
                  <h4 className="text-lg font-semibold">FIBA 2024</h4>
                  <p className="text-slate-400 text-xs mt-1">4×10 Min · 5 Fouls · Kein Def. 3-Sek</p>
                </motion.div>
                <motion.div whileHover={{ scale: 1.02 }} onClick={() => setRuleset('nba')}
                  className={`glass p-5 cursor-pointer text-center transition-all border-2 ${ruleset === 'nba' ? 'border-red-500/50 bg-red-500/5' : 'border-transparent'}`}>
                  <span className="text-3xl block mb-2">🇺🇸</span>
                  <h4 className="text-lg font-semibold">NBA 2024-25</h4>
                  <p className="text-slate-400 text-xs mt-1">4×12 Min · 6 Fouls · Def. 3-Sek Regel</p>
                </motion.div>
              </div>
            </div>

            {/* Performance Slider */}
            <div className="glass p-5">
              <div className="flex justify-between items-center mb-2">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <Activity className="w-5 h-5 text-orange-400" /> KI Auswertungsrate (FPS)
                </h3>
                <span className="text-xl font-bold text-orange-400">{processingFps} FPS</span>
              </div>
              <p className="text-sm text-slate-400 mb-4">
                Je höher die Rate, desto flüssiger das KI-Tracking (erfordert mehr Rechenleistung).
                Das Video-Playback bleibt immer flüssig.
              </p>
              <input
                type="range"
                min="5"
                max="30"
                step="5"
                value={processingFps}
                onChange={(e) => setProcessingFps(parseInt(e.target.value))}
                className="w-full accent-orange-500 bg-slate-700 rounded-lg appearance-none h-2 cursor-pointer"
              />
              <div className="flex justify-between text-xs text-slate-500 mt-2">
                <span>Leistungsschwache PCs (5)</span>
                <span>Normal (10)</span>
                <span>Workstation (30+)</span>
              </div>
            </div>

            {/* Agent Selection */}
            <div>
              <h3 className="text-lg font-semibold mb-3">Agenten auswählen</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.keys(enabledAgents).map(key => (
                  <AgentCard key={key} agentKey={key} enabled={enabledAgents[key]} onToggle={toggleAgent} />
                ))}
              </div>
            </div>

            {/* Pose Model Selection (Conditional if pose agent is enabled) */}
            {enabledAgents.pose && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="glass p-5 space-y-3 border-l-4 border-indigo-500">
                <div className="flex justify-between items-center">
                  <h3 className="text-lg font-semibold flex items-center gap-2">
                    🧍‍♂️ Körperhaltung-Modell (Pose Model)
                  </h3>
                  <span className="text-xs uppercase tracking-widest text-indigo-400 font-bold px-2 py-0.5 rounded bg-indigo-500/10 border border-indigo-500/20">
                    Kombinierter Modus
                  </span>
                </div>
                <p className="text-sm text-slate-400">
                  Wähle aus, welches KI-Modell für die biomechanische Analyse des Schützen verwendet werden soll.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    onClick={() => setPoseModel('mediapipe')}
                    className={`glass p-4 cursor-pointer text-left transition-all border-2 flex items-start gap-3 ${
                      poseModel === 'mediapipe'
                        ? 'border-indigo-500/50 bg-indigo-500/5 shadow-[0_0_15px_rgba(99,102,241,0.1)]'
                        : 'border-transparent hover:bg-slate-800/30'
                    }`}
                  >
                    <div className="text-2xl pt-1">⚡</div>
                    <div>
                      <h4 className="font-semibold text-slate-200">MediaPipe Pose</h4>
                      <p className="text-slate-400 text-xs mt-0.5">Leichtgewichtig & extrem schnell. Läuft flüssig auf Standard-CPUs ohne Grafikkarte.</p>
                    </div>
                  </motion.div>
                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    onClick={() => setPoseModel('densepose')}
                    className={`glass p-4 cursor-pointer text-left transition-all border-2 flex items-start gap-3 ${
                      poseModel === 'densepose'
                        ? 'border-indigo-500/50 bg-indigo-500/5 shadow-[0_0_15px_rgba(99,102,241,0.1)]'
                        : 'border-transparent hover:bg-slate-800/30'
                    }`}
                  >
                    <div className="text-2xl pt-1">🌀</div>
                    <div>
                      <h4 className="font-semibold text-slate-200 flex items-center gap-1.5">
                        DensePose (Detectron2)
                      </h4>
                      <p className="text-slate-400 text-xs mt-0.5">Volles 3D-Körpernetz. Extrem detailliert, erfordert NVIDIA GPU-Server (z.B. RTX 4090).</p>
                    </div>
                  </motion.div>
                </div>
              </motion.div>
            )}

            {/* Camera Inputs */}
            <div>
              <div className="flex justify-between items-center mb-3">
                <h3 className="text-lg font-semibold">Kamera-Quellen</h3>
                <button onClick={addCamera} className="text-sm text-orange-400 hover:text-orange-300 font-medium">+ Kamera hinzüfugen</button>
              </div>
              <div className="space-y-3">
                {cameras.map((cam, idx) => (
                  <div key={idx} className="glass p-4 flex gap-3 items-center">
                    <Video className="w-5 h-5 text-slate-400 shrink-0" />
                    <input value={cam.id} onChange={(e) => updateCamera(idx, 'id', e.target.value)} placeholder="Kamera-ID"
                      className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm w-32 focus:border-orange-500 outline-none" />
                    <input value={cam.source} onChange={(e) => updateCamera(idx, 'source', e.target.value)}
                      placeholder="Pfad oder RTSP URL (z.B. uploads/game.mp4)"
                      className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm flex-1 focus:border-orange-500 outline-none" />
                    {cameras.length > 1 && (
                      <button onClick={() => removeCamera(idx)} className="text-rose-400 hover:text-rose-300 text-sm">✕</button>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Start Button */}
            <div className="text-center pt-4">
              <button onClick={startLiveGame}
                className="btn-primary px-10 py-4 rounded-xl font-semibold text-lg shadow-lg hover:shadow-orange-500/25 inline-flex items-center gap-3">
                <Play className="w-5 h-5" /> Spiel starten
              </button>
            </div>
          </motion.div>
        )}

        {/* ─── LIVE DASHBOARD ─── */}
        {appState === 'LIVE' && (
          <motion.div key="live" initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="w-full z-10 flex justify-center">
            <LiveDashboard onStop={stopLiveGame} />
          </motion.div>
        )}

        {/* ─── MANUAL SETUP ─── */}
        {appState === 'MANUAL_SETUP' && (
          <ManualSetup 
            onStart={() => setAppState('MANUAL_REVIEW')} 
            onBack={() => setAppState('MODE_SELECT')} 
          />
        )}

        {/* ─── MANUAL REVIEW DASHBOARD ─── */}
        {appState === 'MANUAL_REVIEW' && (
          <motion.div key="manual_review" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="w-full h-full max-w-7xl">
            <ManualDashboard onExit={() => setAppState('MODE_SELECT')} />
          </motion.div>
        )}

        {/* ─── CLIENT AI BENCHMARK ─── */}
        {appState === 'CLIENT_AI_BENCHMARK' && (
          <ClientAIBenchmark onBack={() => setAppState('MODE_SELECT')} />
        )}

        {/* ─── FILE UPLOAD ─── */}
        {appState === 'UPLOAD' && (
          <motion.div key="upload" initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.9, opacity: 0 }}
            className="w-full max-w-xl glass p-10 flex flex-col items-center justify-center text-center z-10"
          >
            <button onClick={() => setAppState('MODE_SELECT')} className="self-start text-sm text-slate-400 hover:text-slate-200 mb-4">← Zurück</button>
            <div className="w-24 h-24 rounded-full bg-slate-800/50 flex items-center justify-center mb-6 shadow-inner border border-slate-700/50">
              <UploadCloud className="w-10 h-10 text-orange-400" />
            </div>
            <h2 className="text-2xl font-semibold mb-2">Video hochladen</h2>
            <p className="text-slate-400 mb-8 max-w-sm">Ein Weitwinkel-Video des Spielfelds hochladen. MP4 und MOV werden unterstützt.</p>
            <label className="btn-primary cursor-pointer px-8 py-4 rounded-xl font-semibold shadow-lg hover:shadow-orange-500/25 flex items-center gap-3 w-3/4 justify-center">
              <FileVideo className="w-5 h-5" />
              <span>Video auswählen</span>
              <input type="file" accept="video/*" className="hidden" onChange={handleFileUpload} />
            </label>
          </motion.div>
        )}

        {/* ─── PROCESSING ─── */}
        {appState === 'PROCESSING' && (
          <motion.div key="processing" initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.9, opacity: 0 }}
            className="w-full max-w-2xl glass p-10 flex flex-col items-center text-center z-10"
          >
            <Activity className="w-12 h-12 text-orange-500 mb-4 animate-pulse-slow" />
            <h2 className="text-2xl font-semibold mb-2">Analyse läuft...</h2>
            <p className="text-orange-400 font-medium mb-10 h-6">{statusMsg}</p>
            <div className="w-full bg-slate-800/80 rounded-full h-4 mb-4 relative overflow-hidden border border-slate-700">
              <motion.div className="absolute inset-y-0 left-0 bg-gradient-to-r from-orange-600 to-orange-400 rounded-full"
                initial={{ width: '0%' }} animate={{ width: `${progress}%` }} transition={{ ease: "circOut", duration: 0.5 }} />
            </div>
            <div className="w-full relative h-12 mt-2">
              <motion.div className="absolute bottom-0 text-3xl" initial={{ left: '0%' }} animate={{ left: `${progress}%` }}
                transition={{ ease: "circOut", duration: 0.5 }} style={{ translateX: '-50%' }}>
                <motion.div animate={{ y: [0, -25, 0], rotate: 360 }}
                  transition={{ y: { repeat: Infinity, duration: 0.6, ease: "easeInOut" }, rotate: { repeat: Infinity, duration: 1.5, ease: "linear" } }}>
                  🏀
                </motion.div>
              </motion.div>
            </div>
            <p className="text-slate-500 text-sm mt-6 font-mono font-medium">{progress}% Abgeschlossen</p>
          </motion.div>
        )}

        {/* ─── RESULTS ─── */}
        {appState === 'RESULTS' && resultsData && (
          <motion.div key="results" initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="w-full max-w-5xl z-10">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-3xl font-bold flex items-center gap-3"><Target className="text-orange-500" /> Erweiterte Metriken</h2>
              <button onClick={() => setAppState('MODE_SELECT')} className="px-5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-sm font-semibold transition-colors">
                Neue Analyse
              </button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              {resultsData.team_summary && Object.entries(resultsData.team_summary).map(([teamName, stats]) => (
                <div key={teamName} className="glass p-6">
                  <div className="flex items-center gap-3 mb-6">
                    <Shield className={teamName === 'Home' ? 'text-blue-400' : 'text-emerald-400'} />
                    <h3 className="text-2xl font-semibold">Team {teamName}</h3>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <StatBox label="Field Goal %" value={`${stats['FG%']}%`} subtext={`${stats.FGM} / ${stats.FGA}`} highlight />
                    <StatBox label="Effective FG %" value={`${stats['eFG%']}%`} subtext="Weighted 3PTs" />
                    <StatBox label="3-Pointers" value={`${stats['3PM']} / ${stats['3PA']}`} />
                    <StatBox label="Top Speed" value={`${stats.avg_max_speed} m/s`} />
                  </div>
                </div>
              ))}
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
              <div className="lg:col-span-2 glass p-6 h-full flex flex-col justify-center">
                <ShotChart shots={resultsData.shots || []} title="Shot-Verteilung" />
              </div>
              <div className="glass p-6 h-full overflow-hidden flex flex-col">
                <h3 className="text-xl font-semibold mb-4 flex items-center gap-2"><Activity className="w-5 h-5" /> Wurfprotokoll</h3>
                <div className="overflow-y-auto pr-2 flex-grow max-h-[400px]">
                  <div className="space-y-3">
                    {(resultsData.shots || []).map((shot, idx) => (
                      <div key={idx} className="bg-slate-800/40 p-3 rounded-lg border border-slate-700/50 flex justify-between items-center text-sm">
                        <div>
                          <p className="font-semibold text-slate-200">#{shot.shooter_id || '?'} - {shot.zone}</p>
                          <p className="text-slate-400 text-xs">Frame {shot.start_frame}</p>
                        </div>
                        <div>
                          {shot.status === 'made'
                            ? <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded bg-emerald-500/10 text-emerald-400 font-medium text-xs"><CheckCircle className="w-3 h-3" /> Treffer</span>
                            : <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded bg-rose-500/10 text-rose-400 font-medium text-xs"><XCircle className="w-3 h-3" /> Daneben</span>
                          }
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}

      </AnimatePresence>
    </div>
  )
}

function StatBox({ label, value, subtext, highlight }) {
  return (
    <div className={`p-4 rounded-xl border ${highlight ? 'bg-orange-500/10 border-orange-500/20' : 'bg-slate-800/40 border-slate-700/50'} flex flex-col justify-center`}>
      <p className="text-slate-400 text-sm font-medium mb-1">{label}</p>
      <p className={`text-3xl font-bold mb-1 tracking-tight ${highlight ? 'text-orange-400' : 'text-slate-100'}`}>{value}</p>
      {subtext && <p className="text-slate-500 text-xs font-medium uppercase tracking-wider">{subtext}</p>}
    </div>
  )
}

export default App
