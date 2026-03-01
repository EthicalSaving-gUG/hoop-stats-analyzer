import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, FileVideo, Activity, Target, Shield, CheckCircle, XCircle } from 'lucide-react';
import axios from 'axios';
import ShotChart from './components/ShotChart';

// API Configuration
const API_BASE = 'http://localhost:8000';

function App() {
  const [appState, setAppState] = useState('UPLOAD'); // UPLOAD, PROCESSING, RESULTS
  const [jobId, setJobId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState('');
  const [resultsData, setResultsData] = useState(null);

  // Poll for progress automatically when processing
  useEffect(() => {
    let interval;
    if (appState === 'PROCESSING' && jobId) {
      interval = setInterval(async () => {
        try {
          const res = await axios.get(`${API_BASE}/analyze/progress/${jobId}`);
          if (res.data.status === 'error') {
            console.error("Job not found");
            clearInterval(interval);
            return;
          }

          setProgress(res.data.progress);
          setStatusMsg(res.data.status);

          if (res.data.progress === 100) {
            clearInterval(interval);
            setTimeout(() => fetchResults(res.data.result_file), 1000);
          }
        } catch (e) {
          console.error("Polling error:", e);
        }
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
      console.error("Upload failed", error);
      setStatusMsg('Upload failed. Please try again.');
      setAppState('UPLOAD');
    }
  };

  const fetchResults = async (resultFileName) => {
    try {
      const res = await axios.get(`${API_BASE}/results/${resultFileName}`);
      setResultsData(res.data);
      setAppState('RESULTS');
    } catch (e) {
      console.error("Could not fetch results data:", e);
      setStatusMsg('Analysis finished, but could not load results.');
    }
  };


  return (
    <div className="min-h-screen gradient-bg text-slate-100 flex flex-col items-center justify-center p-6 relative overflow-hidden">

      {/* Background Decorative Circles */}
      <div className="absolute top-[-10%] left-[-10%] w-[500px] h-[500px] rounded-full bg-orange-500/10 blur-[100px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[600px] h-[600px] rounded-full bg-blue-500/10 blur-[120px] pointer-events-none" />

      <motion.div
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="text-center z-10 mb-10"
      >
        <h1 className="text-5xl font-bold tracking-tight mb-2 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-orange-600">
          Hoop Stats Pro
        </h1>
        <p className="text-slate-400 text-lg font-light tracking-wide">AI-Powered Basketball Trajectory Analytics</p>
      </motion.div>

      <AnimatePresence mode="wait">

        {/* --- STATE: UPLOAD --- */}
        {appState === 'UPLOAD' && (
          <motion.div
            key="upload"
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="w-full max-w-xl glass p-10 flex flex-col items-center justify-center text-center z-10"
          >
            <div className="w-24 h-24 rounded-full bg-slate-800/50 flex items-center justify-center mb-6 shadow-inner border border-slate-700/50 relative">
              <UploadCloud className="w-10 h-10 text-orange-400" />
            </div>
            <h2 className="text-2xl font-semibold mb-2">Upload Game Footage</h2>
            <p className="text-slate-400 mb-8 max-w-sm">Provide a clear, wide-angle video of the court. We accept MP4 and MOV formats.</p>

            <label className="btn-primary cursor-pointer px-8 py-4 rounded-xl font-semibold shadow-lg hover:shadow-orange-500/25 flex items-center gap-3 w-3/4 justify-center">
              <FileVideo className="w-5 h-5" />
              <span>Select Video File</span>
              <input type="file" accept="video/mp4,video/x-m4v,video/*" className="hidden" onChange={handleFileUpload} />
            </label>
          </motion.div>
        )}

        {/* --- STATE: PROCESSING --- */}
        {appState === 'PROCESSING' && (
          <motion.div
            key="processing"
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="w-full max-w-2xl glass p-10 flex flex-col items-center justify-center text-center z-10 relative overflow-hidden"
          >
            <Activity className="w-12 h-12 text-orange-500 mb-4 animate-pulse-slow" />
            <h2 className="text-2xl font-semibold mb-2">Analyzing Gameplay...</h2>
            <p className="text-orange-400 font-medium mb-10 h-6">
              {statusMsg}
            </p>

            <div className="w-full bg-slate-800/80 rounded-full h-4 mb-4 relative overflow-hidden border border-slate-700">
              {/* Progress Fill */}
              <motion.div
                className="absolute inset-y-0 left-0 bg-gradient-to-r from-orange-600 to-orange-400 rounded-full"
                initial={{ width: '0%' }}
                animate={{ width: `${progress}%` }}
                transition={{ ease: "circOut", duration: 0.5 }}
              />
            </div>

            {/* Bouncing Basketball Animation mapping to progress */}
            <div className="w-full relative h-12 mt-2">
              <motion.div
                className="absolute bottom-0 text-3xl"
                initial={{ left: '0%' }}
                animate={{ left: `${progress}%` }}
                transition={{ ease: "circOut", duration: 0.5 }}
                style={{ translateX: '-50%' }}
              >
                <motion.div
                  animate={{ y: [0, -25, 0], rotate: 360 }}
                  transition={{
                    y: { repeat: Infinity, duration: 0.6, ease: "easeInOut" },
                    rotate: { repeat: Infinity, duration: 1.5, ease: "linear" }
                  }}
                >
                  🏀
                </motion.div>
              </motion.div>
            </div>

            <p className="text-slate-500 text-sm mt-6 font-mono font-medium">{progress}% Complete</p>
          </motion.div>
        )}

        {/* --- STATE: RESULTS --- */}
        {appState === 'RESULTS' && resultsData && (
          <motion.div
            key="results"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="w-full max-w-5xl z-10"
          >
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-3xl font-bold flex items-center gap-3"><Target className="text-orange-500" /> Advanced Shot Metrics</h2>
              <button onClick={() => setAppState('UPLOAD')} className="px-5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-sm font-semibold transition-colors">
                New Analysis
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              {Object.entries(resultsData.team_summary).map(([teamName, stats]) => (
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
                <ShotChart shots={resultsData.shots} title="Team Shot Distribution" />
              </div>

              <div className="glass p-6 h-full overflow-hidden flex flex-col">
                <h3 className="text-xl font-semibold mb-4 flex items-center gap-2"><Activity className="w-5 h-5" /> Detailed Log</h3>
                <div className="overflow-y-auto pr-2 flex-grow max-h-[400px]">
                  <div className="space-y-3">
                    {resultsData.shots.map((shot, idx) => (
                      <div key={idx} className="bg-slate-800/40 p-3 rounded-lg border border-slate-700/50 flex justify-between items-center text-sm">
                        <div>
                          <p className="font-semibold text-slate-200">#{"shooter_id" in shot ? shot.shooter_id : "?"} - {shot.zone}</p>
                          <p className="text-slate-400 text-xs">Frame {shot.start_frame}</p>
                        </div>
                        <div>
                          {shot.status === 'made'
                            ? <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded bg-emerald-500/10 text-emerald-400 font-medium text-xs"><CheckCircle className="w-3 h-3" /> Made</span>
                            : <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded bg-rose-500/10 text-rose-400 font-medium text-xs"><XCircle className="w-3 h-3" /> Missed</span>
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
