import { useEffect, useState } from 'react';
import { ArrowLeft, RefreshCw, Save } from 'lucide-react';
import axios from 'axios';
import { API_BASE } from '../config';

const DEFAULT_VIDEO_POINTS = '[[100,100],[1100,100],[100,900],[1100,900]]';
const DEFAULT_REAL_POINTS = '[[0,0],[28,0],[0,15],[28,15]]';

function QualityDashboard({ onBack }) {
  const [quality, setQuality] = useState(null);
  const [videoPoints, setVideoPoints] = useState(DEFAULT_VIDEO_POINTS);
  const [realPoints, setRealPoints] = useState(DEFAULT_REAL_POINTS);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const load = async () => {
    setError('');
    try {
      const [qualityRes, calibrationRes] = await Promise.all([
        axios.get(`${API_BASE}/quality`),
        axios.get(`${API_BASE}/calibration`),
      ]);
      setQuality(qualityRes.data);
      setVideoPoints(JSON.stringify(calibrationRes.data.video || [], null, 0));
      setRealPoints(JSON.stringify(calibrationRes.data.real || [], null, 0));
    } catch (err) {
      setError(err.response?.data?.detail || 'Qualitaetsdaten konnten nicht geladen werden.');
    }
  };

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load();
  }, []);

  const saveCalibration = async () => {
    setMessage('');
    setError('');
    try {
      const payload = {
        video: JSON.parse(videoPoints),
        real: JSON.parse(realPoints),
      };
      await axios.post(`${API_BASE}/calibration`, payload);
      setMessage('Calibration gespeichert.');
      load();
    } catch (err) {
      setError(err.response?.data?.detail || 'Calibration konnte nicht gespeichert werden.');
    }
  };

  const cameras = quality?.cameras ? Object.entries(quality.cameras) : [];

  return (
    <div className="w-full max-w-6xl z-10 space-y-6">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-2xl font-bold">Datenqualitaet & Calibration</h2>
          <p className="text-sm text-slate-400 mt-1">Runtime-Qualitaet, Warnungen und Court-Kalibrierung.</p>
        </div>
        <button onClick={onBack} className="self-start md:self-auto inline-flex items-center gap-2 text-sm text-slate-400 hover:text-slate-200">
          <ArrowLeft className="w-4 h-4" /> Zurueck
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Metric label="Kameras" value={`${quality?.connected_cameras || 0}/${quality?.total_cameras || 0}`} />
        <Metric label="Events" value={quality?.events || 0} />
        <Metric label="Pending Reviews" value={quality?.pending_reviews || 0} />
        <Metric label="Jobs" value={quality?.jobs ? Object.keys(quality.jobs).length : 0} />
      </div>

      {error && <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">{error}</div>}
      {message && <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-200">{message}</div>}

      <section className="glass p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-slate-100">Warnungen</h3>
          <button onClick={load} className="inline-flex items-center gap-2 text-xs text-slate-300 hover:text-white">
            <RefreshCw className="w-4 h-4" /> Aktualisieren
          </button>
        </div>
        {quality?.warnings?.length ? (
          <div className="space-y-2">
            {quality.warnings.map((warning, index) => (
              <div key={index} className="rounded border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-200">{warning}</div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-slate-500">Keine aktuellen Warnungen.</p>
        )}
      </section>

      <section className="glass p-5">
        <h3 className="font-semibold text-slate-100 mb-4">Kameras</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {cameras.map(([camId, info]) => (
            <div key={camId} className="rounded-lg border border-slate-700 bg-slate-900/50 p-3">
              <div className="flex items-center justify-between">
                <span className="font-semibold">{camId}</span>
                <span className={info.connected ? 'text-emerald-300 text-xs' : 'text-rose-300 text-xs'}>
                  {info.connected ? 'verbunden' : 'getrennt'}
                </span>
              </div>
              <p className="text-xs text-slate-500 truncate mt-1">{info.source}</p>
              <p className="text-xs text-slate-400 mt-2">{info.resolution?.[0] || 0}x{info.resolution?.[1] || 0} · {(info.fps || 0).toFixed(1)} FPS</p>
            </div>
          ))}
          {cameras.length === 0 && <p className="text-sm text-slate-500">Keine Live-Kameras aktiv.</p>}
        </div>
      </section>

      <section className="glass p-5 space-y-4">
        <h3 className="font-semibold text-slate-100">Court Calibration</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <label>
            <span className="block text-sm text-slate-300 mb-1">Video-Punkte</span>
            <textarea value={videoPoints} onChange={(event) => setVideoPoints(event.target.value)} rows={5} className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm font-mono outline-none focus:border-orange-400" />
          </label>
          <label>
            <span className="block text-sm text-slate-300 mb-1">Real-Punkte Meter</span>
            <textarea value={realPoints} onChange={(event) => setRealPoints(event.target.value)} rows={5} className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm font-mono outline-none focus:border-orange-400" />
          </label>
        </div>
        <button onClick={saveCalibration} className="btn-primary rounded-lg px-4 py-2 text-sm font-semibold inline-flex items-center gap-2">
          <Save className="w-4 h-4" /> Calibration speichern
        </button>
      </section>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="glass p-4">
      <p className="text-xs text-slate-500 uppercase tracking-wider">{label}</p>
      <p className="text-2xl font-bold text-slate-100">{value}</p>
    </div>
  );
}

export default QualityDashboard;
