import { useEffect, useState } from 'react';
import { ArrowLeft, Check, RefreshCw, X } from 'lucide-react';
import axios from 'axios';
import { API_BASE } from '../config';

const STATUS_ACTIONS = [
  { status: 'accepted', label: 'Akzeptieren', icon: Check, className: 'text-emerald-300 border-emerald-500/30 bg-emerald-500/10' },
  { status: 'rejected', label: 'Verwerfen', icon: X, className: 'text-rose-300 border-rose-500/30 bg-rose-500/10' },
  { status: 'needs_review', label: 'Pruefen', icon: RefreshCw, className: 'text-amber-300 border-amber-500/30 bg-amber-500/10' },
];

function ReviewQueue({ onBack, resultFile }) {
  const [events, setEvents] = useState([]);
  const [inputResultFile, setInputResultFile] = useState(resultFile || '');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const fetchEvents = async () => {
    setLoading(true);
    setError('');
    try {
      const params = inputResultFile ? { result_file: inputResultFile } : {};
      const res = await axios.get(`${API_BASE}/review/events`, { params });
      setEvents(res.data.events || []);
    } catch (err) {
      setError(err.response?.data?.detail || 'Review-Events konnten nicht geladen werden.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEvents();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const updateStatus = async (eventId, reviewStatus) => {
    await axios.post(`${API_BASE}/review/events/${encodeURIComponent(eventId)}`, {
      review_status: reviewStatus,
    });
    setEvents(prev => prev.map(evt => evt.id === eventId ? { ...evt, review_status: reviewStatus } : evt));
  };

  const counts = events.reduce((acc, evt) => {
    acc[evt.review_status || 'pending'] = (acc[evt.review_status || 'pending'] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="w-full max-w-6xl z-10 space-y-6">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-2xl font-bold">Review-Queue</h2>
          <p className="text-sm text-slate-400 mt-1">AI-Events bestaetigen, verwerfen oder fuer manuelle Pruefung markieren.</p>
        </div>
        <button onClick={onBack} className="self-start md:self-auto inline-flex items-center gap-2 text-sm text-slate-400 hover:text-slate-200">
          <ArrowLeft className="w-4 h-4" /> Zurueck
        </button>
      </div>

      <div className="glass p-4 flex flex-col md:flex-row gap-3 md:items-end">
        <label className="flex-1">
          <span className="block text-sm text-slate-300 mb-1">Optionales Analyse-Ergebnis</span>
          <input
            value={inputResultFile}
            onChange={(event) => setInputResultFile(event.target.value)}
            placeholder="z.B. stats_upload_abcd1234.json"
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm outline-none focus:border-orange-400"
          />
        </label>
        <button onClick={fetchEvents} disabled={loading} className="btn-primary rounded-lg px-4 py-2 text-sm font-semibold inline-flex items-center gap-2 disabled:opacity-60">
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Laden
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Metric label="Gesamt" value={events.length} />
        <Metric label="Pending" value={counts.pending || 0} />
        <Metric label="Accepted" value={counts.accepted || 0} />
        <Metric label="Rejected" value={counts.rejected || 0} />
      </div>

      {error && <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">{error}</div>}

      <div className="space-y-3">
        {events.map(evt => (
          <div key={evt.id} className="glass p-4">
            <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-semibold text-slate-100">{evt.type}</span>
                  <span className="text-xs text-slate-500">{evt.agent}</span>
                  <span className="text-xs rounded border border-slate-700 px-2 py-0.5 text-slate-300">{evt.review_status || 'pending'}</span>
                  {evt.confidence != null && <span className="text-xs text-cyan-300">Conf {Math.round(evt.confidence * 100)}%</span>}
                  {evt.frame_ref != null && <span className="text-xs text-slate-400">Frame {evt.frame_ref}</span>}
                </div>
                <pre className="mt-2 max-h-24 overflow-auto rounded bg-slate-950/50 p-2 text-xs text-slate-400">{JSON.stringify(evt.data || {}, null, 2)}</pre>
              </div>
              <div className="flex flex-wrap gap-2 shrink-0">
                {STATUS_ACTIONS.map(action => {
                  const Icon = action.icon;
                  return (
                    <button
                      key={action.status}
                      onClick={() => updateStatus(evt.id, action.status)}
                      className={`inline-flex items-center gap-1.5 rounded border px-3 py-1.5 text-xs font-semibold ${action.className}`}
                    >
                      <Icon className="w-3.5 h-3.5" />
                      {action.label}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        ))}
        {events.length === 0 && !loading && (
          <div className="glass p-8 text-center text-slate-500">Keine Review-Events vorhanden.</div>
        )}
      </div>
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

export default ReviewQueue;
