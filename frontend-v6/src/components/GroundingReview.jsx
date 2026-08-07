import { useEffect, useMemo, useState } from 'react';
import { ArrowLeft, Check, Crosshair, Download, ImageUp, Loader2, Search, SquareDashedMousePointer, X } from 'lucide-react';
import axios from 'axios';
import { API_BASE } from '../config';

const TASKS = [
  { value: 'ground_multi', label: 'Mehrere Treffer' },
  { value: 'ground_single', label: 'Ein Treffer' },
  { value: 'detect', label: 'Kategorien' },
  { value: 'ground_text', label: 'Text finden' },
  { value: 'detect_text', label: 'Alle Texte' },
  { value: 'point', label: 'Punkt zeigen' },
];

const EXAMPLE_PROMPTS = [
  'basketball rim',
  'shot clock',
  'scoreboard',
  'referee hand signal',
];

function GroundingReview({ onBack }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [query, setQuery] = useState('basketball rim');
  const [task, setTask] = useState('ground_multi');
  const [mode, setMode] = useState('hybrid');
  const [videoSource, setVideoSource] = useState('');
  const [timestamp, setTimestamp] = useState('0');
  const [status, setStatus] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [exportStatus, setExportStatus] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    axios.get(`${API_BASE}/grounding/status`)
      .then(res => {
        if (!cancelled) setStatus(res.data);
      })
      .catch(() => {
        if (!cancelled) setError('Grounding-Status konnte nicht geladen werden.');
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!file) {
      setPreviewUrl('');
      return undefined;
    }
    const nextUrl = URL.createObjectURL(file);
    setPreviewUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);

  const scale = useMemo(() => {
    if (!result?.width || !result?.height) return { x: 1, y: 1 };
    return { x: 100 / result.width, y: 100 / result.height };
  }, [result]);

  const runGrounding = async () => {
    if (!file) {
      setError('Bitte zuerst ein Bild oder einen extrahierten Frame auswählen.');
      return;
    }
    setLoading(true);
    setError('');
    setExportStatus('');
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('query', query);
    formData.append('task', task);
    formData.append('mode', mode);

    try {
      const res = await axios.post(`${API_BASE}/grounding/image`, formData);
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Grounding fehlgeschlagen.');
    } finally {
      setLoading(false);
    }
  };

  const extractFrame = async () => {
    if (!videoSource.trim()) {
      setError('Bitte Video-URL oder lokalen Pfad eintragen.');
      return;
    }
    setLoading(true);
    setError('');
    setExportStatus('');
    try {
      const res = await axios.post(`${API_BASE}/video/extract-frame`, {
        video_url: videoSource,
        timestamp: Number(timestamp) || 0,
      });
      const imageRes = await axios.get(res.data.image_url, { responseType: 'blob' });
      const extractedFile = new File([imageRes.data], res.data.image, { type: imageRes.data.type || 'image/jpeg' });
      setFile(extractedFile);
      setResult(null);
      setExportStatus(`Frame ${res.data.frame} extrahiert.`);
    } catch (err) {
      setError(err.response?.data?.detail || 'Frame-Extraktion fehlgeschlagen.');
    } finally {
      setLoading(false);
    }
  };

  const exportYolo = async () => {
    if (!result?.annotation_file) return;
    setExportStatus('');
    setError('');
    try {
      const res = await axios.post(`${API_BASE}/grounding/export-yolo/${result.annotation_file}`, null, {
        params: { accepted_only: true },
      });
      setExportStatus(`${res.data.labels} Label nach results/labels/${res.data.label_file} exportiert.`);
    } catch (err) {
      setError(err.response?.data?.detail || 'YOLO-Export fehlgeschlagen.');
    }
  };

  const updateBoxStatus = async (boxId, reviewStatus) => {
    if (!result?.annotation_file) return;
    try {
      const res = await axios.post(`${API_BASE}/grounding/annotation/${result.annotation_file}/boxes/${boxId}/review`, {
        review_status: reviewStatus,
      });
      setResult(prev => ({
        ...prev,
        boxes: prev.boxes.map(box => box.id === boxId ? res.data.box : box),
      }));
    } catch (err) {
      setError(err.response?.data?.detail || 'Box-Status konnte nicht gespeichert werden.');
    }
  };

  return (
    <div className="w-full max-w-6xl z-10 space-y-6">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-3">
            <Crosshair className="w-6 h-6 text-cyan-300" />
            LocateAnything Review
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Frame hochladen, Objekt/Text per Prompt lokalisieren und Annotation als JSON sichern.
          </p>
        </div>
        <button onClick={onBack} className="self-start md:self-auto inline-flex items-center gap-2 text-sm text-slate-400 hover:text-slate-200">
          <ArrowLeft className="w-4 h-4" /> Zurück
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[360px_1fr] gap-6">
        <section className="glass p-5 space-y-5">
          <div className="rounded-lg border border-slate-700 bg-slate-900/50 p-4">
            <p className="text-sm font-semibold text-slate-200">Backend</p>
            <p className={`text-xs mt-1 ${status?.enabled ? 'text-emerald-300' : 'text-amber-300'}`}>
              {status ? `${status.backend} · ${status.model}` : 'Status wird geladen...'}
            </p>
            {!status?.enabled && (
              <p className="text-xs text-slate-500 mt-2">
                Aktivierung per HOOP_STATS_GROUNDING_BACKEND=openai oder local.
              </p>
            )}
          </div>

          <label className="block">
            <span className="text-sm text-slate-300 mb-2 block">Frame/Bild</span>
            <div className="border border-dashed border-slate-600 rounded-lg p-4 bg-slate-900/40 hover:border-cyan-400/60 transition-colors">
              <div className="flex items-center gap-3">
                <ImageUp className="w-5 h-5 text-cyan-300" />
                <span className="text-sm text-slate-300 truncate">{file?.name || 'Bild auswählen'}</span>
              </div>
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(event) => setFile(event.target.files?.[0] || null)}
              />
            </div>
          </label>

          <div className="rounded-lg border border-slate-700 bg-slate-900/40 p-3 space-y-3">
            <p className="text-sm font-semibold text-slate-200">Frame aus Video extrahieren</p>
            <input
              value={videoSource}
              onChange={(event) => setVideoSource(event.target.value)}
              placeholder="uploads/game.mp4 oder Video-URL"
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm outline-none focus:border-cyan-400"
            />
            <div className="grid grid-cols-[1fr_auto] gap-2">
              <input
                value={timestamp}
                onChange={(event) => setTimestamp(event.target.value)}
                placeholder="Sekunde"
                className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm outline-none focus:border-cyan-400"
              />
              <button type="button" onClick={extractFrame} disabled={loading} className="rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-3 py-2 text-sm font-semibold text-cyan-100 disabled:opacity-60">
                Extrahieren
              </button>
            </div>
          </div>

          <div>
            <label className="text-sm text-slate-300 mb-2 block" htmlFor="grounding-query">Prompt</label>
            <input
              id="grounding-query"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm outline-none focus:border-cyan-400"
              placeholder="z.B. basketball rim"
              disabled={task === 'detect_text'}
            />
            <div className="flex flex-wrap gap-2 mt-3">
              {EXAMPLE_PROMPTS.map(prompt => (
                <button
                  key={prompt}
                  type="button"
                  onClick={() => setQuery(prompt)}
                  className="px-2.5 py-1 rounded border border-slate-700 bg-slate-800/70 text-xs text-slate-300 hover:border-cyan-500/60"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="text-sm text-slate-300 mb-2 block">Aufgabe</span>
              <select
                value={task}
                onChange={(event) => setTask(event.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm outline-none focus:border-cyan-400"
              >
                {TASKS.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
              </select>
            </label>
            <label className="block">
              <span className="text-sm text-slate-300 mb-2 block">Modus</span>
              <select
                value={mode}
                onChange={(event) => setMode(event.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm outline-none focus:border-cyan-400"
              >
                <option value="hybrid">hybrid</option>
                <option value="fast">fast</option>
                <option value="slow">slow</option>
              </select>
            </label>
          </div>

          <button
            type="button"
            onClick={runGrounding}
            disabled={loading}
            className="w-full btn-primary px-4 py-3 rounded-lg font-semibold inline-flex justify-center items-center gap-2 disabled:opacity-60"
          >
            {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Search className="w-5 h-5" />}
            Lokalisieren
          </button>

          {error && (
            <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
              {error}
            </div>
          )}
        </section>

        <section className="glass p-5 min-h-[520px]">
          <div className="relative bg-slate-950/70 border border-slate-800 rounded-lg min-h-[420px] flex items-center justify-center overflow-hidden">
            {previewUrl ? (
              <div className="relative max-w-full max-h-[680px]">
                <img src={previewUrl} alt="Grounding preview" className="max-w-full max-h-[680px] object-contain" />
                {result?.boxes?.map((box, index) => (
                  <div
                    key={`box-${index}`}
                    className="absolute border-2 border-cyan-300 shadow-[0_0_16px_rgba(103,232,249,0.45)]"
                    style={{
                      left: `${box.x1 * scale.x}%`,
                      top: `${box.y1 * scale.y}%`,
                      width: `${Math.max(0, box.x2 - box.x1) * scale.x}%`,
                      height: `${Math.max(0, box.y2 - box.y1) * scale.y}%`,
                    }}
                  />
                ))}
                {result?.points?.map((point, index) => (
                  <div
                    key={`point-${index}`}
                    className="absolute w-4 h-4 rounded-full bg-amber-300 border-2 border-slate-950 shadow-lg"
                    style={{
                      left: `${point.x * scale.x}%`,
                      top: `${point.y * scale.y}%`,
                      transform: 'translate(-50%, -50%)',
                    }}
                  />
                ))}
              </div>
            ) : (
              <div className="text-center text-slate-500">
                <SquareDashedMousePointer className="w-10 h-10 mx-auto mb-3" />
                <p className="text-sm">Noch kein Frame ausgewählt.</p>
              </div>
            )}
          </div>

          {result && (
            <>
              <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
                <Metric label="Boxen" value={result.boxes?.length || 0} />
                <Metric label="Punkte" value={result.points?.length || 0} />
                <Metric label="Annotation" value={result.annotation_file || '-'} compact />
              </div>
              <div className="mt-4 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                <button
                  type="button"
                  onClick={exportYolo}
                  disabled={!result.boxes?.length}
                  className="inline-flex items-center justify-center gap-2 rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-4 py-2 text-sm font-semibold text-cyan-100 hover:bg-cyan-500/20 disabled:opacity-50"
                >
                  <Download className="w-4 h-4" />
                  YOLO-Label exportieren
                </button>
                {exportStatus && <p className="text-xs text-emerald-300">{exportStatus}</p>}
              </div>
              {result.boxes?.length > 0 && (
                <div className="mt-4 space-y-2">
                  {result.boxes.map((box, index) => (
                    <div key={box.id || index} className="rounded-lg border border-slate-700 bg-slate-900/50 p-3 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                      <div className="text-xs text-slate-300">
                        <span className="font-semibold">Box {index + 1}</span>
                        <span className="ml-2 text-slate-500">{box.review_status || 'pending'}</span>
                      </div>
                      <div className="flex gap-2">
                        <button onClick={() => updateBoxStatus(box.id, 'accepted')} className="inline-flex items-center gap-1 rounded border border-emerald-500/30 bg-emerald-500/10 px-2 py-1 text-xs text-emerald-200">
                          <Check className="w-3 h-3" /> akzeptieren
                        </button>
                        <button onClick={() => updateBoxStatus(box.id, 'rejected')} className="inline-flex items-center gap-1 rounded border border-rose-500/30 bg-rose-500/10 px-2 py-1 text-xs text-rose-200">
                          <X className="w-3 h-3" /> verwerfen
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </section>
      </div>
    </div>
  );
}

function Metric({ label, value, compact = false }) {
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900/50 p-3 min-w-0">
      <p className="text-slate-500 text-xs uppercase tracking-wider">{label}</p>
      <p className={`text-slate-100 font-semibold mt-1 ${compact ? 'text-xs truncate' : 'text-xl'}`}>{value}</p>
    </div>
  );
}

export default GroundingReview;
