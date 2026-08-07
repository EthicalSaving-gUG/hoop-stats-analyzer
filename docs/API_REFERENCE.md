# API Reference

Kurzreferenz fuer interne Agenten und Integrationen. Die API ist FastAPI-basiert
und startet standardmaessig auf `http://localhost:8000`.

## Health

### `GET /health`

Liefert Runtime-Status:

- `ultralytics_version`
- `selected_model`
- `selected_model_exists`
- `hoop_model`
- `grounding`
- `jobs`
- `game_running`

## Offline Analyse

### `POST /analyze/file`

Multipart:

- `file`: Video, erlaubte Endungen: `.mp4`, `.mov`, `.m4v`, `.avi`, `.mkv`, `.webm`

Antwort:

```json
{"status": "success", "job_id": "..."}
```

### `GET /analyze/progress/{job_id}`

Antwort enthaelt:

- `progress`
- `status`
- `result_file`
- `error`

### `POST /analyze/visualize`

Multipart:

- `file`: Video
- `mode`: `wireframe` oder `overlay`

### `POST /analyze/export3d`

Exportiert primitive 3D-Daten als JSON.

## Live Game

### `POST /game/start`

Beispiel:

```json
{
  "cameras": [{"id": "cam1", "source": "uploads/game.mp4"}],
  "agents": ["shot_clock", "game_clock", "scoresheet", "referee"],
  "team_home": "Team Schwarz",
  "team_away": "Team Gelb",
  "ruleset": "fiba",
  "processing_fps": 10,
  "pose_model": "mediapipe"
}
```

### `GET /game/state`

Liefert Snapshot aus `GameState`, aktive Agenten und Kamera-Infos.

### `POST /game/stop`

Stoppt StreamManager und alle Agenten.

### `GET /game/highlights`

Erzeugt Clips aus Events wie `SHOT_MAKE`, `VIOLATION_ALERT`, `REFEREE_WHISTLE`.

## Manual Mode

### `POST /manual/start`

Initialisiert Teams, Spieler und optionales Video.

### `POST /manual/update-player`

Erlaubte Felder:

- `points_2pt`
- `points_3pt`
- `points_ft`
- `fouls`
- `rebounds_off`
- `rebounds_def`
- `assists`
- `steals`
- `blocks`
- `turnovers`
- `minutes`

### `POST /manual/upload-video`

Multipart-Video-Upload fuer den manuellen Review.

## Grounding

### `GET /grounding/status`

Zeigt, ob LocateAnything-Grounding aktiv ist.

### `POST /grounding/image`

Multipart:

- `file`: Bildframe, erlaubte Endungen: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`
- `query`: Sprachprompt, z.B. `basketball rim`
- `task`: `ground_multi`, `ground_single`, `detect`, `ground_text`, `detect_text`, `point`
- `mode`: `hybrid`, `fast`, `slow`

Antwort:

```json
{
  "image": "frame_abc123.png",
  "width": 1280,
  "height": 720,
  "query": "basketball rim",
  "boxes": [{"x1": 100.0, "y1": 50.0, "x2": 180.0, "y2": 120.0}],
  "points": [],
  "annotation_file": "grounding_frame_abc123.json"
}
```

### `POST /grounding/export-yolo/{annotation_file}`

Konvertiert die gespeicherten Pixel-Boxen einer Grounding-Annotation in YOLO-TXT.
Der Export landet unter `results/labels/`.

Query:

- `class_id`: numerische YOLO-Klasse, Default `0`

Antwort:

```json
{
  "status": "success",
  "annotation_file": "grounding_frame_abc123.json",
  "label_file": "grounding_frame_abc123.txt",
  "labels": 1
}
```

### `POST /grounding/annotation/{annotation_file}/boxes/{box_id}/review`

Speichert Review-Status pro Grounding-Box.

Body:

```json
{"review_status": "accepted", "review_note": "rim ok"}
```

Erlaubte Statuswerte: `pending`, `accepted`, `rejected`, `needs_review`.

## Review

### `GET /review/events`

Listet Live-Events und optional Offline-Events aus einem Analyse-Resultat.

Query:

- `result_file`: optionaler Dateiname unter `results/`

### `POST /review/events/{event_id}`

Speichert Review-Status eines Events.

```json
{"review_status": "accepted", "review_note": "checked"}
```

## Video Frames

### `POST /video/extract-frame`

Extrahiert einen Frame aus einem lokalen oder hochgeladenen Video.

```json
{"video_url": "uploads/game.mp4", "timestamp": 12.5}
```

Antwort enthaelt `image_url` unter `/results/frames/...`.

## Calibration & Quality

### `GET /calibration`

Liefert aktuell gespeicherte Court-Kalibrierung.

### `POST /calibration`

Speichert Matching-Punkte:

```json
{
  "video": [[100, 100], [1100, 100], [100, 900], [1100, 900]],
  "real": [[0, 0], [28, 0], [0, 15], [28, 15]]
}
```

### `GET /quality`

Liefert Kamera-, Job-, Review- und Warnstatus fuer das Datenqualitaets-Dashboard.
