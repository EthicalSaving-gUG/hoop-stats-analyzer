# Roadmap

Diese Roadmap trennt kurzfristig umsetzbare technische Arbeit von groesseren
Produkt- und Forschungsaufgaben. Statuswerte:

- `done`: im aktuellen Stand umgesetzt.
- `next`: hoechste Prioritaet fuer die naechste Iteration.
- `planned`: sinnvoll, aber nach `next`.
- `research`: braucht Datensaetze, Modellvergleiche oder Hardwaretests.

## Done

| Thema | Ergebnis |
| --- | --- |
| YOLO26-freundliche Modellwahl | Backend bevorzugt lokale `yolo26n.pt` und faellt kontrolliert zurueck. |
| Sichere Upload-Pfade | Video-Uploads nutzen erlaubte Endungen und sichere Dateinamen. |
| Backend Health | `/health` zeigt Modell-, Grounding- und Jobstatus. |
| LocateAnything Review-Pfad | Optionales Grounding fuer Einzelbilder mit JSON-Annotationen. |
| Annotation-to-YOLO Export | Grounding-Boxen koennen als YOLO-TXT unter `results/labels/` exportiert werden. |
| Frontend API-Konfiguration | `VITE_API_BASE_URL` ersetzt hart codierte API-URL. |
| Agenten-Dokumentation | `docs/AGENT_GUIDE.md` beschreibt Architektur und Arbeitsvertraege. |
| GameState-Konsolidierung | Doppelte `player_fouls`-Initialisierung entfernt. |
| Review-Queue fuer AI-Events | Live- und Offline-Events koennen bestaetigt, verworfen und markiert werden. |
| Frame-Extraktion | Videos koennen serverseitig zu Bildframes fuer Grounding extrahiert werden. |
| Annotation Review Status | Grounding-Boxen speichern `accepted`, `rejected`, `pending` oder `needs_review`. |
| Datenqualitaets-Dashboard | UI und API zeigen Kamera-, Job-, Warn- und Review-Status. |
| Konfigurierbare Court Calibration | Court-Punkte koennen ueber API/UI gespeichert und von Analysen genutzt werden. |
| Agent Confidence Contract | Events enthalten Confidence, Evidence, Review-Status und Kameraquelle. |

## Next

| Prioritaet | Thema | Akzeptanzkriterium |
| --- | --- | --- |
| P0 | Review-Queue UX verfeinern | Bulk-Aktionen, Filter und Export der bestaetigten Events. |
| P0 | Visuelle Calibration | Court-Punkte direkt auf einem Frame anklicken statt JSON manuell einzugeben. |
| P1 | Annotation Editor | Grounding-Boxen ziehen/resizen statt nur Status setzen. |
| P1 | Datenqualitaets-Metriken erweitern | Detektionsrate, OCR-Rate und Frame-Drop-Rate pro Quelle messen. |

## Planned

| Thema | Nutzen |
| --- | --- |
| Persistenter Job Store | Analysejobs ueber Server-Neustarts sichtbar halten. |
| SQLite/Parquet Export | Reproduzierbare Auswertung statt nur JSON/CSV-Dateien. |
| Multi-Camera Sync | Saubere Zuordnung von Events ueber mehrere Kameraquellen. |
| Highlight Review Timeline | Event-basierte Clips und manuelle Korrektur an einer Timeline. |
| Modell-Benchmark Matrix | YOLO26, YOLOE, eigenes Hoop-Modell und Ball-Modell vergleichbar machen. |
| Install-/Run-Healthcheck | Windows-Zertifikate, ffmpeg, CUDA, Node/Python Versionen pruefen. |

## Research

| Thema | Fragestellung |
| --- | --- |
| Ball-spezifisches Modell | Wie viele gelabelte Hallenframes braucht robuste Ball-Erkennung? |
| Referee Signals | Welche Signale sind aus realen Kameraperspektiven unterscheidbar? |
| LocateAnything vs YOLOE | Welches Tool bootstrapped Labels schneller und mit weniger Korrekturaufwand? |
| Pose-basierte Wurfqualitaet | Welche biomechanischen Metriken sind bei Weitwinkelvideos belastbar? |
| Echtzeit auf Consumer-Hardware | Welche Pipeline laeuft stabil auf CPU/iGPU/RTX-Klassen? |

## Implementierungsprinzip

Produktiv wird nicht das Modell mit der spektakulaersten Demo, sondern der Pfad mit
Review, Messbarkeit und reproduzierbaren Daten. Deshalb:

1. Live-Pfad schlank halten: YOLO/Tracker/Event-Heuristiken.
2. Offline-Pfad fuer schwere VLMs und Label-Bootstrapping nutzen.
3. Menschliche Review als offizielles Gate vor Statistikexport.
4. Jede Modellverbesserung gegen echte Hallenvideos messen.
