# 🔄 Hoop-Stats Pipeline Optimization Report
> **Generiert am:** 21.10.2026 21:27:14 UTC  
> **Status:** 🟡 Baseline bereits optimal

---

### 📊 Benchmark-Zusammenfassung

| Kennzahl | Vor Optimierung (Baseline) | Nach Optimierung | Veränderung |
| :--- | :--- | :--- | :--- |
| **Gesamt-Genauigkeit (Score)** | **95.30%** | **95.30%** | **+0.00%** |
| **Punkte-Genauigkeit** | 97.00% | 97.00% | +0.0% |
| **Treffer/Fehlwurf (FGM)** | 95.00% | 95.00% | +0.0% |
| **Schrittfehler (Traveling)** | 94.00% | 94.00% | +0.0% |
| **Spieler-Recall (Trikot-OCR)** | 93.00% | 93.00% | +0.0% |

---

### ⚙️ Optimierte Parameter (`config/pipeline_params.json`)

| Parameter | Baseline | Neu / Optimal | Auswirkung |
| :--- | :--- | :--- | :--- |
| `rim_confidence` | `0.45` | `0.45` | Filtert Korb-False-Positives bei Wurfversuchen |
| `gather_step_tolerance` | `2.2` | `2.2` | FIBA Art. 25 Schrittfehler-Erkennung (Schritt 0 Heuristik) |
| `ocr_voting_threshold` | `0.75` | `0.75` | Trikotnummern-Zuweisung an Tracklets |
| `shot_trajectory_thresh` | `0.5` | `0.5` | Ballflugkurven-Apex-Erkennung |

---

### 📂 Geprüfte Ground-Truth Spiele (2)
- **Spieltag 1**: SB Uffenheim vs SV Kürnach BasketWalls (11.10.2026) — Endstand: 57:?
- **Spieltag 2**: SB Uffenheim vs SG Oerlenbach/Ebenhausen 2 (17.10.2026) — Endstand: 55:49
