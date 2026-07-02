# 🏁 F1 Podium Predictor

**Machine-learning Formula 1 podium predictions** from the *changing factors* that
decide a race — starting grid, qualifying pace, recent form, car reliability and
the weather. Built on **real** [FastF1](https://docs.fastf1.dev/) data (2021–2024),
served through a **FastAPI** backend and a custom dark-themed dashboard.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009485)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-EB5E28)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 What it does

For any Grand Prix from 2021–2024, the model gives **every driver a probability of
finishing on the podium (top 3)** and ranks the field. Then you can play with the
conditions and watch the prediction react:

- **🏎️ Pick a race** — any round from four seasons of real data
- **🏆 Predicted podium** — top-3 drivers with podium probability, checked against the *actual* result
- **📊 Full-grid probabilities** — all 20 drivers ranked, with podium-chance bars
- **🏗️ Constructor standings** — predicted points per team for that race
- **⚙️ What-if scenarios** — force a wet race, change track temperature/wind, or hand a
  driver a **grid penalty**, and see the podium re-shuffle live
- **🔍 Explainability** — a SHAP plot shows exactly what the model learned

---

## 🧠 How it works

```
                    FastF1 (real F1 data, 2021–2024)
                                 │
            ┌────────────────────┴────────────────────┐
            │  src/build_dataset.py  → feature engineering
            │  grid · quali gap · form · pace · reliability · weather
            └────────────────────┬────────────────────┘
                                 │  data/dataset.csv  (1,799 driver-races)
                    src/train.py │  XGBoost + SHAP, expanding-window eval
                                 │  models/model.joblib
            ┌────────────────────┴────────────────────┐
            │            main.py  (FastAPI)            │
            │   /api/predict  →  probabilities + JSON  │
            └────────────────────┬────────────────────┘
                                 │
                   web/  (custom HTML/CSS/JS dashboard)
```

### The features (the "changing factors")

| Factor | Features engineered |
|---|---|
| **Qualifying** | starting grid position, gap to pole (seconds) |
| **Recent form** | driver's avg finish & podium-rate over last 5 races, championship points so far |
| **Race pace** | avg places gained/lost on Sundays (signal the grid doesn't contain) |
| **Reliability** | driver & constructor DNF-rate over last 5 races |
| **Constructor** | team (one-hot), constructor recent form |
| **Weather** | air/track temp, humidity, rainfall, wind |

---

## 📈 Model performance (evaluated honestly)

Evaluation is **expanding-window**: each season is predicted using *only earlier
seasons*, so no future data ever leaks in. Pooled over 68 out-of-sample races:

| Metric | Model | Grid-only baseline |
|---|---|---|
| **ROC-AUC** (full-field ranking) | **0.918** ✅ | 0.902 |
| **PR-AUC** | 0.614 | — |
| **Precision@3** (exact top-3) | 0.60 | 0.61 |

**The honest story:** in F1 the starting grid is a *very* strong predictor, so the
model only ~ties it on picking the exact top-3. But it **beats the grid on
full-field ranking quality (AUC 0.918 vs 0.902)** because it also reads pace, form,
reliability and weather — which is what lets it produce calibrated probabilities and
respond to changing conditions.

**What SHAP says it learned** (in order of impact):
`grid position → qualifying pace → championship form → constructor strength` —
exactly what a race engineer would tell you, learned purely from data.

---

## 🛠️ Tech stack

- **Data:** FastF1 (real telemetry, results, qualifying, weather)
- **ML:** scikit-learn pipeline + XGBoost, SHAP for explainability
- **Backend:** FastAPI + Uvicorn
- **Frontend:** vanilla HTML/CSS/JS (no build step) — dark F1-themed dashboard
- **Deploy:** Render (single web service)

---

## 🚀 Run it locally

```bash
# 1. create a venv and install runtime deps
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 2. start the app (model + dataset are committed, so it runs immediately)
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** — the dashboard and API are both served there.
API docs (Swagger) are at **http://localhost:8000/docs**.

### Rebuild the data & model from scratch (optional)

```bash
pip install -r requirements-dev.txt
python src/build_dataset.py 2021 2024    # pulls real data via FastF1 (cached locally)
python src/train.py                      # trains, evaluates, writes models/
```

---

## ☁️ Deploy to Render

The repo ships committed `models/` and `data/dataset.csv`, so the deployed app never
has to hit the F1 API or retrain.

1. Push this repo to GitHub.
2. On [Render](https://render.com): **New → Blueprint**, pick the repo. `render.yaml`
   configures everything (build, start command, Python version).
3. Done — Render serves the FastAPI app on your URL.

---

## 📁 Project structure

```
f1 race prediction/
├── main.py                 # FastAPI backend + serves the dashboard
├── web/                    # custom dashboard (index.html, styles.css, app.js)
├── src/
│   ├── build_dataset.py    # FastF1 → feature-engineered dataset
│   └── train.py            # XGBoost training, evaluation, SHAP
├── models/                 # model.joblib, metadata.json, shap_summary.png
├── data/dataset.csv        # 1,799 driver-races (data/cache/ is git-ignored)
├── requirements.txt        # lean runtime deps (for deploy)
├── requirements-dev.txt    # + fastf1, shap, matplotlib (for rebuilding)
└── render.yaml             # Render Blueprint
```

---

## 🙏 Acknowledgments

- [FastF1](https://docs.fastf1.dev/) — the F1 data that makes this possible
- Formula 1 timing data is © FOM; used here for a non-commercial educational project

## 📄 License

MIT
