# F1 Podium Predictor

**Machine-learning Formula 1 podium predictions** from the *changing factors* that
decide a race — starting grid, qualifying pace, recent form, car reliability and
the weather. Built on **real** [FastF1](https://docs.fastf1.dev/) data (2021–2024),
served through a **FastAPI** backend and a custom dark-themed dashboard.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009485)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-EB5E28)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What it does

For any Grand Prix from 2021–2024, the model gives **every driver a probability of
finishing on the podium (top 3)** and ranks the field. Then you can play with the
conditions and watch the prediction react:

- **Pick a race** — any round from four seasons of real data
- **Predicted podium** — top-3 drivers with podium probability, checked against the *actual* result
- **Full-grid probabilities** — all 20 drivers ranked, with podium-chance bars
- **Constructor standings** — predicted points per team for that race
- **What-if scenarios** — force a wet race, change track temperature/wind, or hand a
  driver a **grid penalty**, and see the podium re-shuffle live
- **Live weather** — pull *current* real-world conditions at the circuit
  (OpenWeatherMap) with one click, then predict on them
- **Explainability** — a SHAP plot shows exactly what the model learned

---

## How it works

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

## Model performance (evaluated honestly)

Evaluation is **expanding-window**: each season is predicted using *only earlier
seasons*, so no future data ever leaks in. Pooled over 68 out-of-sample races.
(These numbers are computed entirely on **historical** FastF1 data. Live weather is
only a runtime what-if input in the dashboard — it never enters training or
evaluation, so it does not affect these metrics.)

| Metric | Model | Grid-only baseline |
|---|---|---|
| **ROC-AUC** (full-field ranking) | **0.918** | 0.902 |
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

## Tech stack

- **Data:** FastF1 (real telemetry, results, qualifying, weather)
- **ML:** scikit-learn pipeline + XGBoost, SHAP for explainability
- **Backend:** FastAPI + Uvicorn, organised into a **service layer**
  (`f1_data_service`, `prediction_service`, `weather_service`)
- **Frontend:** vanilla HTML/CSS/JS (no build step) — dark F1-themed dashboard
- **Live weather:** OpenWeatherMap (optional, opt-in via API key)
- **Testing:** pytest + coverage (29 tests, ~96% coverage)
- **Deploy:** Docker / Render (single web service)

---

## Run it locally

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

### Enable live weather (optional)

The **"Use live conditions"** button needs a free
[OpenWeatherMap](https://openweathermap.org/api) API key. Without one, everything
else still works and the button shows a friendly hint.

1. Create a free account at [openweathermap.org](https://openweathermap.org/api),
   **confirm your account via the verification email**, and copy your key from the
   **API keys** tab. (A new key can take up to ~2 hours to activate.)
2. Put it in a `.env` file next to `main.py` — it's loaded automatically and is
   git-ignored:

   ```bash
   cp .env.example .env        # Windows: copy .env.example .env
   # then edit .env:  OPENWEATHER_API_KEY=your_key_here
   ```
3. Restart the app and click **Use live conditions**.

Prefer an environment variable instead of a file? That works too:

```bash
export OPENWEATHER_API_KEY=your_key        # Windows PowerShell: $env:OPENWEATHER_API_KEY="your_key"
uvicorn main:app --reload --port 8000
```

### Run the tests

```bash
pip install -r requirements-dev.txt
pytest -v                                  # 29 tests
pytest --cov=. --cov-report=term-missing   # with coverage (~96%)
```

Tests cover the feature-engineering functions (incl. a **no-leakage** check), the
prediction service, the live-weather service (offline), and every API endpoint.

### Run with Docker

```bash
docker compose up --build                  # then open http://localhost:8000
```

### Rebuild the data & model from scratch (optional)

```bash
pip install -r requirements-dev.txt
python src/build_dataset.py 2021 2024    # pulls real data via FastF1 (cached locally)
python src/train.py                      # trains, evaluates, writes models/
```

---

## Deploy to Render

The repo ships committed `models/` and `data/dataset.csv`, so the deployed app never
has to hit the F1 API or retrain.

1. Push this repo to GitHub.
2. On [Render](https://render.com): **New → Blueprint**, pick the repo. `render.yaml`
   configures everything (build, start command, Python version).
3. Add `OPENWEATHER_API_KEY` as an environment variable in the Render dashboard to
   enable live weather (optional — it falls back gracefully without it).

---

## Project structure

```
f1 race prediction/
├── main.py                 # FastAPI app — thin routing layer
├── constants.py            # team colours, driver names, track coords, helpers
├── services/
│   ├── f1_data_service.py  # dataset access + race lookups
│   ├── prediction_service.py  # model + prediction payload
│   └── weather_service.py  # live OpenWeatherMap integration (opt-in)
├── web/                    # custom dashboard (index.html, styles.css, app.js)
├── src/
│   ├── features.py         # pure feature-engineering (unit-tested)
│   ├── build_dataset.py    # FastF1 → feature-engineered dataset
│   └── train.py            # XGBoost training, evaluation, SHAP
├── tests/                  # pytest suite (features, services, API)
├── models/                 # model.joblib, metadata.json, shap_summary.png
├── data/dataset.csv        # 1,799 driver-races (data/cache/ is git-ignored)
├── requirements.txt        # lean runtime deps (for deploy)
├── requirements-dev.txt    # + fastf1, shap, matplotlib, pytest (for dev)
├── Dockerfile              # production image
├── docker-compose.yml      # one-command local run
└── render.yaml             # Render Blueprint
```

---

## Acknowledgments

- [FastF1](https://docs.fastf1.dev/) — the F1 data that makes this possible
- Formula 1 timing data is © FOM; used here for a non-commercial educational project

## License

MIT
