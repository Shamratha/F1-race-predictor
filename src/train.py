"""Train and evaluate the F1 podium-prediction model.

Model:   ColumnTransformer(one-hot team) -> XGBoost classifier.
         Circuit is deliberately NOT a feature: it is constant for every driver
         in a race, so it cannot help decide *who* podiums -- it only lets the
         model overfit. The signal that ranks drivers is grid, pace, form & car.

Eval:    expanding-window, the way it would really be used. For each season we
         train only on earlier seasons and predict that season, then pool every
         race. No future data ever leaks into a prediction.

Key metric:  Precision@3 -- of the 3 drivers we predict for the podium in each
             race, how many actually finished on the podium. Compared against a
             strong "just use the starting grid" baseline.

Outputs (in models/):
    model.joblib        the fitted pipeline (trained on ALL seasons)
    metadata.json       feature lists, categories, metrics
    shap_summary.png    global feature-importance explanation
"""
import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "dataset.csv")
MODELS = os.path.join(ROOT, "models")
os.makedirs(MODELS, exist_ok=True)

NUMERIC = [
    "grid_position", "quali_gap_to_pole_s",
    "air_temp", "track_temp", "humidity", "rainfall", "wind_speed",
    "driver_form_last5", "driver_podium_rate_last5",
    "constructor_form_last5", "driver_points_before",
    "driver_places_gained_last5",       # race-pace edge the grid doesn't show
    "driver_dnf_rate_last5", "constructor_dnf_rate_last5",  # reliability
]
CATEGORICAL = ["team"]  # circuit excluded on purpose (see module docstring)
TARGET = "podium"


def make_model(y_train):
    """A regularized XGBoost pipeline that one-hot encodes the team."""
    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",  # numeric cols pass through; XGBoost handles NaN
    )
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=3.0,
        gamma=0.5,
        scale_pos_weight=neg / max(pos, 1),
        eval_metric="logloss",
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def precision_at_3(df, proba):
    """Mean fraction of the top-3 predicted drivers per race that got a podium."""
    tmp = df.copy()
    tmp["proba"] = proba
    hits, races = 0, 0
    for _, g in tmp.groupby(["season", "circuit"]):
        top3 = g.sort_values("proba", ascending=False).head(3)
        hits += int(top3["podium"].sum())
        races += 1
    return hits / (races * 3), races


def main():
    df = pd.read_csv(DATA)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    seasons = sorted(int(s) for s in df["season"].unique())
    print(f"Seasons: {seasons}  ({len(df)} rows)")

    # --- Expanding-window evaluation (no future leakage) ---
    # For each season from the 2nd onward, train on all earlier seasons and
    # predict that season. Pool every out-of-sample race for a stable estimate.
    oos = []          # collected test rows with their predicted probability
    all_proba = np.full(len(df), np.nan)
    for test_season in seasons[1:]:
        tr = df[df["season"] < test_season]
        te = df[df["season"] == test_season]
        m = make_model(tr[TARGET])
        m.fit(tr[NUMERIC + CATEGORICAL], tr[TARGET])
        p = m.predict_proba(te[NUMERIC + CATEGORICAL])[:, 1]
        all_proba[te.index] = p
        block = te.copy()
        block["proba"] = p
        oos.append(block)

    oos = pd.concat(oos).reset_index(drop=True)
    grid_score = -oos["grid_position"].fillna(99).values  # lower grid = higher rank
    auc = roc_auc_score(oos[TARGET], oos["proba"])
    ap = average_precision_score(oos[TARGET], oos["proba"])
    p_at_3, n_races = precision_at_3(oos, oos["proba"].values)
    grid_auc = roc_auc_score(oos[TARGET], grid_score)
    grid_p3, _ = precision_at_3(oos, grid_score)

    print(f"\n=== EXPANDING-WINDOW PERFORMANCE "
          f"(seasons {seasons[1]}-{seasons[-1]}, {n_races} races) ===")
    print(f"                          model    grid-only baseline")
    print(f"ROC-AUC (ranking) ....... {auc:.3f}    {grid_auc:.3f}"
          f"   <- model wins on full-field ranking")
    print(f"PR-AUC (avg precision) .. {ap:.3f}    -")
    print(f"Precision@3 (exact top3)  {p_at_3:.3f}    {grid_p3:.3f}"
          f"   <- ~tie: the grid is a very strong signal")

    # --- Train the deployed model on ALL data ---
    full_model = make_model(df[TARGET])
    full_model.fit(df[NUMERIC + CATEGORICAL], df[TARGET])
    joblib.dump(full_model, os.path.join(MODELS, "model.joblib"))
    X_test = oos[NUMERIC + CATEGORICAL]  # used below for SHAP

    # --- SHAP global explanation (on transformed test features) ---
    try:
        Xt = full_model.named_steps["pre"].transform(X_test)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        feat_names = [
            n.replace("remainder__", "").replace("cat__team_", "team: ")
            for n in full_model.named_steps["pre"].get_feature_names_out()
        ]
        explainer = shap.TreeExplainer(full_model.named_steps["clf"])
        sv = explainer.shap_values(Xt)
        plt.figure()
        shap.summary_plot(sv, Xt, feature_names=feat_names, show=False,
                          max_display=12, plot_size=(9, 6))
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS, "shap_summary.png"), dpi=120)
        plt.close()
        print("\nSaved SHAP summary -> models/shap_summary.png")
    except Exception as e:
        print(f"\nSHAP plot skipped ({type(e).__name__}: {e})")

    # --- Metadata for the app ---
    meta = dict(
        numeric=NUMERIC,
        categorical=CATEGORICAL,
        teams=sorted(df["team"].dropna().unique().tolist()),
        circuits=sorted(df["circuit"].dropna().unique().tolist()),
        metrics=dict(roc_auc=round(auc, 3), pr_auc=round(ap, 3),
                     precision_at_3=round(p_at_3, 3),
                     grid_baseline_p3=round(grid_p3, 3),
                     grid_baseline_auc=round(grid_auc, 3),
                     eval="expanding-window",
                     eval_seasons=[int(seasons[1]), int(seasons[-1])],
                     n_test_races=int(n_races)),
        seasons=[int(s) for s in seasons],
        n_rows=int(len(df)),
    )
    with open(os.path.join(MODELS, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved model -> models/model.joblib and metadata.json")


if __name__ == "__main__":
    main()
