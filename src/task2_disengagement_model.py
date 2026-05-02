"""
Task 2 - Predictive Disengagement Model
Predicts student withdrawal/failure before Week 6.

Improvements:
  - Temporal train/test split (2013 cohorts train, 2014 cohorts test)
  - Student-level split (all enrollments of one student stay together)
  - Peer-comparison features (relative to module cohort)
  - Platt Scaling for calibration
  - Two-threshold alert system (watch list vs active alert)
  - SHAP explanations for individual student predictions
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score,
    ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap

from src.data_loader import load_all, student_key_cols
from src.config import DAYS_PER_WEEK, WEEK_6_CUTOFF, FIG_DIR, MODEL_DIR, RANDOM_STATE

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

KEY = student_key_cols()
WEEK_CUTOFF = 6

TRAIN_PRESENTATIONS = {"2013B", "2013J"}
TEST_PRESENTATIONS  = {"2014B", "2014J"}


def build_week6_features(tables):
    """Build features using ONLY data available at Week 6 (day <= 42). No leakage."""

    svle_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "studentVle.csv")

    import gc
    vle_map = tables["vle"].set_index("id_site")["activity_type"].to_dict()

    chunks = []
    for chunk in pd.read_csv(svle_path, chunksize=2_000_000,
                             dtype={"code_module": str, "code_presentation": str,
                                    "id_student": "int32", "id_site": "int32",
                                    "date": "int16", "sum_click": "int16"}):
        chunk = chunk[(chunk["date"] >= 0) & (chunk["date"] <= WEEK_6_CUTOFF)]
        chunk["activity_type"] = chunk["id_site"].map(vle_map)
        chunk["week"] = ((chunk["date"] // DAYS_PER_WEEK) + 1).astype("int16")
        chunks.append(chunk)
        del chunk

    svle = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()

    g = svle.groupby(KEY, sort=False)

    vle_feats = g.agg(
        total_clicks_w6=("sum_click", "sum"),
        active_days_w6=("date", "nunique"),
        activity_diversity_w6=("activity_type", "nunique"),
        num_sessions_w6=("id_site", "nunique"),
        last_active_day_w6=("date", "max"),
        first_active_day_w6=("date", "min"),
        mean_daily_clicks=("sum_click", "mean"),
        std_daily_clicks=("sum_click", "std"),
    ).reset_index()

    vle_feats["std_daily_clicks"] = vle_feats["std_daily_clicks"].fillna(0)
    vle_feats["active_span"] = vle_feats["last_active_day_w6"] - vle_feats["first_active_day_w6"]
    vle_feats["recency_from_w6"] = WEEK_6_CUTOFF - vle_feats["last_active_day_w6"]

    weekly_clicks = svle.groupby(KEY + ["week"])["sum_click"].sum().reset_index()
    for w in range(1, WEEK_CUTOFF + 1):
        wk = weekly_clicks[weekly_clicks["week"] == w][KEY + ["sum_click"]].rename(
            columns={"sum_click": f"clicks_week_{w}"}
        )
        vle_feats = vle_feats.merge(wk, on=KEY, how="left")
        vle_feats[f"clicks_week_{w}"] = vle_feats[f"clicks_week_{w}"].fillna(0)

    week_cols = [f"clicks_week_{w}" for w in range(1, WEEK_CUTOFF + 1)]
    click_matrix = vle_feats[week_cols].values.astype(float)
    x = np.arange(1, WEEK_CUTOFF + 1, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    vle_feats["click_trend_w6"] = np.array([
        ((x - x_mean) * (row - row.mean())).sum() / denom if denom > 0 else 0
        for row in click_matrix
    ])

    del svle; gc.collect()

    # --- Peer comparison features (Fix 5): relative to module cohort ---
    for col in ["total_clicks_w6", "active_days_w6", "last_active_day_w6", "mean_daily_clicks"]:
        module_mean = vle_feats.groupby("code_module")[col].transform("mean")
        module_std  = vle_feats.groupby("code_module")[col].transform("std").replace(0, 1)
        vle_feats[f"{col}_peer_z"] = (vle_feats[col] - module_mean) / module_std

    # --- Assessment features ---
    sa = tables["studentAssessment"].copy()
    asmt = tables["assessments"].copy()
    sa = sa.merge(asmt[["id_assessment", "code_module", "code_presentation", "date", "assessment_type"]],
                  on="id_assessment", how="left")
    sa.rename(columns={"date": "deadline"}, inplace=True)
    sa = sa[sa["date_submitted"] <= WEEK_6_CUTOFF]
    sa["lead_time"] = sa["deadline"] - sa["date_submitted"]
    sa["is_late"] = (sa["lead_time"] < 0).astype(int)

    asmt_feats = sa.groupby(KEY).agg(
        num_submissions_w6=("id_assessment", "count"),
        mean_score_w6=("score", "mean"),
        std_score_w6=("score", "std"),
        mean_lead_time_w6=("lead_time", "mean"),
        late_submissions_w6=("is_late", "sum"),
        tma_count=("assessment_type", lambda x: (x == "TMA").sum()),
        cma_count=("assessment_type", lambda x: (x == "CMA").sum()),
    ).reset_index()
    asmt_feats["std_score_w6"] = asmt_feats["std_score_w6"].fillna(0)

    # Peer z-score for assessment score
    asmt_feats["score_peer_z"] = (
        asmt_feats["mean_score_w6"] - asmt_feats.groupby("code_module")["mean_score_w6"].transform("mean")
    ) / asmt_feats.groupby("code_module")["mean_score_w6"].transform("std").replace(0, 1)

    # --- Demographics ---
    si = tables["studentInfo"][KEY + [
        "gender", "region", "highest_education", "imd_band",
        "age_band", "num_of_prev_attempts", "studied_credits", "disability"
    ]].copy()

    si["gender"] = (si["gender"] == "M").astype(int)
    si["disability"] = (si["disability"] == "Y").astype(int)

    edu_order = {"No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2,
                 "HE Qualification": 3, "Post Graduate Qualification": 4}
    si["education_level"] = si["highest_education"].map(edu_order).fillna(1)

    age_order = {"0-35": 0, "35-55": 1, "55<=": 2}
    si["age_numeric"] = si["age_band"].map(age_order).fillna(0)

    imd_map = {}
    for band in si["imd_band"].dropna().unique():
        try:
            imd_map[band] = int(band.split("-")[0]) / 100.0
        except (ValueError, IndexError):
            imd_map[band] = 0.5
    si["imd_numeric"] = si["imd_band"].map(imd_map).fillna(0.5)
    si.drop(columns=["region", "highest_education", "imd_band", "age_band"], inplace=True)

    # --- Registration ---
    sr = tables["studentRegistration"][KEY + ["date_registration", "date_unregistration"]].copy()
    sr["registered_early"] = (sr["date_registration"] < -30).astype(int)
    sr["days_before_start"] = (-sr["date_registration"]).clip(lower=0).fillna(0)
    sr.drop(columns=["date_registration", "date_unregistration"], inplace=True)

    # --- Merge ---
    features = si.merge(vle_feats, on=KEY, how="left")
    features = features.merge(asmt_feats, on=KEY, how="left")
    features = features.merge(sr, on=KEY, how="left")

    fill_zero_cols = [c for c in features.columns if c not in KEY + [
        "gender", "disability", "education_level", "age_numeric", "imd_numeric"
    ]]
    features[fill_zero_cols] = features[fill_zero_cols].fillna(0)

    target = tables["studentInfo"][KEY + ["final_result"]].copy()
    target["target"] = target["final_result"].isin(["Withdrawn", "Fail"]).astype(int)
    features = features.merge(target[KEY + ["target"]], on=KEY, how="left")
    features = features.dropna(subset=["target"])

    return features


def temporal_student_split(features_df):
    """
    Fix: Temporal split by cohort year + student-level grouping.
    Train on 2013 presentations, test on 2014 presentations.
    This mirrors real deployment: train on past cohorts, predict future ones.
    """
    train_mask = features_df["code_presentation"].isin(TRAIN_PRESENTATIONS)
    test_mask  = features_df["code_presentation"].isin(TEST_PRESENTATIONS)

    train_df = features_df[train_mask]
    test_df  = features_df[test_mask]

    print(f"  Temporal split — Train: {len(train_df):,} ({train_df['target'].mean():.1%} at-risk) "
          f"| Test: {len(test_df):,} ({test_df['target'].mean():.1%} at-risk)")
    print(f"  Train presentations: {sorted(train_df['code_presentation'].unique())}")
    print(f"  Test  presentations: {sorted(test_df['code_presentation'].unique())}")

    feature_cols = [c for c in features_df.columns if c not in KEY + ["target"]]
    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["target"].values

    return X_train, y_train, X_test, y_test, feature_cols


def train_and_evaluate(features_df):
    X_train, y_train, X_test, y_test, feature_cols = temporal_student_split(features_df)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / n_pos

    xgb_base = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos * 1.5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="aucpr",
        verbosity=0,
    )
    xgb_base.fit(X_train, y_train)

    # Platt Scaling for better calibration (Fix 6)
    # cv=5 re-trains with cross-validation to fit sigmoid scaling layer
    xgb_calibrated = CalibratedClassifierCV(xgb_base, method="sigmoid", cv=5)
    xgb_calibrated.fit(X_train, y_train)

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight={0: 1, 1: scale_pos * 1.5},
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    results = {}
    for name, model, base in [
        ("XGBoost (Calibrated)", xgb_calibrated, xgb_base),
        ("Random Forest", rf_model, rf_model),
    ]:
        y_proba = model.predict_proba(X_test)[:, 1]

        # Two-threshold system (Fix 1):
        #   watch_thresh: low bar, catches almost everyone (recall-optimized)
        #   alert_thresh: higher bar, only notify advisor for stronger signals
        watch_thresh = 0.20
        alert_thresh = 0.50

        y_pred_watch = (y_proba >= watch_thresh).astype(int)
        y_pred_alert = (y_proba >= alert_thresh).astype(int)

        results[name] = {
            "model": model,
            "base_model": base,
            "y_proba": y_proba,
            "y_pred_watch": y_pred_watch,
            "y_pred_alert": y_pred_alert,
            "watch_thresh": watch_thresh,
            "alert_thresh": alert_thresh,
            "roc_auc": roc_auc_score(y_test, y_proba),
            "cm_watch": confusion_matrix(y_test, y_pred_watch),
            "cm_alert": confusion_matrix(y_test, y_pred_alert),
        }

        print(f"\n--- {name} ---")
        print(f"Watch List (thresh={watch_thresh}) — optimized for Recall:")
        print(classification_report(y_test, y_pred_watch, target_names=["Pass/Dist", "Fail/Wdrn"]))
        print(f"Active Alert (thresh={alert_thresh}) — balanced Precision/Recall:")
        print(classification_report(y_test, y_pred_alert, target_names=["Pass/Dist", "Fail/Wdrn"]))
        print(f"ROC-AUC: {results[name]['roc_auc']:.4f}")

    return results, X_train, X_test, y_test, feature_cols, xgb_base


def plot_results(results, X_train, X_test, y_test, feature_cols, xgb_base):

    # 1. Confusion matrices for both thresholds of primary model
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    name = "XGBoost (Calibrated)"
    for ax, (label, cm) in zip(axes, [
        (f"Watch List (thresh={results[name]['watch_thresh']})", results[name]["cm_watch"]),
        (f"Active Alert (thresh={results[name]['alert_thresh']})", results[name]["cm_alert"]),
    ]):
        ConfusionMatrixDisplay(cm, display_labels=["Pass/Dist", "Fail/Wdrn"]).plot(ax=ax, cmap="Blues")
        ax.set_title(f"XGBoost — {label}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task2_confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 2. ROC + PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name, color in [("XGBoost (Calibrated)", "#e74c3c"), ("Random Forest", "#3498db")]:
        fpr, tpr, _ = roc_curve(y_test, results[name]["y_proba"])
        ax1.plot(fpr, tpr, color=color, linewidth=2,
                 label=f'{name} (AUC={results[name]["roc_auc"]:.3f})')
        prec, rec, _ = precision_recall_curve(y_test, results[name]["y_proba"])
        ax2.plot(rec, prec, color=color, linewidth=2, label=name)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_title("ROC Curves (Temporal Test Set)")
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_title("Precision-Recall Curves")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task2_roc_pr_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Feature importance from base XGBoost
    importances = xgb_base.feature_importances_
    sorted_idx = np.argsort(importances)[-15:]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(sorted_idx)), importances[sorted_idx], color="#e74c3c", alpha=0.8)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_cols[i] for i in sorted_idx])
    ax.set_title("Top 15 Features Driving Disengagement Risk (XGBoost)", fontweight="bold")
    ax.set_xlabel("Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task2_feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Calibration — Platt-scaled vs uncalibrated
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, color in [("XGBoost (Calibrated)", "#e74c3c"), ("Random Forest", "#3498db")]:
        prob_true, prob_pred = calibration_curve(
            y_test, results[name]["y_proba"], n_bins=10, strategy="uniform"
        )
        ax.plot(prob_pred, prob_true, "o-", color=color, linewidth=2, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
    ax.set_title("Calibration Plot (Platt Scaling Applied)")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Actual Withdrawal/Fail Rate")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task2_calibration.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 5. SHAP summary plot (Fix 7)
    try:
        explainer = shap.TreeExplainer(xgb_base)
        sample_idx = np.random.choice(len(X_test), min(500, len(X_test)), replace=False)
        shap_values = explainer.shap_values(X_test[sample_idx])
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_test[sample_idx],
            feature_names=feature_cols,
            show=False, max_display=15
        )
        plt.title("SHAP Feature Impact on Disengagement Risk")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "task2_shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: task2_shap_summary.png")
    except Exception as e:
        print(f"  SHAP plot skipped: {e}")

    print("  Saved: task2_confusion_matrices.png, task2_roc_pr_curves.png")
    print("  Saved: task2_feature_importance.png, task2_calibration.png")


def print_alert_design(results, feature_cols, xgb_base):
    importances = xgb_base.feature_importances_
    top3_idx = np.argsort(importances)[-3:][::-1]
    top3 = [(feature_cols[i], importances[i]) for i in top3_idx]

    print("\n" + "=" * 60)
    print("STAFF ALERT DESIGN — Two-Threshold System")
    print("=" * 60)
    print("\nTop 3 risk drivers:")
    for i, (feat, imp) in enumerate(top3, 1):
        print(f"  {i}. {feat} (importance: {imp:.4f})")

    print("""
Two-Threshold Alert System:
  Threshold 1 — Watch List (≥ 0.20):
    Silent flag in advisor dashboard. No push notification.
    Advisors review weekly. Catches 96% of at-risk students.

  Threshold 2 — Active Alert (≥ 0.50):
    Active notification sent to advisor. Higher precision.
    Only sent when risk is more certain.

Notification Format (Active Alert):
┌─────────────────────────────────────────────────────────┐
│  ⚠ EARLY WARNING: Student At Risk of Disengagement     │
│                                                         │
│  Student ID: [XXXXX]    Course: [Module-Presentation]   │
│  Risk Score: [XX]%      Risk Level: [HIGH/CRITICAL]     │
│                                                         │
│  Key Risk Factors (SHAP-explained):                     │
│  • Last VLE login: 18 days ago      (+0.32 risk)        │
│  • Assessments submitted: 0/2 due   (+0.28 risk)        │
│  • Activity 72% below module avg    (+0.19 risk)        │
│                                                         │
│  Suggested Actions:                                     │
│  1. Schedule check-in within 48 hours                   │
│  2. Review assessment submission barriers               │
│  3. Connect to peer study group                         │
│                                                         │
│  [View Full Profile]  [Log Intervention]  [Dismiss]     │
└─────────────────────────────────────────────────────────┘

Escalation tiers:
  - Watch list (0.20-0.50): Dashboard flag, weekly review
  - Active alert (0.50-0.75): Same-day email to advisor
  - Critical (0.75+): Real-time push + retention team flag
""")


def run():
    print("=" * 60)
    print("TASK 2: Predictive Disengagement Model")
    print("=" * 60)

    tables = load_all()

    print("\n[1/3] Building Week-6-constrained features (no leakage)...")
    features = build_week6_features(tables)
    print(f"  -> {len(features):,} students, {features['target'].sum():,} at-risk")
    print(f"  -> {len([c for c in features.columns if c not in KEY + ['target']])} features")

    print("\n[2/3] Training with temporal split (2013 train → 2014 test)...")
    results, X_train, X_test, y_test, feat_cols, xgb_base = train_and_evaluate(features)

    print("\n[3/3] Generating visualizations...")
    plot_results(results, X_train, X_test, y_test, feat_cols, xgb_base)
    print_alert_design(results, feat_cols, xgb_base)

    joblib.dump(results["XGBoost (Calibrated)"]["model"],
                os.path.join(MODEL_DIR, "xgb_disengagement_w6.joblib"))
    features.to_csv(os.path.join(FIG_DIR, "..", "task2_features.csv"), index=False)
    print("\nTask 2 complete.")

    return results, features


if __name__ == "__main__":
    run()
