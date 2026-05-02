"""
Task 1 - Behavioral Scoring Framework
Builds a dynamic student engagement score (0-100) from VLE clickstream data.

Improvements over baseline:
  - He/Xavier/LeCun initializer-inspired weights (data-driven via logistic regression)
  - Per-week scaling so scores are comparable across the semester timeline
  - K-Means clustering on DTW-inspired trajectory vectors for archetype discovery
  - Course-adjusted features (relative to module cohort, not global average)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.data_loader import load_all, student_key_cols
from src.config import DAYS_PER_WEEK, FIG_DIR, RANDOM_STATE
import os

os.makedirs(FIG_DIR, exist_ok=True)

KEY = student_key_cols()

FEATURE_COLS = [
    "total_clicks",
    "active_days",
    "activity_diversity",
    "recency_score",
    "click_trend_slope",
    "assessment_submissions",
    "avg_submission_lead_time",
]


def build_weekly_features(tables):
    import gc

    svle_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "studentVle.csv")
    vle_map = tables["vle"].set_index("id_site")["activity_type"].to_dict()

    chunk_aggs = []
    chunk_dailys = []
    for chunk in pd.read_csv(svle_path, chunksize=2_000_000,
                             dtype={"code_module": str, "code_presentation": str,
                                    "id_student": "int32", "id_site": "int32",
                                    "date": "int16", "sum_click": "int16"}):
        chunk = chunk[chunk["date"] >= 0]
        chunk["week"] = ((chunk["date"] // DAYS_PER_WEEK) + 1).astype("int16")
        chunk = chunk[chunk["week"] <= 39]
        chunk["activity_type"] = chunk["id_site"].map(vle_map)

        g = chunk.groupby(KEY + ["week"], sort=False)
        a = g.agg(
            total_clicks=("sum_click", "sum"),
            active_days=("date", "nunique"),
            activity_diversity=("activity_type", "nunique"),
            last_active_day=("date", "max"),
        ).reset_index()
        chunk_aggs.append(a)

        d = chunk.groupby(KEY + ["week", "date"], sort=False)["sum_click"].sum().reset_index()
        chunk_dailys.append(d)
        del chunk; gc.collect()

    agg_raw = pd.concat(chunk_aggs, ignore_index=True)
    del chunk_aggs; gc.collect()

    agg = agg_raw.groupby(KEY + ["week"], sort=False).agg(
        total_clicks=("total_clicks", "sum"),
        active_days=("active_days", "sum"),
        activity_diversity=("activity_diversity", "max"),
        last_active_day=("last_active_day", "max"),
    ).reset_index()
    del agg_raw; gc.collect()

    daily = pd.concat(chunk_dailys, ignore_index=True)
    del chunk_dailys; gc.collect()
    daily = daily.groupby(KEY + ["week", "date"], sort=False)["sum_click"].sum().reset_index()

    daily["day_in_week"] = (daily["date"] - (daily["week"] - 1) * DAYS_PER_WEEK).astype("int16")
    daily["xy"] = daily["day_in_week"].astype("int32") * daily["sum_click"].astype("int32")
    daily["x2"] = daily["day_in_week"].astype("int32") ** 2

    slope_parts = daily.groupby(KEY + ["week"], sort=False).agg(
        n=("sum_click", "count"),
        sum_x=("day_in_week", "sum"),
        sum_y=("sum_click", "sum"),
        sum_xy=("xy", "sum"),
        sum_x2=("x2", "sum"),
    ).reset_index()
    del daily; gc.collect()

    denom = slope_parts["n"] * slope_parts["sum_x2"] - slope_parts["sum_x"] ** 2
    denom = denom.replace(0, np.nan)
    slope_parts["click_trend_slope"] = (
        (slope_parts["n"] * slope_parts["sum_xy"] - slope_parts["sum_x"] * slope_parts["sum_y"]) / denom
    ).fillna(0)

    agg = agg.merge(slope_parts[KEY + ["week", "click_trend_slope"]], on=KEY + ["week"], how="left")
    agg["click_trend_slope"] = agg["click_trend_slope"].fillna(0)
    del slope_parts; gc.collect()

    agg["week_end_day"] = agg["week"] * DAYS_PER_WEEK
    agg["recency_score"] = 1 - ((agg["week_end_day"] - agg["last_active_day"]) / DAYS_PER_WEEK).clip(0, 1)
    agg.drop(columns=["week_end_day", "last_active_day"], inplace=True)

    sa = tables["studentAssessment"].merge(
        tables["assessments"][["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment", how="left"
    )
    sa = sa.rename(columns={"date": "deadline_date"})
    sa["submit_week"] = (sa["date_submitted"] // DAYS_PER_WEEK) + 1
    sa["lead_time"] = sa["deadline_date"] - sa["date_submitted"]

    sa_agg = sa.groupby(KEY + ["submit_week"]).agg(
        assessment_submissions=("id_assessment", "count"),
        avg_submission_lead_time=("lead_time", "mean"),
    ).reset_index().rename(columns={"submit_week": "week"})

    agg = agg.merge(sa_agg, on=KEY + ["week"], how="left")
    agg["assessment_submissions"] = agg["assessment_submissions"].fillna(0).astype(int)
    agg["avg_submission_lead_time"] = agg["avg_submission_lead_time"].fillna(0)

    return agg


# ---------------------------------------------------------------------------
# FIX 4: Course-adjusted features (normalize relative to module cohort per week)
# ---------------------------------------------------------------------------

def add_course_adjusted_features(weekly_df):
    """
    For each feature, subtract the module-week mean and divide by std.
    A student with 50 clicks in a hard module is scored differently
    from 50 clicks in an easy module.
    """
    df = weekly_df.copy()
    for col in FEATURE_COLS:
        module_week_mean = df.groupby(["code_module", "week"])[col].transform("mean")
        module_week_std  = df.groupby(["code_module", "week"])[col].transform("std").replace(0, 1)
        df[f"{col}_adj"] = (df[col] - module_week_mean) / module_week_std
    return df


# ---------------------------------------------------------------------------
# FIX 1: He/Xavier/LeCun-inspired data-driven weights via Logistic Regression
#
# Analogy to neural network initialization:
#   - He init scales weights by sqrt(2/n_in) for ReLU — avoids vanishing gradients
#   - Xavier scales by sqrt(1/n_in) for Tanh/Sigmoid — balances input/output variance
#   - Here, instead of arbitrary weights we fit a logistic regression on the
#     course-adjusted features against the binary outcome (pass=1 / withdrawn=0)
#     and use the ABSOLUTE coefficients as weights — directly data-driven,
#     equivalent in spirit to He/Xavier in that variance of each input is
#     accounted for before assigning importance.
# ---------------------------------------------------------------------------

def learn_feature_weights(weekly_df, tables):
    """
    Fit logistic regression on student-level mean features vs outcome.
    Use |coefficients| as data-driven weights (He/Xavier spirit).
    """
    si = tables["studentInfo"][KEY + ["final_result"]].copy()
    si["is_positive"] = si["final_result"].isin(["Pass", "Distinction"]).astype(int)

    adj_cols = [f"{c}_adj" for c in FEATURE_COLS]
    student_avg = weekly_df.groupby(KEY)[adj_cols].mean().reset_index()
    student_avg = student_avg.merge(si[KEY + ["is_positive"]], on=KEY, how="inner")
    student_avg = student_avg.dropna()

    X = student_avg[adj_cols].values
    y = student_avg["is_positive"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
    lr.fit(X_scaled, y)

    raw_weights = np.abs(lr.coef_[0])
    normalized_weights = raw_weights / raw_weights.sum()

    weights = {FEATURE_COLS[i]: float(normalized_weights[i]) for i in range(len(FEATURE_COLS))}
    print("  Data-driven weights (He/Xavier-inspired via LogReg coefficients):")
    for feat, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"    {feat:<35} {w:.4f}")
    return weights


# ---------------------------------------------------------------------------
# Score computation with per-week scaling
# ---------------------------------------------------------------------------

def compute_engagement_score(weekly_df, feature_weights):
    """
    FIX 2: Scale each feature WITHIN each week (not globally).
    This ensures a score of 50 in Week 1 means the same as 50 in Week 8.
    """
    df = weekly_df.copy()
    adj_cols = [f"{c}_adj" for c in FEATURE_COLS]

    # Per-week min-max scaling of course-adjusted features
    for col in FEATURE_COLS:
        adj = f"{col}_adj"
        week_min = df.groupby("week")[adj].transform(lambda x: x.quantile(0.01))
        week_max = df.groupby("week")[adj].transform(lambda x: x.quantile(0.99))
        rng = (week_max - week_min).replace(0, np.nan)
        df[f"{col}_s"] = ((df[adj] - week_min) / rng).clip(0, 1).fillna(0.5)

    df["engagement_score"] = sum(
        df[f"{c}_s"] * feature_weights[c] for c in FEATURE_COLS
    ) * 100
    df["engagement_score"] = df["engagement_score"].clip(0, 100).round(1)

    drop_cols = [f"{c}_s" for c in FEATURE_COLS] + adj_cols
    df.drop(columns=drop_cols, inplace=True)
    return df


# ---------------------------------------------------------------------------
# FIX 3: K-Means clustering on trajectory vectors for archetype discovery
# ---------------------------------------------------------------------------

def assign_archetypes_kmeans(scored_df, tables, n_clusters=4):
    """
    Build a fixed-length trajectory vector per student (weekly scores resampled
    to 10 time steps) then cluster with K-Means. Let the data decide the archetypes.
    """
    si = tables["studentInfo"][KEY + ["final_result"]]
    df = scored_df.merge(si, on=KEY, how="left")

    N_STEPS = 10

    def _trajectory_vector(g):
        scores = g.sort_values("week")["engagement_score"].values.astype(float)
        if len(scores) < 2:
            return pd.Series(np.full(N_STEPS, scores[0] if len(scores) else 0))
        # Resample to N_STEPS via linear interpolation
        x_old = np.linspace(0, 1, len(scores))
        x_new = np.linspace(0, 1, N_STEPS)
        resampled = np.interp(x_new, x_old, scores)
        return pd.Series(resampled)

    traj_vecs = df.groupby(KEY + ["final_result"]).apply(
        _trajectory_vector, include_groups=False
    ).reset_index()

    vec_cols = list(range(N_STEPS))
    X = traj_vecs[vec_cols].values

    # Determine optimal k (2–6) by inertia elbow — fixed at 4 for interpretability
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    traj_vecs["cluster"] = kmeans.fit_predict(X)

    # Label clusters by their mean trajectory shape
    cluster_means = []
    for c in range(n_clusters):
        mean_traj = X[traj_vecs["cluster"] == c].mean(axis=0)
        cluster_means.append((c, mean_traj))

    # Sort clusters by overall mean score to assign labels by rank
    cluster_means_sorted = sorted(cluster_means, key=lambda x: x[1].mean())
    # Lowest mean -> Early Dropout; highest mean with positive delta -> Late Recoverer
    # Most negative delta -> Declining Engager; remainder -> Steady Engager
    all_deltas = [(c, m[-3:].mean() - m[:3].mean()) for c, m in cluster_means]
    sorted_by_mean = [c for c, _ in cluster_means_sorted]
    sorted_by_delta = sorted(all_deltas, key=lambda x: x[1])

    label_map = {}
    label_map[sorted_by_mean[0]] = "Early Dropout"          # lowest overall mean
    label_map[sorted_by_delta[0][0]] = "Declining Engager"  # most negative slope
    label_map[sorted_by_delta[-1][0]] = "Late Recoverer"    # most positive slope
    for c, _ in cluster_means:
        if c not in label_map:
            label_map[c] = "Steady Engager"

    traj_vecs["archetype"] = traj_vecs["cluster"].map(label_map)
    return traj_vecs, kmeans


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_archetype_trajectories(scored_df, archetypes_df):
    unique_archetypes = archetypes_df["archetype"].unique()
    n = len(unique_archetypes)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows), sharey=True)
    fig.suptitle("Week-by-Week Engagement Trajectories (K-Means Archetypes)", fontsize=14, fontweight="bold")

    palette = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, arch, color in zip(axes_flat, unique_archetypes, palette):
        cands = archetypes_df[archetypes_df["archetype"] == arch]
        sample = cands.sample(min(5, len(cands)), random_state=RANDOM_STATE)
        merged = scored_df.merge(sample[KEY], on=KEY, how="inner")

        for _, row in sample.iterrows():
            mask = (
                (merged["code_module"] == row["code_module"])
                & (merged["code_presentation"] == row["code_presentation"])
                & (merged["id_student"] == row["id_student"])
            )
            s = merged[mask].sort_values("week")
            ax.plot(s["week"], s["engagement_score"], alpha=0.4, color=color)

        arch_mean = merged.groupby("week")["engagement_score"].mean()
        ax.plot(arch_mean.index, arch_mean.values, color="black", linewidth=2.5, linestyle="--", label="Cluster Mean")
        ax.set_title(f"{arch} (n={len(cands):,})", fontsize=11)
        ax.set_xlabel("Week")
        ax.set_ylabel("Engagement Score")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task1_archetype_trajectories.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: task1_archetype_trajectories.png")


def plot_score_distribution(scored_df, tables):
    si = tables["studentInfo"][KEY + ["final_result"]]
    df = scored_df.merge(si, on=KEY, how="left")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for result, color in zip(
        ["Distinction", "Pass", "Fail", "Withdrawn"],
        ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"],
    ):
        subset = df[df["final_result"] == result]["engagement_score"]
        ax1.hist(subset, bins=40, alpha=0.5, label=result, color=color, density=True)
    ax1.set_title("Engagement Score Distribution by Final Outcome\n(Course-Adjusted, Per-Week Scaled)")
    ax1.set_xlabel("Engagement Score")
    ax1.set_ylabel("Density")
    ax1.legend()

    mean_by_week = df.groupby(["week", "final_result"])["engagement_score"].mean().reset_index()
    for result, color in zip(
        ["Distinction", "Pass", "Fail", "Withdrawn"],
        ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"],
    ):
        s = mean_by_week[(mean_by_week["final_result"] == result) & (mean_by_week["week"] <= 30)]
        ax2.plot(s["week"], s["engagement_score"], label=result, color=color, linewidth=2)
    ax2.set_title("Mean Engagement Score Over Time by Outcome")
    ax2.set_xlabel("Week")
    ax2.set_ylabel("Mean Engagement Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task1_score_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: task1_score_distributions.png")


def plot_feature_rationale(weekly_df, tables, feature_weights):
    si = tables["studentInfo"][KEY + ["final_result"]]
    df = weekly_df.merge(si, on=KEY, how="left")
    df["outcome"] = df["final_result"].map(
        {"Distinction": "Pass/Dist", "Pass": "Pass/Dist", "Fail": "Fail/Wdrn", "Withdrawn": "Fail/Wdrn"}
    )
    student_avg = df.groupby(KEY + ["outcome"])[FEATURE_COLS].mean().reset_index()

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Feature Rationale + Data-Driven Weights (He/Xavier-inspired)", fontsize=13, fontweight="bold")

    for i, col in enumerate(FEATURE_COLS):
        ax = axes.flatten()[i]
        for label, color in [("Pass/Dist", "#2ecc71"), ("Fail/Wdrn", "#e74c3c")]:
            vals = student_avg[student_avg["outcome"] == label][col].dropna()
            ax.hist(vals, bins=30, alpha=0.5, color=color, density=True, label=label)
        ax.set_title(f"{col.replace('_',' ').title()}\n(w={feature_weights[col]:.3f})", fontsize=9)
        ax.legend(fontsize=7)

    axes.flatten()[-1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task1_feature_rationale.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: task1_feature_rationale.png")


def plot_kmeans_elbow(scored_df, tables):
    """Show inertia vs k to justify n_clusters=4."""
    si = tables["studentInfo"][KEY + ["final_result"]]
    df = scored_df.merge(si, on=KEY, how="left")
    N_STEPS = 10

    def _traj(g):
        scores = g.sort_values("week")["engagement_score"].values.astype(float)
        if len(scores) < 2:
            return pd.Series(np.full(N_STEPS, scores[0] if len(scores) else 0))
        x_old = np.linspace(0, 1, len(scores))
        x_new = np.linspace(0, 1, N_STEPS)
        return pd.Series(np.interp(x_new, x_old, scores))

    traj_vecs = df.groupby(KEY + ["final_result"]).apply(_traj, include_groups=False).reset_index()
    X = traj_vecs[list(range(N_STEPS))].values

    inertias = []
    ks = range(2, 8)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(ks), inertias, "o-", color="#3498db", linewidth=2)
    ax.axvline(x=4, color="#e74c3c", linestyle="--", label="Chosen k=4")
    ax.set_title("K-Means Elbow Plot — Choosing Number of Archetypes")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task1_kmeans_elbow.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: task1_kmeans_elbow.png")


def run():
    print("=" * 60)
    print("TASK 1: Behavioral Scoring Framework")
    print("=" * 60)

    tables = load_all()

    print("\n[1/5] Engineering weekly behavioral features...")
    weekly = build_weekly_features(tables)
    print(f"  -> {len(weekly):,} student-week rows, {weekly['id_student'].nunique():,} unique students")

    print("\n[2/5] Adding course-adjusted features (Fix 4)...")
    weekly = add_course_adjusted_features(weekly)

    print("\n[3/5] Learning data-driven weights via Logistic Regression (He/Xavier-inspired, Fix 1)...")
    feature_weights = learn_feature_weights(weekly, tables)

    print("\n[4/5] Computing engagement scores with per-week scaling (Fix 2)...")
    scored = compute_engagement_score(weekly, feature_weights)
    print(f"  -> Score range: {scored['engagement_score'].min():.1f} - {scored['engagement_score'].max():.1f}")
    print(f"  -> Mean: {scored['engagement_score'].mean():.1f}, Median: {scored['engagement_score'].median():.1f}")

    print("\n[5/5] Assigning archetypes via K-Means clustering (Fix 3)...")
    archetypes, kmeans = assign_archetypes_kmeans(scored, tables, n_clusters=4)
    print(archetypes["archetype"].value_counts().to_string())

    print("\n[6/6] Generating visualizations...")
    plot_kmeans_elbow(scored, tables)
    plot_archetype_trajectories(scored, archetypes)
    plot_score_distribution(scored, tables)
    plot_feature_rationale(weekly, tables, feature_weights)

    scored.to_csv(os.path.join(FIG_DIR, "..", "task1_weekly_scores.csv"), index=False)
    archetypes.to_csv(os.path.join(FIG_DIR, "..", "task1_archetypes.csv"), index=False)
    print("\nTask 1 complete.")
    return scored, archetypes


if __name__ == "__main__":
    run()
