"""
Task 1 - Behavioral Scoring Framework
Builds a dynamic student engagement score (0-100) from VLE clickstream data.
Produces week-by-week trajectories and identifies student archetypes.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from src.data_loader import load_all, student_key_cols
from src.config import DAYS_PER_WEEK, FIG_DIR
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

FEATURE_WEIGHTS = {
    "total_clicks": 0.20,
    "active_days": 0.15,
    "activity_diversity": 0.15,
    "recency_score": 0.15,
    "click_trend_slope": 0.10,
    "assessment_submissions": 0.15,
    "avg_submission_lead_time": 0.10,
}


def build_weekly_features(tables):
    import gc

    # Process VLE data in chunks to reduce peak memory
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

        # Daily aggregation for slope
        d = chunk.groupby(KEY + ["week", "date"], sort=False)["sum_click"].sum().reset_index()
        chunk_dailys.append(d)
        del chunk; gc.collect()

    # Combine chunk results - re-aggregate
    agg_raw = pd.concat(chunk_aggs, ignore_index=True)
    del chunk_aggs; gc.collect()

    agg = agg_raw.groupby(KEY + ["week"], sort=False).agg(
        total_clicks=("total_clicks", "sum"),
        active_days=("active_days", "sum"),
        activity_diversity=("activity_diversity", "max"),
        last_active_day=("last_active_day", "max"),
    ).reset_index()
    del agg_raw; gc.collect()

    # Slope from daily data
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

    # Recency score
    agg["week_end_day"] = agg["week"] * DAYS_PER_WEEK
    agg["recency_score"] = 1 - ((agg["week_end_day"] - agg["last_active_day"]) / DAYS_PER_WEEK).clip(0, 1)
    agg.drop(columns=["week_end_day", "last_active_day"], inplace=True)

    # Assessment features
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


def compute_engagement_score(weekly_df):
    df = weekly_df.copy()
    for col in FEATURE_COLS:
        mn, mx = df[col].quantile(0.01), df[col].quantile(0.99)
        if mx - mn < 1e-9:
            df[f"{col}_s"] = 0.5
        else:
            df[f"{col}_s"] = ((df[col] - mn) / (mx - mn)).clip(0, 1)

    df["engagement_score"] = sum(
        df[f"{c}_s"] * FEATURE_WEIGHTS[c] for c in FEATURE_COLS
    ) * 100
    df["engagement_score"] = df["engagement_score"].clip(0, 100).round(1)

    drop_cols = [f"{c}_s" for c in FEATURE_COLS]
    df.drop(columns=drop_cols, inplace=True)
    return df


def assign_archetypes(scored_df, tables):
    si = tables["studentInfo"][KEY + ["final_result"]]
    df = scored_df.merge(si, on=KEY, how="left")

    # Compute trajectory stats per student
    def _traj_stats(g):
        scores = g.sort_values("week")["engagement_score"].values
        n = len(scores)
        if n < 2:
            return pd.Series({"mean_score": scores.mean(), "delta": 0, "num_weeks": n})
        half = n // 2
        return pd.Series({
            "mean_score": scores.mean(),
            "delta": scores[half:].mean() - scores[:half].mean(),
            "num_weeks": n,
        })

    traj = df.groupby(KEY + ["final_result"]).apply(_traj_stats, include_groups=False).reset_index()

    conditions = [
        (traj["final_result"] == "Withdrawn") & (traj["num_weeks"] <= 10),
        traj["delta"] > 10,
        traj["delta"] < -10,
    ]
    choices = ["Early Dropout", "Late Recoverer", "Declining Engager"]
    traj["archetype"] = np.select(conditions, choices, default="Steady Engager")
    return traj


def plot_archetype_trajectories(scored_df, archetypes_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.suptitle("Week-by-Week Engagement Trajectories by Student Archetype", fontsize=14, fontweight="bold")

    names = ["Steady Engager", "Early Dropout", "Late Recoverer", "Declining Engager"]
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]

    for ax, arch, color in zip(axes.flatten(), names, colors):
        cands = archetypes_df[archetypes_df["archetype"] == arch]
        if len(cands) == 0:
            ax.set_title(f"{arch} (no examples)")
            continue

        sample = cands.sample(min(5, len(cands)), random_state=42)
        merged = scored_df.merge(sample[KEY], on=KEY, how="inner")

        for _, row in sample.iterrows():
            mask = (
                (merged["code_module"] == row["code_module"])
                & (merged["code_presentation"] == row["code_presentation"])
                & (merged["id_student"] == row["id_student"])
            )
            s = merged[mask].sort_values("week")
            ax.plot(s["week"], s["engagement_score"], alpha=0.5, color=color)

        arch_mean = merged.groupby("week")["engagement_score"].mean()
        ax.plot(arch_mean.index, arch_mean.values, color="black", linewidth=2.5, linestyle="--", label="Archetype Mean")

        ax.set_title(f"{arch} (n={len(cands):,})", fontsize=12)
        ax.set_xlabel("Week")
        ax.set_ylabel("Engagement Score")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

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
    ax1.set_title("Engagement Score Distribution by Final Outcome")
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


def plot_feature_rationale(weekly_df, tables):
    si = tables["studentInfo"][KEY + ["final_result"]]
    df = weekly_df.merge(si, on=KEY, how="left")
    df["outcome"] = df["final_result"].map(
        {"Distinction": "Pass/Dist", "Pass": "Pass/Dist", "Fail": "Fail/Wdrn", "Withdrawn": "Fail/Wdrn"}
    )

    student_avg = df.groupby(KEY + ["outcome"])[FEATURE_COLS].mean().reset_index()

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Feature Rationale: Distributions by Outcome Group", fontsize=13, fontweight="bold")

    for i, col in enumerate(FEATURE_COLS):
        ax = axes.flatten()[i]
        for label, color in [("Pass/Dist", "#2ecc71"), ("Fail/Wdrn", "#e74c3c")]:
            vals = student_avg[student_avg["outcome"] == label][col].dropna()
            ax.hist(vals, bins=30, alpha=0.5, color=color, density=True, label=label)
        ax.set_title(col.replace("_", " ").title(), fontsize=10)
        ax.legend(fontsize=7)

    axes.flatten()[-1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task1_feature_rationale.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: task1_feature_rationale.png")


def run():
    print("=" * 60)
    print("TASK 1: Behavioral Scoring Framework")
    print("=" * 60)

    tables = load_all()

    print("\n[1/4] Engineering weekly behavioral features...")
    weekly = build_weekly_features(tables)
    print(f"  -> {len(weekly):,} student-week rows, {weekly['id_student'].nunique():,} unique students")

    print("\n[2/4] Computing composite engagement scores...")
    scored = compute_engagement_score(weekly)
    print(f"  -> Score range: {scored['engagement_score'].min():.1f} - {scored['engagement_score'].max():.1f}")
    print(f"  -> Mean: {scored['engagement_score'].mean():.1f}, Median: {scored['engagement_score'].median():.1f}")

    print("\n[3/4] Assigning student archetypes...")
    archetypes = assign_archetypes(scored, tables)
    print(archetypes["archetype"].value_counts().to_string())

    print("\n[4/4] Generating visualizations...")
    plot_archetype_trajectories(scored, archetypes)
    plot_score_distribution(scored, tables)
    plot_feature_rationale(weekly, tables)

    scored.to_csv(os.path.join(FIG_DIR, "..", "task1_weekly_scores.csv"), index=False)
    archetypes.to_csv(os.path.join(FIG_DIR, "..", "task1_archetypes.csv"), index=False)
    print("\nTask 1 complete.")
    return scored, archetypes


if __name__ == "__main__":
    run()
