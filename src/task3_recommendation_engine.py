"""
Task 3 - Course Recommendation Engine
Recommends top-3 courses for students using:
  A) Content-based filtering (student profile + course metadata)
  B) Collaborative filtering (interaction patterns of similar students)
Includes cold-start handling and evaluation.

Primary user: Students seeking next-semester course opportunities.
Rationale: Students benefit most from personalized guidance; staff can then
review recommendations in the advisor dashboard built in Task 2.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

from src.data_loader import load_all, student_key_cols, course_key_cols
from src.config import FIG_DIR, RANDOM_STATE
import os

os.makedirs(FIG_DIR, exist_ok=True)

KEY = student_key_cols()
CKEY = course_key_cols()


def build_student_profiles(tables):
    """Build a feature profile per student-course enrollment."""
    si = tables["studentInfo"].copy()

    edu_map = {"No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2,
               "HE Qualification": 3, "Post Graduate Qualification": 4}
    age_map = {"0-35": 0, "35-55": 1, "55<=": 2}
    result_map = {"Distinction": 3, "Pass": 2, "Fail": 1, "Withdrawn": 0}

    si["edu_level"] = si["highest_education"].map(edu_map).fillna(1)
    si["age_num"] = si["age_band"].map(age_map).fillna(0)
    si["result_num"] = si["final_result"].map(result_map)
    si["gender_num"] = (si["gender"] == "M").astype(int)
    si["disability_num"] = (si["disability"] == "Y").astype(int)

    # Aggregate VLE behavior per student-course
    svle_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "studentVle.csv")
    chunks = []
    for chunk in pd.read_csv(svle_path, chunksize=2_000_000,
                             dtype={"code_module": str, "code_presentation": str,
                                    "id_student": "int32", "id_site": "int32",
                                    "date": "int16", "sum_click": "int16"}):
        agg = chunk.groupby(KEY).agg(total_clicks=("sum_click", "sum")).reset_index()
        chunks.append(agg)

    vle_agg = pd.concat(chunks).groupby(KEY)["total_clicks"].sum().reset_index()

    # Assessment performance
    sa = tables["studentAssessment"].merge(
        tables["assessments"][["id_assessment", "code_module", "code_presentation"]],
        on="id_assessment", how="left"
    )
    asmt_agg = sa.groupby(KEY).agg(
        mean_score=("score", "mean"),
        num_assessments=("id_assessment", "count"),
    ).reset_index()

    profiles = si[KEY + ["edu_level", "age_num", "gender_num", "disability_num",
                         "num_of_prev_attempts", "studied_credits", "result_num"]].copy()
    profiles = profiles.merge(vle_agg, on=KEY, how="left")
    profiles = profiles.merge(asmt_agg, on=KEY, how="left")
    profiles["total_clicks"] = profiles["total_clicks"].fillna(0)
    profiles["mean_score"] = profiles["mean_score"].fillna(50)
    profiles["num_assessments"] = profiles["num_assessments"].fillna(0)

    return profiles


def build_course_profiles(tables):
    """Build feature vectors for each course (module)."""
    si = tables["studentInfo"]
    courses = tables["courses"].copy()

    # Aggregate student outcomes per course
    course_stats = si.groupby("code_module").agg(
        avg_pass_rate=("final_result", lambda x: (x.isin(["Pass", "Distinction"])).mean()),
        avg_distinction_rate=("final_result", lambda x: (x == "Distinction").mean()),
        withdrawal_rate=("final_result", lambda x: (x == "Withdrawn").mean()),
        total_students=("id_student", "count"),
        avg_prev_attempts=("num_of_prev_attempts", "mean"),
        avg_credits=("studied_credits", "mean"),
    ).reset_index()

    # Module length
    mod_len = courses.groupby("code_module")["module_presentation_length"].mean().reset_index()
    mod_len.rename(columns={"module_presentation_length": "avg_length"}, inplace=True)

    course_profiles = course_stats.merge(mod_len, on="code_module", how="left")
    return course_profiles


# ---------------------------------------------------------------------------
# Approach A: Content-Based Filtering
# ---------------------------------------------------------------------------

class ContentBasedRecommender:
    """Recommends courses based on similarity between student profile
    and course characteristics. Students who succeeded in similar courses
    will be recommended courses with matching profiles."""

    def __init__(self, student_profiles, course_profiles, tables):
        self.tables = tables
        self.student_profiles = student_profiles
        self.course_profiles = course_profiles

        # Build student-course success matrix
        self.si = tables["studentInfo"][KEY + ["final_result"]].copy()
        self.si["success"] = self.si["final_result"].isin(["Pass", "Distinction"]).astype(int)

        # Course feature vectors (standardized)
        feat_cols = ["avg_pass_rate", "avg_distinction_rate", "withdrawal_rate",
                     "total_students", "avg_prev_attempts", "avg_credits", "avg_length"]
        self.scaler = StandardScaler()
        self.course_features = self.scaler.fit_transform(
            course_profiles[feat_cols].fillna(0).values
        )
        self.course_modules = course_profiles["code_module"].values

    def recommend(self, student_id, n=3, exclude_modules=None, known_modules=None):
        """Recommend top-n courses for a student based on their success profile.
        known_modules: override which modules to treat as the student's history (for eval)."""
        if exclude_modules is None:
            exclude_modules = set()

        student_hist = self.si[self.si["id_student"] == student_id]

        if known_modules is not None:
            successful_modules = known_modules
        else:
            successful_modules = set(student_hist[student_hist["success"] == 1]["code_module"].unique())

        if len(successful_modules) == 0:
            return self._cold_start_recommend(student_id, n, exclude_modules)

        successful_idx = [i for i, m in enumerate(self.course_modules) if m in successful_modules]
        if len(successful_idx) == 0:
            return self._cold_start_recommend(student_id, n, exclude_modules)

        student_vec = self.course_features[successful_idx].mean(axis=0).reshape(1, -1)
        sims = cosine_similarity(student_vec, self.course_features).flatten()

        candidates = [
            (self.course_modules[i], sims[i])
            for i in range(len(self.course_modules))
            if self.course_modules[i] not in exclude_modules
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n]

    def _cold_start_recommend(self, student_id, n, exclude_modules):
        """Cold-start strategy: recommend courses with highest overall pass rate
        and lowest withdrawal rate — safest choices for new students."""
        cp = self.course_profiles.copy()
        cp["cold_score"] = cp["avg_pass_rate"] * 0.6 - cp["withdrawal_rate"] * 0.4
        cp = cp[~cp["code_module"].isin(exclude_modules)]
        cp = cp.sort_values("cold_score", ascending=False)
        return [(row["code_module"], row["cold_score"]) for _, row in cp.head(n).iterrows()]


# ---------------------------------------------------------------------------
# Approach B: Collaborative Filtering (User-Based)
# ---------------------------------------------------------------------------

class CollaborativeRecommender:
    """User-based collaborative filtering: find similar students based on
    their course engagement patterns, recommend what similar students
    succeeded in."""

    def __init__(self, student_profiles, tables):
        self.tables = tables
        si = tables["studentInfo"].copy()

        # Build student-module success matrix
        si["success"] = si["final_result"].isin(["Pass", "Distinction"]).astype(float)
        self.success_matrix = si.pivot_table(
            index="id_student", columns="code_module", values="success", aggfunc="max"
        ).fillna(0)

        # Build student feature matrix for similarity
        feat_cols = ["edu_level", "age_num", "total_clicks", "mean_score",
                     "num_of_prev_attempts", "studied_credits"]
        student_avg = student_profiles.groupby("id_student")[feat_cols].mean().reset_index()
        student_avg = student_avg.set_index("id_student")

        self.scaler = StandardScaler()
        valid_idx = student_avg.index.intersection(self.success_matrix.index)
        self.student_features = pd.DataFrame(
            self.scaler.fit_transform(student_avg.loc[valid_idx].fillna(0)),
            index=valid_idx,
            columns=feat_cols,
        )
        self.success_matrix = self.success_matrix.loc[valid_idx]

    def recommend(self, student_id, n=3, exclude_modules=None, k_neighbors=20):
        """Find k most similar students, recommend their successful courses."""
        if exclude_modules is None:
            exclude_modules = set()

        if student_id not in self.student_features.index:
            return self._cold_start_recommend(n, exclude_modules)

        student_vec = self.student_features.loc[[student_id]].values
        all_vecs = self.student_features.values
        sims = cosine_similarity(student_vec, all_vecs).flatten()

        sim_series = pd.Series(sims, index=self.student_features.index)
        sim_series = sim_series.drop(student_id, errors="ignore")
        top_neighbors = sim_series.nlargest(k_neighbors)

        neighbor_success = self.success_matrix.loc[top_neighbors.index]
        weights = top_neighbors.values.reshape(-1, 1)
        weighted_scores = (neighbor_success.values * weights).sum(axis=0) / max(weights.sum(), 1e-9)
        course_scores = pd.Series(weighted_scores, index=self.success_matrix.columns)

        course_scores = course_scores.drop(labels=list(exclude_modules), errors="ignore")
        course_scores = course_scores.sort_values(ascending=False)

        return [(mod, score) for mod, score in course_scores.head(n).items()]

    def _cold_start_recommend(self, n, exclude_modules):
        """Cold-start: recommend most universally successful courses."""
        avg_success = self.success_matrix.mean(axis=0).sort_values(ascending=False)
        avg_success = avg_success.drop(labels=list(exclude_modules), errors="ignore")
        return [(mod, score) for mod, score in avg_success.head(n).items()]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_recommendations(tables, content_rec, collab_rec, n=3):
    """Evaluation using two metrics:
    1. Success prediction: do recommended courses have higher success rates for similar students?
    2. Coverage: how many distinct courses appear in recommendations?
    3. Holdout hit rate: for multi-course students, can we recover a hidden successful course?
    """
    si = tables["studentInfo"].copy()
    si["success"] = si["final_result"].isin(["Pass", "Distinction"]).astype(int)
    all_modules = set(si["code_module"].unique())

    np.random.seed(RANDOM_STATE)

    # Metric 1: Average success rate of recommended courses for each student
    sample_ids = si["id_student"].drop_duplicates().sample(min(1000, si["id_student"].nunique()), random_state=RANDOM_STATE)
    student_actual = si.groupby("id_student")[["code_module", "success"]].apply(
        lambda x: dict(zip(x["code_module"], x["success"]))
    )

    results = {"Content-Based": {"avg_rec_success": [], "hits": 0, "total": 0, "rec_modules": set()},
               "Collaborative": {"avg_rec_success": [], "hits": 0, "total": 0, "rec_modules": set()}}

    for sid in sample_ids:
        for name, rec in [("Content-Based", content_rec), ("Collaborative", collab_rec)]:
            try:
                recs = rec.recommend(sid, n=n)
                rec_modules = [r[0] for r in recs]
                results[name]["rec_modules"].update(rec_modules)

                # Check if student actually took any recommended courses and succeeded
                if sid in student_actual.index:
                    actual = student_actual[sid]
                    for m in rec_modules:
                        if m in actual:
                            results[name]["avg_rec_success"].append(actual[m])
            except Exception:
                pass

    # Metric 2: Holdout for multi-course students
    # Hide one successful course, use remaining as "known history", check if hidden appears in recs
    multi = si[si["success"] == 1].groupby("id_student")["code_module"].apply(list).reset_index()
    multi = multi[multi["code_module"].apply(len) >= 2]

    for _, row in multi.iterrows():
        sid = row["id_student"]
        modules = row["code_module"]
        hidden = np.random.choice(modules)
        visible = set(modules) - {hidden}

        # Content-based: pass visible modules as known history, exclude visible from recs
        try:
            recs = content_rec.recommend(sid, n=n, exclude_modules=visible, known_modules=visible)
            rec_modules = [r[0] for r in recs]
            if hidden in rec_modules:
                results["Content-Based"]["hits"] += 1
            results["Content-Based"]["total"] += 1
        except Exception:
            pass

        # Collaborative: exclude visible modules from recs
        try:
            recs = collab_rec.recommend(sid, n=n, exclude_modules=visible)
            rec_modules = [r[0] for r in recs]
            if hidden in rec_modules:
                results["Collaborative"]["hits"] += 1
            results["Collaborative"]["total"] += 1
        except Exception:
            pass

    print("\n  Evaluation Results:")
    for name, res in results.items():
        avg_success = np.mean(res["avg_rec_success"]) if res["avg_rec_success"] else 0
        hit_rate = res["hits"] / max(res["total"], 1)
        coverage = len(res["rec_modules"]) / len(all_modules)
        print(f"\n    {name}:")
        print(f"      Avg success rate of rec'd courses: {avg_success:.3f}")
        print(f"      Holdout hit rate @{n}: {res['hits']}/{res['total']} = {hit_rate:.3f}")
        print(f"      Coverage: {len(res['rec_modules'])}/{len(all_modules)} = {coverage:.1%}")

    baseline_success = si["success"].mean()
    print(f"\n    Baseline success rate (random): {baseline_success:.3f}")

    return results


def plot_recommendation_comparison(eval_results):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(eval_results.keys())
    hit_rates = [eval_results[n]["hits"] / max(eval_results[n]["total"], 1) for n in names]
    colors = ["#3498db", "#e74c3c"]

    bars = ax.bar(names, hit_rates, color=colors, alpha=0.8, width=0.5)
    for bar, rate in zip(bars, hit_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{rate:.1%}", ha="center", fontsize=12, fontweight="bold")

    ax.set_title("Recommendation Accuracy: Hit Rate @3", fontweight="bold")
    ax.set_ylabel("Hit Rate (hidden course in top-3)")
    ax.set_ylim(0, max(hit_rates) * 1.3 + 0.05)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task3_recommendation_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: task3_recommendation_comparison.png")


def demo_recommendations(content_rec, collab_rec, tables):
    """Show sample recommendations for 3 different student profiles."""
    si = tables["studentInfo"]
    np.random.seed(RANDOM_STATE)

    # Pick 3 diverse students
    dist_students = si[si["final_result"] == "Distinction"]["id_student"].unique()
    pass_students = si[si["final_result"] == "Pass"]["id_student"].unique()

    demo_ids = [
        np.random.choice(dist_students),
        np.random.choice(pass_students),
        -1,  # cold start
    ]
    labels = ["High Achiever", "Average Student", "New Student (Cold Start)"]

    print("\n  Sample Recommendations:")
    print("  " + "-" * 55)
    for sid, label in zip(demo_ids, labels):
        print(f"\n  {label} (ID: {sid}):")

        for name, rec in [("Content-Based", content_rec), ("Collaborative", collab_rec)]:
            try:
                recs = rec.recommend(sid, n=3)
                rec_str = ", ".join(f"{m} ({s:.2f})" for m, s in recs)
                print(f"    {name}: {rec_str}")
            except Exception as e:
                print(f"    {name}: Error - {e}")


def run():
    print("=" * 60)
    print("TASK 3: Course Recommendation Engine")
    print("=" * 60)

    tables = load_all()

    print("\n[1/5] Building student profiles...")
    student_profiles = build_student_profiles(tables)
    print(f"  -> {len(student_profiles):,} student-course enrollments")

    print("\n[2/5] Building course profiles...")
    course_profiles = build_course_profiles(tables)
    print(f"  -> {len(course_profiles)} unique modules")
    print(course_profiles[["code_module", "avg_pass_rate", "withdrawal_rate", "total_students"]].to_string(index=False))

    print("\n[3/5] Training Content-Based recommender...")
    content_rec = ContentBasedRecommender(student_profiles, course_profiles, tables)
    print("  -> Ready (cosine similarity on course feature vectors)")

    print("\n[4/5] Training Collaborative Filtering recommender...")
    collab_rec = CollaborativeRecommender(student_profiles, tables)
    print(f"  -> Ready ({len(collab_rec.student_features)} students in similarity matrix)")

    print("\n[5/5] Evaluating and comparing...")
    eval_results = evaluate_recommendations(tables, content_rec, collab_rec)
    plot_recommendation_comparison(eval_results)
    demo_recommendations(content_rec, collab_rec, tables)

    print("\n  Cold-Start Strategy:")
    print("  For new students with no history, we recommend courses with the")
    print("  highest pass rates and lowest withdrawal rates — the 'safest' choices.")
    print("  As the student completes their first course, the system switches to")
    print("  personalized recommendations based on their actual performance.")

    student_profiles.to_csv(os.path.join(FIG_DIR, "..", "task3_student_profiles.csv"), index=False)
    course_profiles.to_csv(os.path.join(FIG_DIR, "..", "task3_course_profiles.csv"), index=False)
    print("\nTask 3 complete.")

    return content_rec, collab_rec, eval_results


if __name__ == "__main__":
    run()
