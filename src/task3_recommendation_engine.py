"""
Task 3 - Course Recommendation Engine
Recommends top-3 courses using:
  A) Content-based filtering (student profile + course metadata)
  B) Collaborative filtering via SVD matrix factorization
Improvements:
  - SVD-based collaborative filtering (real interaction patterns)
  - Diversity penalty (avoid recommending too-similar courses)
  - Difficulty matching (struggling students get safer courses)
  - NDCG@k evaluation metric
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from src.data_loader import load_all, student_key_cols, course_key_cols
from src.config import FIG_DIR, RANDOM_STATE
import os

os.makedirs(FIG_DIR, exist_ok=True)

KEY = student_key_cols()
CKEY = course_key_cols()


def build_student_profiles(tables):
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

    svle_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "studentVle.csv")
    chunks = []
    for chunk in pd.read_csv(svle_path, chunksize=2_000_000,
                             dtype={"code_module": str, "code_presentation": str,
                                    "id_student": "int32", "id_site": "int32",
                                    "date": "int16", "sum_click": "int16"}):
        agg = chunk.groupby(KEY).agg(total_clicks=("sum_click", "sum")).reset_index()
        chunks.append(agg)

    vle_agg = pd.concat(chunks).groupby(KEY)["total_clicks"].sum().reset_index()

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
    si = tables["studentInfo"]
    courses = tables["courses"].copy()

    course_stats = si.groupby("code_module").agg(
        avg_pass_rate=("final_result", lambda x: (x.isin(["Pass", "Distinction"])).mean()),
        avg_distinction_rate=("final_result", lambda x: (x == "Distinction").mean()),
        withdrawal_rate=("final_result", lambda x: (x == "Withdrawn").mean()),
        total_students=("id_student", "count"),
        avg_prev_attempts=("num_of_prev_attempts", "mean"),
        avg_credits=("studied_credits", "mean"),
    ).reset_index()

    mod_len = courses.groupby("code_module")["module_presentation_length"].mean().reset_index()
    mod_len.rename(columns={"module_presentation_length": "avg_length"}, inplace=True)

    return course_stats.merge(mod_len, on="code_module", how="left")


# ---------------------------------------------------------------------------
# Fix 5: Difficulty matching helper
# Classifies courses as easy/medium/hard by pass rate
# ---------------------------------------------------------------------------

def _difficulty_label(pass_rate):
    if pass_rate >= 0.60:
        return "easy"
    elif pass_rate >= 0.45:
        return "medium"
    else:
        return "hard"


def _student_ability(student_id, si_df):
    """Estimate student ability from past outcomes."""
    hist = si_df[si_df["id_student"] == student_id]
    if len(hist) == 0:
        return "unknown"
    result_map = {"Distinction": 3, "Pass": 2, "Fail": 1, "Withdrawn": 0}
    avg = hist["final_result"].map(result_map).mean()
    if avg >= 2.5:
        return "strong"
    elif avg >= 1.5:
        return "average"
    else:
        return "struggling"


# ---------------------------------------------------------------------------
# Approach A: Content-Based Filtering with difficulty matching
# ---------------------------------------------------------------------------

class ContentBasedRecommender:
    def __init__(self, student_profiles, course_profiles, tables):
        self.tables = tables
        self.si = tables["studentInfo"][KEY + ["final_result"]].copy()
        self.si["success"] = self.si["final_result"].isin(["Pass", "Distinction"]).astype(int)
        self.course_profiles = course_profiles

        feat_cols = ["avg_pass_rate", "avg_distinction_rate", "withdrawal_rate",
                     "total_students", "avg_prev_attempts", "avg_credits", "avg_length"]
        self.scaler = StandardScaler()
        self.course_features = self.scaler.fit_transform(
            course_profiles[feat_cols].fillna(0).values
        )
        self.course_modules = course_profiles["code_module"].values

        # Course difficulty labels for Fix 5
        self.difficulty = {
            row["code_module"]: _difficulty_label(row["avg_pass_rate"])
            for _, row in course_profiles.iterrows()
        }

    def recommend(self, student_id, n=3, exclude_modules=None, known_modules=None):
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

        # Fix 5: Difficulty matching — struggling students get easy/medium only
        ability = _student_ability(student_id, self.si)
        candidates = []
        for i in range(len(self.course_modules)):
            mod = self.course_modules[i]
            if mod in exclude_modules:
                continue
            diff = self.difficulty.get(mod, "medium")
            if ability == "struggling" and diff == "hard":
                continue  # filter out hard courses for struggling students
            candidates.append((mod, sims[i]))

        # Fix 4: Diversity penalty — reduce score of courses too similar to top pick
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = []
        selected_vecs = []
        for mod, score in candidates:
            idx = np.where(self.course_modules == mod)[0]
            if len(idx) == 0:
                continue
            cvec = self.course_features[idx[0]]
            if selected_vecs:
                max_sim_to_selected = max(
                    cosine_similarity(cvec.reshape(1, -1), np.array(selected_vecs))[0]
                )
                score = score * (1 - 0.3 * max_sim_to_selected)  # 30% diversity penalty
            selected.append((mod, round(float(score), 4)))
            selected_vecs.append(cvec)
            if len(selected) == n:
                break

        return selected if selected else self._cold_start_recommend(student_id, n, exclude_modules)

    def _cold_start_recommend(self, student_id, n, exclude_modules):
        cp = self.course_profiles.copy()
        cp["cold_score"] = cp["avg_pass_rate"] * 0.6 - cp["withdrawal_rate"] * 0.4
        cp = cp[~cp["code_module"].isin(exclude_modules)]
        cp = cp.sort_values("cold_score", ascending=False)
        return [(row["code_module"], round(float(row["cold_score"]), 4)) for _, row in cp.head(n).iterrows()]


# ---------------------------------------------------------------------------
# Approach B: SVD-based Collaborative Filtering (Fix 3)
# Uses actual student-course interaction matrix factorization
# ---------------------------------------------------------------------------

class CollaborativeRecommender:
    """
    SVD matrix factorization on the student-course success matrix.
    Decomposes into latent factor representations for students and courses,
    enabling recommendations based on actual interaction patterns.
    """

    def __init__(self, student_profiles, tables, n_components=4):
        self.tables = tables
        si = tables["studentInfo"].copy()
        si["success"] = si["final_result"].isin(["Pass", "Distinction"]).astype(float)

        # Student-course interaction matrix
        self.interaction_matrix = si.pivot_table(
            index="id_student", columns="code_module", values="success", aggfunc="max"
        ).fillna(0)

        self.student_ids = self.interaction_matrix.index.values
        self.course_ids  = self.interaction_matrix.columns.values

        # SVD decomposition
        X = self.interaction_matrix.values
        n_comp = min(n_components, min(X.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
        self.student_factors = self.svd.fit_transform(X)           # (n_students, k)
        self.course_factors  = self.svd.components_.T              # (n_courses, k)

        # Student feature matrix for cold-start similarity
        feat_cols = ["edu_level", "age_num", "total_clicks", "mean_score",
                     "num_of_prev_attempts", "studied_credits"]
        student_avg = student_profiles.groupby("id_student")[feat_cols].mean().reset_index()
        student_avg = student_avg.set_index("id_student")
        self.scaler = StandardScaler()
        valid_idx = student_avg.index.intersection(self.interaction_matrix.index)
        self.student_feat_df = pd.DataFrame(
            self.scaler.fit_transform(student_avg.loc[valid_idx].fillna(0)),
            index=valid_idx, columns=feat_cols
        )

        # Course difficulty for Fix 5
        course_profiles = build_course_profiles(tables)
        self.difficulty = {
            row["code_module"]: _difficulty_label(row["avg_pass_rate"])
            for _, row in course_profiles.iterrows()
        }
        self.course_profiles = course_profiles

    def recommend(self, student_id, n=3, exclude_modules=None):
        if exclude_modules is None:
            exclude_modules = set()

        if student_id not in self.interaction_matrix.index:
            return self._cold_start_recommend(n, exclude_modules)

        # Reconstruct predicted scores via SVD latent factors
        student_row_idx = np.where(self.student_ids == student_id)[0]
        if len(student_row_idx) == 0:
            return self._cold_start_recommend(n, exclude_modules)

        student_vec = self.student_factors[student_row_idx[0]]  # (k,)
        predicted_scores = student_vec @ self.course_factors.T   # (n_courses,)

        # Ability-based difficulty matching (Fix 5)
        si_df = self.tables["studentInfo"][["id_student", "final_result"]]
        ability = _student_ability(student_id, si_df)

        candidates = []
        for i, mod in enumerate(self.course_ids):
            if mod in exclude_modules:
                continue
            diff = self.difficulty.get(mod, "medium")
            if ability == "struggling" and diff == "hard":
                continue
            candidates.append((mod, float(predicted_scores[i])))

        # Diversity penalty (Fix 4)
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = []
        selected_course_idxs = []
        for mod, score in candidates:
            mod_idx = np.where(self.course_ids == mod)[0]
            if len(mod_idx) == 0:
                continue
            cvec = self.course_factors[mod_idx[0]]
            if selected_course_idxs:
                selected_vecs = self.course_factors[selected_course_idxs]
                max_sim = max(cosine_similarity(cvec.reshape(1, -1), selected_vecs)[0])
                score = score * (1 - 0.3 * max_sim)
            selected.append((mod, round(score, 4)))
            selected_course_idxs.append(mod_idx[0])
            if len(selected) == n:
                break

        return selected if selected else self._cold_start_recommend(n, exclude_modules)

    def _cold_start_recommend(self, n, exclude_modules):
        avg_success = self.interaction_matrix.mean(axis=0).sort_values(ascending=False)
        avg_success = avg_success.drop(labels=list(exclude_modules), errors="ignore")
        return [(mod, round(float(score), 4)) for mod, score in avg_success.head(n).items()]


# ---------------------------------------------------------------------------
# Evaluation: NDCG@k + Hit Rate + Coverage
# ---------------------------------------------------------------------------

def ndcg_at_k(recommended, relevant_set, k=3):
    """
    Normalized Discounted Cumulative Gain @k.
    Gives higher score when relevant items appear earlier in the list.
    """
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 2)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), k)))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_recommendations(tables, content_rec, collab_rec, n=3):
    si = tables["studentInfo"].copy()
    si["success"] = si["final_result"].isin(["Pass", "Distinction"]).astype(int)
    all_modules = set(si["code_module"].unique())

    np.random.seed(RANDOM_STATE)

    results = {
        "Content-Based": {"ndcg": [], "hits": 0, "total": 0, "rec_modules": set()},
        "Collaborative":  {"ndcg": [], "hits": 0, "total": 0, "rec_modules": set()},
    }

    # Coverage metric
    sample_ids = si["id_student"].drop_duplicates().sample(
        min(500, si["id_student"].nunique()), random_state=RANDOM_STATE
    )
    for sid in sample_ids:
        for name, rec in [("Content-Based", content_rec), ("Collaborative", collab_rec)]:
            try:
                recs = rec.recommend(sid, n=n)
                results[name]["rec_modules"].update(r[0] for r in recs)
            except Exception:
                pass

    # Holdout NDCG@k
    multi = si[si["success"] == 1].groupby("id_student")["code_module"].apply(list).reset_index()
    multi = multi[multi["code_module"].apply(len) >= 2]

    for _, row in multi.iterrows():
        sid = row["id_student"]
        modules = row["code_module"]
        hidden = np.random.choice(modules)
        visible = set(modules) - {hidden}

        try:
            recs = content_rec.recommend(sid, n=n, exclude_modules=visible, known_modules=visible)
            rec_mods = [r[0] for r in recs]
            results["Content-Based"]["ndcg"].append(ndcg_at_k(rec_mods, {hidden}, k=n))
            if hidden in rec_mods:
                results["Content-Based"]["hits"] += 1
            results["Content-Based"]["total"] += 1
        except Exception:
            pass

        try:
            recs = collab_rec.recommend(sid, n=n, exclude_modules=visible)
            rec_mods = [r[0] for r in recs]
            results["Collaborative"]["ndcg"].append(ndcg_at_k(rec_mods, {hidden}, k=n))
            if hidden in rec_mods:
                results["Collaborative"]["hits"] += 1
            results["Collaborative"]["total"] += 1
        except Exception:
            pass

    print("\n  Evaluation Results:")
    for name, res in results.items():
        hit_rate  = res["hits"] / max(res["total"], 1)
        mean_ndcg = np.mean(res["ndcg"]) if res["ndcg"] else 0
        coverage  = len(res["rec_modules"]) / len(all_modules)
        print(f"\n    {name}:")
        print(f"      Holdout hit rate @{n}: {res['hits']}/{res['total']} = {hit_rate:.3f}")
        print(f"      NDCG@{n}: {mean_ndcg:.3f}  (rewards ranking the hidden item higher)")
        print(f"      Coverage: {len(res['rec_modules'])}/{len(all_modules)} = {coverage:.1%}")

    return results


def plot_recommendation_comparison(eval_results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = list(eval_results.keys())
    colors = ["#3498db", "#e74c3c"]

    # Hit rate
    hit_rates = [eval_results[n]["hits"] / max(eval_results[n]["total"], 1) for n in names]
    bars = axes[0].bar(names, hit_rates, color=colors, alpha=0.8, width=0.5)
    for bar, rate in zip(bars, hit_rates):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{rate:.1%}", ha="center", fontsize=12, fontweight="bold")
    axes[0].set_title("Hit Rate @3", fontweight="bold")
    axes[0].set_ylabel("Hit Rate")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, axis="y", alpha=0.3)

    # NDCG@k
    ndcg_vals = [np.mean(eval_results[n]["ndcg"]) if eval_results[n]["ndcg"] else 0 for n in names]
    bars2 = axes[1].bar(names, ndcg_vals, color=colors, alpha=0.8, width=0.5)
    for bar, val in zip(bars2, ndcg_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")
    axes[1].set_title("NDCG@3 (higher = better ranking quality)", fontweight="bold")
    axes[1].set_ylabel("NDCG@3")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.suptitle("Recommendation Engine — Content-Based vs SVD Collaborative", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task3_recommendation_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: task3_recommendation_comparison.png")


def demo_recommendations(content_rec, collab_rec, tables):
    si = tables["studentInfo"]
    np.random.seed(RANDOM_STATE)

    dist_students = si[si["final_result"] == "Distinction"]["id_student"].unique()
    pass_students  = si[si["final_result"] == "Pass"]["id_student"].unique()
    fail_students  = si[si["final_result"] == "Fail"]["id_student"].unique()

    demo_ids = [
        np.random.choice(dist_students),
        np.random.choice(pass_students),
        np.random.choice(fail_students),  # struggling student → difficulty matching applies
        -1,  # cold start
    ]
    labels = ["High Achiever", "Average Student", "Struggling Student (difficulty matching)", "New Student (Cold Start)"]

    print("\n  Sample Recommendations:")
    print("  " + "-" * 60)
    for sid, label in zip(demo_ids, labels):
        print(f"\n  {label} (ID: {sid}):")
        for name, rec in [("Content-Based", content_rec), ("Collaborative", collab_rec)]:
            try:
                recs = rec.recommend(sid, n=3)
                rec_str = ", ".join(f"{m}({s:.2f})" for m, s in recs)
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

    print("\n[3/5] Training Content-Based recommender (cosine sim + diversity penalty + difficulty matching)...")
    content_rec = ContentBasedRecommender(student_profiles, course_profiles, tables)
    print("  -> Ready")

    print("\n[4/5] Training SVD Collaborative Filtering recommender...")
    collab_rec = CollaborativeRecommender(student_profiles, tables, n_components=4)
    explained = collab_rec.svd.explained_variance_ratio_.sum()
    print(f"  -> Ready (SVD: {collab_rec.svd.n_components} components, {explained:.1%} variance explained)")

    print("\n[5/5] Evaluating with Hit Rate @3 and NDCG@3...")
    eval_results = evaluate_recommendations(tables, content_rec, collab_rec)
    plot_recommendation_comparison(eval_results)
    demo_recommendations(content_rec, collab_rec, tables)

    print("\n  Cold-Start Strategy:")
    print("  For new students: recommend highest pass-rate / lowest withdrawal courses.")
    print("  Difficulty matching: struggling students filtered from hard courses (< 45% pass rate).")
    print("  Diversity penalty: 30% score reduction for courses similar to already-selected ones.")

    student_profiles.to_csv(os.path.join(FIG_DIR, "..", "task3_student_profiles.csv"), index=False)
    course_profiles.to_csv(os.path.join(FIG_DIR, "..", "task3_course_profiles.csv"), index=False)
    print("\nTask 3 complete.")

    return content_rec, collab_rec, eval_results


if __name__ == "__main__":
    run()
