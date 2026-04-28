# PathAI Engine — STUDOR DS Screening Project

A data science pipeline for student engagement scoring, disengagement prediction, and course recommendations built on the **Open University Learning Analytics Dataset (OULAD)**.

## Dataset

**Open University Learning Analytics Dataset (OULAD)**  
- **Source**: https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad
- **Size**: ~32,000 students across 7 courses (22 course-presentation pairs)
- **Time Period**: Multiple academic presentations (semesters)
- **Features**: 
  - Student demographics (age, gender, education level, disability status, region)
  - VLE (Virtual Learning Environment) clickstream activity (~10M interactions)
  - Assessment scores and submission patterns
  - Course registration and outcome labels (Pass/Fail/Withdrawn/Distinction)
  
**Citation**: Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics Dataset. Scientific Data, 4, 170171.

The dataset automatically downloads via `kagglehub` on first run. See Setup section below.

## Project Structure

**In GitHub repo (source code):**
```
Studor/
├── main.py                          # Run all three tasks
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── .gitignore
├── notebooks/
│   └── PathAI_Engine_Analysis.ipynb # Interactive analysis notebook
├── outputs/                         # Pre-computed results (included for review)
│   ├── figures/
│   │   ├── task1_archetype_trajectories.png
│   │   ├── task1_score_distributions.png
│   │   ├── task1_feature_rationale.png
│   │   ├── task2_confusion_matrices.png
│   │   ├── task2_roc_pr_curves.png
│   │   ├── task2_feature_importance.png
│   │   ├── task2_calibration.png
│   │   └── task3_recommendation_comparison.png
│   ├── models/
│   │   └── xgb_disengagement_w6.joblib
│   ├── task1_weekly_scores.csv
│   ├── task1_archetypes.csv
│   ├── task2_features.csv
│   ├── task3_student_profiles.csv
│   └── task3_course_profiles.csv
└── src/
    ├── config.py                    # Shared constants
    ├── data_loader.py               # Dataset loading utilities
    ├── task1_behavioral_scoring.py  # Engagement scoring framework
    ├── task2_disengagement_model.py # Predictive disengagement model
    └── task3_recommendation_engine.py # Course recommendation engine
```

**Auto-downloaded on first run:**
```
data/                               # OULAD dataset (via kagglehub)
├── studentInfo.csv
├── studentVle.csv
├── vle.csv
├── assessments.csv
├── studentAssessment.csv
├── courses.csv
└── studentRegistration.csv
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download OULAD dataset (auto-downloads via kagglehub)
#    Or manually: search "OULAD dataset" on Kaggle and place CSVs in data/

# 3. Run the full pipeline
python main.py

# Or run individual tasks:
python -m src.task1_behavioral_scoring
python -m src.task2_disengagement_model
python -m src.task3_recommendation_engine
```

## Task Summaries

### Task 1: Behavioral Scoring Framework (35 pts)
- **7 behavioral features**: total clicks, active days, activity diversity, recency score, click trend slope, assessment submissions, submission lead time
- **Dynamic 0-100 engagement score** computed weekly using weighted feature scaling
- **4 student archetypes**: Steady Engager, Early Dropout, Late Recoverer, Declining Engager
- Clear separation between outcome groups validates feature choices

### Task 2: Predictive Disengagement Model (35 pts)
- **XGBoost classifier** using only Week 1-6 data (no leakage)
- **ROC-AUC: 0.864**, Recall: 96% at optimized threshold
- **Calibration analysis** and confusion matrix included
- **Top 3 risk drivers**: last active day, mean assessment score, TMA submission count
- **Staff alert design** with 3-tier escalation system

### Task 3: Course Recommendation Engine (30 pts)
- **Content-based filtering**: cosine similarity on course feature vectors
- **Collaborative filtering**: user-based with weighted neighbor success patterns
- **Holdout hit rate**: Content-based 76.4%, Collaborative 75.1%
- **Cold-start strategy**: recommend highest pass-rate / lowest withdrawal-rate courses
- Primary user: students seeking next-semester opportunities

## Key Design Decisions

1. **Recall over Precision** (Task 2): In an early warning system, missing an at-risk student (false negative) is far more costly than flagging a student who was fine (false positive). We optimized threshold selection for recall while maintaining F1 > 0.40.

2. **Removed `date_unregistration` feature** (Task 2): This feature was essentially leaking the outcome (unregistration = withdrawal). Removing it dropped ROC-AUC from 0.92 to 0.86 but made the model honest and production-ready.

3. **Content-based > Collaborative** (Task 3): With only 7 modules, content-based filtering leverages richer course metadata while collaborative filtering has limited signal. In a real deployment with hundreds of courses, collaborative filtering would likely close the gap.

## What I Would Do Differently With More Time

- Add SHAP explanations for individual student risk predictions
- Build a Streamlit/Dash dashboard for interactive exploration
- Implement a hybrid recommendation model (content + collaborative ensemble)
- Add temporal cross-validation (train on earlier presentations, test on later)
- Deploy the model as a REST API with batch scoring capability
