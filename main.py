"""
PathAI Engine — STUDOR DS Screening Project
Run all three tasks sequentially.
"""

from src.task1_behavioral_scoring import run as run_task1
from src.task2_disengagement_model import run as run_task2
from src.task3_recommendation_engine import run as run_task3


def main():
    print("\n" + "=" * 60)
    print("  STUDOR PathAI Engine — Full Pipeline")
    print("=" * 60 + "\n")

    print("Running Task 1...")
    scored, archetypes = run_task1()

    print("\n\nRunning Task 2...")
    results, features = run_task2()

    print("\n\nRunning Task 3...")
    content_rec, collab_rec, eval_results = run_task3()

    print("\n" + "=" * 60)
    print("  ALL TASKS COMPLETE")
    print("  Check outputs/ directory for results and figures.")
    print("=" * 60)


if __name__ == "__main__":
    main()
