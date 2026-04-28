import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

DAYS_PER_WEEK = 7
WEEK_6_CUTOFF = 6 * DAYS_PER_WEEK  # day 42

RANDOM_STATE = 42
