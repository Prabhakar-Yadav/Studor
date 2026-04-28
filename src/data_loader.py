import pandas as pd
from src.config import DATA_DIR
import os


def load_all():
    tables = {}
    for name in [
        "studentInfo",
        "studentVle",
        "vle",
        "assessments",
        "studentAssessment",
        "courses",
        "studentRegistration",
    ]:
        tables[name] = pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))
    return tables


def student_key_cols():
    return ["code_module", "code_presentation", "id_student"]


def course_key_cols():
    return ["code_module", "code_presentation"]
