"""
data_loader.py
---------------
Loads a candidate dataset (CSV or JSON) and normalizes it into the
internal candidate schema used by the agents, regardless of the exact
column names in the source file.

Real-world hackathon/recruiting datasets vary a lot in naming
conventions, so this module is intentionally flexible: it tries a list
of likely column-name aliases for every internal field, and you can
also pass an explicit --column-map JSON file to override anything.

Internal candidate schema (all optional except candidate_id/name):
  candidate_id, name, skills_list, skills_text, years_experience,
  current_role, career_history_list, career_history_text,
  project_summary, bio, certifications_count, github_url,
  github_activity_score, behaviour_score
"""

import json
import re

import pandas as pd

ALIASES = {
    "candidate_id": ["candidate_id", "id", "candidate id", "applicant_id"],
    "name": ["name", "full_name", "candidate_name", "applicant_name"],
    "skills": ["skills", "skill_set", "skillset", "key_skills", "technical_skills"],
    "years_experience": [
        "years_experience", "experience_years", "total_experience",
        "years_of_experience", "experience", "yoe",
    ],
    "current_role": ["current_role", "designation", "current_title", "title", "role"],
    "career_history": [
        "career_history", "work_history", "past_roles", "previous_roles",
        "job_history", "role_history",
    ],
    "project_summary": [
        "project_summary", "projects", "project_details", "summary",
        "key_projects",
    ],
    "bio": ["bio", "about", "profile_summary", "description", "objective"],
    "certifications": [
        "certifications", "certificates", "certification_count", "certs",
    ],
    "github_url": ["github_url", "github", "github_profile", "git_url"],
    "github_activity_score": [
        "github_activity_score", "github_score", "platform_activity_score",
        "platform_score", "activity_score",
    ],
    "behaviour_score": [
        "behaviour_score", "behavior_score", "soft_skill_score",
        "recruiter_rating", "communication_score",
    ],
}


def _find_column(df_columns_lower, aliases):
    for alias in aliases:
        if alias in df_columns_lower:
            return df_columns_lower[alias]
    return None


def _split_list_field(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value)
    parts = re.split(r"[;,|]", text)
    return [p.strip() for p in parts if p.strip()]


def load_candidates(path: str, column_map_path: str = None) -> list:
    if path.lower().endswith(".json"):
        raw_df = pd.read_json(path)
    else:
        raw_df = pd.read_csv(path)

    columns_lower = {c.strip().lower(): c for c in raw_df.columns}

    overrides = {}
    if column_map_path:
        with open(column_map_path) as f:
            overrides = json.load(f)

    def col(field):
        if field in overrides:
            return overrides[field]
        return _find_column(columns_lower, ALIASES.get(field, [field]))

    resolved = {field: col(field) for field in ALIASES}

    candidates = []
    for idx, row in raw_df.iterrows():
        def get(field, default=""):
            c = resolved[field]
            if c is None or c not in row or pd.isna(row[c]):
                return default
            return row[c]

        skills_list = _split_list_field(get("skills"))
        career_list = _split_list_field(get("career_history"))

        candidate = {
            "candidate_id": str(get("candidate_id", idx + 1)),
            "name": str(get("name", f"Candidate {idx + 1}")),
            "skills_list": skills_list,
            "skills_text": ", ".join(skills_list),
            "years_experience": float(get("years_experience", 0) or 0),
            "current_role": str(get("current_role", "")),
            "career_history_list": career_list,
            "career_history_text": " -> ".join(career_list) or str(get("current_role", "")),
            "project_summary": str(get("project_summary", "")),
            "bio": str(get("bio", "")),
            "certifications_count": _certs_count(get("certifications", "")),
            "github_url": str(get("github_url", "")),
            "github_activity_score": _safe_float(get("github_activity_score", None)),
            "behaviour_score": _safe_float(get("behaviour_score", None)),
        }
        candidates.append(candidate)

    return candidates


def _certs_count(value):
    if value in (None, "", 0):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return len(_split_list_field(value))


def _safe_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
