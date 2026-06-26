"""
agents/trust_risk_agent.py
----------------------------
Flags inconsistent, low-quality, or suspicious profiles so the Decision
Engine can downweight or surface a warning instead of silently ranking
an unreliable profile highly.

All checks are simple, explainable, rule-based heuristics — deliberately
NOT a black-box ML risk score, because risk/trust judgments are exactly
where explainability matters most to a recruiter.
"""

import re

GENERIC_PHRASES = [
    "hard working team player", "highly motivated individual",
    "results-driven professional", "passionate about technology",
]

GITHUB_URL_PATTERN = re.compile(r"^https?://(www\.)?github\.com/[\w-]+/?$")


def score(candidate: dict, jd: dict, semantic_index) -> dict:
    flags = []
    risk_points = 0  # higher = riskier

    years = float(candidate.get("years_experience", 0) or 0)
    num_roles = len(candidate.get("career_history_list", []) or [])
    if num_roles >= 2:
        avg_tenure = years / num_roles
        if avg_tenure < 0.4 and years > 0:
            flags.append("unusually short average tenure per role")
            risk_points += 2

    skills_count = len(candidate.get("skills_list", []) or [])
    if years <= 1 and skills_count >= 15:
        flags.append("very high skill count relative to experience level")
        risk_points += 2

    github_url = (candidate.get("github_url") or "").strip()
    if github_url and not GITHUB_URL_PATTERN.match(github_url):
        flags.append("malformed or non-standard GitHub URL")
        risk_points += 1

    bio = (candidate.get("bio", "") or "").lower()
    project_summary = (candidate.get("project_summary", "") or "").lower()
    combined_text = f"{bio} {project_summary}"
    generic_hits = sum(1 for phrase in GENERIC_PHRASES if phrase in combined_text)
    if generic_hits:
        flags.append("profile text relies on generic/boilerplate phrasing")
        risk_points += generic_hits

    if not combined_text.strip() and not candidate.get("skills_list"):
        flags.append("profile has almost no verifiable content")
        risk_points += 3

    # Map risk points -> 0-100 trust score (100 = fully trusted) and a label.
    trust_score = max(0, 100 - risk_points * 15)
    if risk_points == 0:
        level = "Low"
    elif risk_points <= 2:
        level = "Medium"
    else:
        level = "High"

    evidence = "; ".join(flags) if flags else "no inconsistencies detected"

    return {
        "score": trust_score,
        "risk_level": level,
        "evidence": evidence,
    }
