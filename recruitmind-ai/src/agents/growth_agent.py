"""
agents/growth_agent.py
------------------------
Infers career trajectory from a seniority-tagged career history string,
e.g. "Junior Developer -> Software Engineer -> Senior Engineer -> Lead".

Rather than asking an LLM to "guess" growth (which would be ungrounded),
this agent parses explicit seniority keywords from the candidate's own
stated history and measures whether the trend moves up, sideways, or
down — fully auditable.
"""

LEVEL_RANK = {
    "intern": 0,
    "trainee": 0,
    "junior": 1,
    "associate": 1,
    "engineer": 2,
    "developer": 2,
    "analyst": 2,
    "mid": 2,
    "senior": 3,
    "lead": 4,
    "staff": 4,
    "principal": 5,
    "manager": 5,
    "director": 6,
    "head": 6,
    "vp": 7,
    "chief": 7,
}


def _level_of(role_title: str) -> int:
    title_l = role_title.lower()
    ranks = [rank for kw, rank in LEVEL_RANK.items() if kw in title_l]
    return max(ranks) if ranks else 2  # assume mid-level if no signal


def score(candidate: dict, jd: dict, semantic_index) -> dict:
    history = candidate.get("career_history_list", [])  # list of role titles, chronological
    certifications = candidate.get("certifications_count", 0) or 0

    if len(history) >= 2:
        levels = [_level_of(role) for role in history]
        deltas = [levels[i + 1] - levels[i] for i in range(len(levels) - 1)]
        upward_steps = sum(1 for d in deltas if d > 0)
        downward_steps = sum(1 for d in deltas if d < 0)
        trend_score = (upward_steps - downward_steps) / max(1, len(deltas))
        trend_score = max(0.0, min(1.0, 0.5 + trend_score / 2))
        trend_note = (
            "consistent upward progression"
            if upward_steps > downward_steps
            else "flat trajectory"
            if upward_steps == downward_steps
            else "some role regression detected"
        )
    elif len(history) == 1:
        trend_score = 0.5
        trend_note = "single role on record — insufficient history to judge trend"
    else:
        trend_score = 0.4
        trend_note = "no structured career history provided"

    cert_score = min(1.0, certifications / 5.0)

    final = 0.75 * trend_score + 0.25 * cert_score
    score_pct = round(final * 100, 1)

    return {
        "score": score_pct,
        "evidence": f"{trend_note}; {certifications} certification(s) on record",
    }
