"""
agents/experience_agent.py
---------------------------
Scores relevance of a candidate's work history: do they have enough
years, and have they actually worked in relevant roles (judged
semantically against the JD's responsibilities/title, not just years).
"""


def score(candidate: dict, jd: dict, semantic_index) -> dict:
    years = float(candidate.get("years_experience", 0) or 0)
    required_years = jd.get("required_years", 0)

    if required_years <= 0:
        years_score = min(1.0, years / 8.0)  # no explicit requirement -> mild credit
    elif years >= required_years:
        # Reward meeting the bar; mild bonus for exceeding it, capped.
        bonus = min(0.25, (years - required_years) * 0.04)
        years_score = min(1.0, 0.75 + bonus)
    else:
        # Partial credit, shrinking fast the further below the bar.
        years_score = max(0.0, years / required_years) * 0.7

    history_text = candidate.get("career_history_text", "") or candidate.get(
        "current_role", ""
    )
    jd_context = " ".join([jd["role_title"]] + jd["responsibilities"]) or jd["raw_text"]
    relevance_sim = semantic_index.similarity(history_text, jd_context)

    final = 0.55 * years_score + 0.45 * relevance_sim
    score_pct = round(final * 100, 1)

    if required_years:
        years_note = f"{years:g} yrs experience vs {required_years}+ required"
    else:
        years_note = f"{years:g} yrs experience"

    relevance_note = (
        "highly relevant role history"
        if relevance_sim > 0.5
        else "moderately relevant role history"
        if relevance_sim > 0.25
        else "limited role relevance"
    )

    return {
        "score": score_pct,
        "evidence": f"{years_note}; {relevance_note}",
    }
