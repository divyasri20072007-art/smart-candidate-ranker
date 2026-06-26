"""
scoring_engine.py
------------------
The "AI Decision Engine" — fuses the five agents' independent scores
into one final, weighted, explainable ranking score plus a confidence
estimate.

Design choices (explainable by construction, not learned/black-box):

- Weights are explicit and configurable (DEFAULT_WEIGHTS below), so a
  recruiter can see and tune exactly how much each dimension matters.
- Trust & Risk is applied as a multiplicative penalty/cap rather than a
  simple additive term — a high-risk profile should never out-rank a
  trustworthy one purely by averaging, regardless of how skill/experience
  scores look in isolation.
- Confidence reflects (a) how much profile data was actually available,
  and (b) how much the agents agree with each other — if every agent
  independently lands in a similar range, the system is more confident
  than when scores are scattered.
"""

import statistics

DEFAULT_WEIGHTS = {
    "skill": 0.35,
    "experience": 0.25,
    "growth": 0.15,
    "behaviour": 0.15,
    # trust_risk is NOT in the weighted sum — see apply_trust_penalty()
}

RISK_PENALTY_FACTOR = {
    "Low": 1.00,
    "Medium": 0.88,
    "High": 0.65,
}


def apply_trust_penalty(raw_score: float, risk_level: str) -> float:
    factor = RISK_PENALTY_FACTOR.get(risk_level, 0.85)
    return raw_score * factor


def compute_confidence(agent_scores: dict, data_completeness: float) -> float:
    values = list(agent_scores.values())
    if len(values) >= 2:
        spread = statistics.pstdev(values)
        agreement = max(0.0, 1.0 - spread / 50.0)  # 50pt spread -> 0 agreement
    else:
        agreement = 0.5
    confidence = 0.6 * agreement + 0.4 * data_completeness
    return round(max(0.0, min(1.0, confidence)) * 100, 1)


def estimate_data_completeness(candidate: dict) -> float:
    fields = [
        "skills_list", "years_experience", "career_history_list",
        "project_summary", "certifications_count", "github_url",
        "behaviour_score", "github_activity_score",
    ]
    present = sum(1 for f in fields if candidate.get(f) not in (None, "", [], 0))
    return present / len(fields)


def combine(agent_results: dict, candidate: dict, weights=None) -> dict:
    """agent_results: {"skill": {...}, "experience": {...}, "growth": {...},
    "behaviour": {...}, "trust_risk": {...}}"""
    weights = weights or DEFAULT_WEIGHTS

    component_scores = {k: agent_results[k]["score"] for k in weights}
    raw_final = sum(component_scores[k] * w for k, w in weights.items())

    risk_level = agent_results["trust_risk"]["risk_level"]
    final_score = apply_trust_penalty(raw_final, risk_level)

    completeness = estimate_data_completeness(candidate)
    confidence = compute_confidence(component_scores, completeness)

    return {
        "final_score": round(final_score, 1),
        "confidence": confidence,
        "risk_level": risk_level,
        "skill_score": component_scores["skill"],
        "experience_score": component_scores["experience"],
        "growth_score": component_scores["growth"],
        "behaviour_score": component_scores["behaviour"],
        "trust_score": agent_results["trust_risk"]["score"],
    }
