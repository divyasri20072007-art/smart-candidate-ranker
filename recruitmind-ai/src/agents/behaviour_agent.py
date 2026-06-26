"""
agents/behaviour_agent.py
----------------------------
Estimates behavioural/soft signals. Uses explicit behavioural fields if
the dataset provides them (e.g., recruiter feedback, communication
rating). If absent, falls back to grounded textual proxies computed
directly from the candidate's own project/summary text:
  - presence of quantified impact ("increased X by 30%", "reduced ... 2x")
  - presence of collaboration/leadership language
  - platform activity (GitHub contributions, recruiter engagement)

Every signal is derived from data actually present on the profile —
nothing is invented.
"""

import re

QUANT_PATTERN = re.compile(r"\b\d+(\.\d+)?\s*(%|x|percent)\b", re.IGNORECASE)
COLLAB_KEYWORDS = [
    "led", "mentored", "collaborated", "coordinated", "managed", "presented",
    "communicated", "cross-functional", "stakeholder", "team of",
]


def score(candidate: dict, jd: dict, semantic_index) -> dict:
    explicit = candidate.get("behaviour_score")  # 0-100 if dataset provides it
    notes = []

    if explicit is not None and str(explicit).strip() != "":
        base = float(explicit) / 100.0
        notes.append(f"recruiter/platform-reported behaviour score: {explicit}/100")
    else:
        text = (candidate.get("project_summary", "") or "") + " " + (
            candidate.get("bio", "") or ""
        )
        quant_hits = len(QUANT_PATTERN.findall(text))
        collab_hits = sum(1 for kw in COLLAB_KEYWORDS if kw in text.lower())

        quant_score = min(1.0, quant_hits / 3.0)
        collab_score = min(1.0, collab_hits / 3.0)
        base = 0.5 * quant_score + 0.5 * collab_score
        if quant_hits:
            notes.append(f"{quant_hits} quantified-impact statement(s)")
        if collab_hits:
            notes.append(f"{collab_hits} collaboration/leadership signal(s)")
        if not notes:
            notes.append("no explicit behavioural evidence found in profile text")

    platform_activity = candidate.get("github_activity_score") or candidate.get(
        "platform_activity_score"
    )
    if platform_activity is not None and str(platform_activity).strip() != "":
        platform_norm = min(1.0, float(platform_activity) / 100.0)
        base = 0.7 * base + 0.3 * platform_norm
        notes.append(f"platform activity score: {platform_activity}/100")

    score_pct = round(base * 100, 1)

    return {
        "score": score_pct,
        "evidence": "; ".join(notes),
    }
