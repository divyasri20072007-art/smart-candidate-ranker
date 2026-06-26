"""
agents/skill_agent.py
----------------------
Scores how well a candidate's skill-set semantically matches the JD —
not via exact keyword overlap, but via the shared SemanticIndex, plus an
explicit-overlap bonus for clearly-stated matching skills (which also
doubles as the human-readable evidence for the explanation).
"""


def score(candidate: dict, jd: dict, semantic_index) -> dict:
    candidate_skills_text = candidate.get("skills_text", "")
    jd_skills_text = " ".join(jd["required_skills"]) or jd["raw_text"]

    semantic_sim = semantic_index.similarity(candidate_skills_text, jd_skills_text)

    candidate_skill_set = {
        s.strip().lower() for s in candidate.get("skills_list", []) if s.strip()
    }
    jd_skill_set = set(jd["required_skills"])
    overlap = sorted(candidate_skill_set & jd_skill_set)

    overlap_ratio = (len(overlap) / len(jd_skill_set)) if jd_skill_set else 0.0

    # Blend semantic similarity (handles synonyms/related tech) with
    # explicit overlap (handles exact, high-confidence matches).
    final = 0.6 * semantic_sim + 0.4 * overlap_ratio
    score_pct = round(final * 100, 1)

    if overlap:
        evidence = f"Direct skill match on {', '.join(overlap[:6])}"
    else:
        evidence = "No exact skill keyword overlap, but related/transferable skills detected" \
            if semantic_sim > 0.25 else "Limited skill alignment with the role"

    return {
        "score": score_pct,
        "evidence": evidence,
        "matched_skills": overlap,
    }
