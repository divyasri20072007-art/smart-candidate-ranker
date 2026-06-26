"""
pipeline.py
------------
Orchestrates the full RecruitMind AI flow:

  JD text --> Requirement Extraction --> Semantic Understanding
      --> Resume/Profile Understanding --> Behaviour Analysis
      --> Hybrid Scoring --> Final Ranking --> AI Explanation

This mirrors the architecture described in the pitch deck exactly, so
the code and the explanation slide stay in sync.
"""

from . import jd_parser, scoring_engine, explain
from .semantic import SemanticIndex
from .agents import skill_agent, experience_agent, growth_agent, behaviour_agent, trust_risk_agent


def _candidate_corpus_text(candidate: dict) -> str:
    return " ".join(
        [
            candidate.get("skills_text", ""),
            candidate.get("career_history_text", ""),
            candidate.get("project_summary", ""),
            candidate.get("bio", ""),
        ]
    )


def run_pipeline(jd_text: str, candidates: list, weights: dict = None, prefer_embeddings: bool = True):
    # 1. Build a skill vocabulary from the dataset itself so JD parsing
    #    can recognise domain-specific skills it wasn't seeded with.
    dataset_skill_vocab = sorted({s for c in candidates for s in c.get("skills_list", [])})

    # 2. Requirement Extraction + Semantic Understanding of the JD.
    jd = jd_parser.parse_job_description(jd_text, skill_vocab=dataset_skill_vocab)

    # 3. Fit the semantic index once over JD + every candidate's text
    #    (needed for the TF-IDF fallback backend; embedding backend
    #    ignores the corpus argument).
    corpus = [jd["raw_text"]] + [_candidate_corpus_text(c) for c in candidates]
    semantic_index = SemanticIndex(corpus, prefer_embeddings=prefer_embeddings)

    results = []
    for candidate in candidates:
        # 4. Resume Understanding + Behaviour Analysis via specialised agents.
        agent_results = {
            "skill": skill_agent.score(candidate, jd, semantic_index),
            "experience": experience_agent.score(candidate, jd, semantic_index),
            "growth": growth_agent.score(candidate, jd, semantic_index),
            "behaviour": behaviour_agent.score(candidate, jd, semantic_index),
            "trust_risk": trust_risk_agent.score(candidate, jd, semantic_index),
        }

        # 5. Hybrid Scoring -> Final Ranking via the AI Decision Engine.
        combined = scoring_engine.combine(agent_results, candidate, weights=weights)

        # 6. AI Explanation — grounded purely in agent evidence strings.
        explanation = explain.build_explanation(candidate["name"], agent_results, combined)

        results.append(
            {
                "candidate_id": candidate["candidate_id"],
                "name": candidate["name"],
                **combined,
                "matched_skills": ", ".join(agent_results["skill"].get("matched_skills", [])),
                "explanation": explanation,
            }
        )

    results.sort(key=lambda r: r["final_score"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    meta = {
        "role_title": jd["role_title"],
        "required_years": jd["required_years"],
        "required_skills": jd["required_skills"],
        "semantic_backend": semantic_index.backend_name,
        "num_candidates": len(candidates),
    }
    return results, meta
