"""
tests/test_pipeline.py
------------------------
Basic sanity + behavioural tests for RecruitMind AI.

Run with: python -m pytest tests/ -v
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import data_loader, pipeline, jd_parser


SAMPLE_JD = """Senior Backend Engineer

Requirements:
- 5+ years of professional software engineering experience
- Strong proficiency in Python, SQL, and REST API design
- Experience with AWS, Docker, and Kubernetes
"""


def _candidates():
    return data_loader.load_candidates(
        os.path.join(os.path.dirname(__file__), "..", "data", "sample_candidates.csv")
    )


def test_jd_parser_extracts_years_and_skills():
    jd = jd_parser.parse_job_description(SAMPLE_JD)
    assert jd["required_years"] == 5
    assert "python" in jd["required_skills"]
    assert "aws" in jd["required_skills"]


def test_data_loader_loads_all_rows():
    candidates = _candidates()
    assert len(candidates) == 10
    assert all("name" in c and "skills_list" in c for c in candidates)


def test_pipeline_produces_full_ranking():
    candidates = _candidates()
    results, meta = pipeline.run_pipeline(SAMPLE_JD, candidates, prefer_embeddings=False)
    assert len(results) == len(candidates)
    # Ranks should be a contiguous 1..N sequence after sorting.
    assert sorted(r["rank"] for r in results) == list(range(1, len(candidates) + 1))
    # Scores should be sorted descending by rank.
    scores = [r["final_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_strong_fit_outranks_irrelevant_profile():
    candidates = _candidates()
    results, _ = pipeline.run_pipeline(SAMPLE_JD, candidates, prefer_embeddings=False)
    by_name = {r["name"]: r for r in results}
    # Meera Joshi (senior, strong overlapping skills) should clearly
    # outrank Rahul Kapoor (1 yr exp, near-empty/boilerplate profile).
    assert by_name["Meera Joshi"]["final_score"] > by_name["Rahul Kapoor"]["final_score"]


def test_risky_profile_is_flagged_and_penalised():
    candidates = _candidates()
    results, _ = pipeline.run_pipeline(SAMPLE_JD, candidates, prefer_embeddings=False)
    by_name = {r["name"]: r for r in results}
    suresh = by_name["Suresh Kumar"]
    assert suresh["risk_level"] in ("Medium", "High")


def test_explanation_is_grounded_and_nonempty():
    candidates = _candidates()
    results, _ = pipeline.run_pipeline(SAMPLE_JD, candidates, prefer_embeddings=False)
    for r in results:
        assert len(r["explanation"]) > 20
        assert "score" in r["explanation"].lower()
