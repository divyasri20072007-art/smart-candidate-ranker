# RecruitMind AI

**An explainable, multi-agent AI hiring intelligence system.**
*"We don't rank resumes. We understand people."*

RecruitMind AI reads a job description, understands what the role actually
needs (not just its keywords), evaluates every candidate across five
independent dimensions, and produces a trustworthy, fully explainable
ranked shortlist — with no hallucinated justifications and explicit
flagging of inconsistent or low-quality profiles.

---

## Why this architecture

Traditional ATS keyword matching fails for two reasons: it can't see
*meaning* (a candidate who says "built scalable services on AWS" gets
missed by a filter looking for "AWS, microservices"), and it gives a
single opaque score with no way for a recruiter to trust or audit it.

RecruitMind AI fixes both problems by:

1. **Understanding the JD semantically**, not just extracting keywords
   (`src/jd_parser.py` + `src/semantic.py`).
2. **Evaluating candidates from five independent, specialised angles**
   instead of one monolithic score (`src/agents/`).
3. **Fusing those signals transparently** with explicit, tunable weights
   instead of a black-box model (`src/scoring_engine.py`).
4. **Explaining every score using only evidence the system actually
   computed** — never free-form LLM text disconnected from the data
   (`src/explain.py`). This is what prevents hallucinated justifications.
5. **Explicitly flagging risk** (inconsistent tenure, boilerplate text,
   malformed links, etc.) so an untrustworthy profile is demoted rather
   than silently ranked highly (`src/agents/trust_risk_agent.py`).

## Architecture

```
JD Input
   │
   ▼
Requirement Extraction  (jd_parser.py)
   │
   ▼
Semantic Understanding  (semantic.py — embeddings or TF-IDF)
   │
   ▼
   ├──────────────┬──────────────┬───────────────┬─────────────────┐
   ▼              ▼              ▼               ▼                 ▼
Skill Agent  Experience Agent  Growth Agent  Behaviour Agent  Trust & Risk Agent
   │              │              │               │                 │
   └──────────────┴──────┬───────┴───────────────┴─────────────────┘
                          ▼
                 AI Decision Engine (scoring_engine.py)
                          │
                          ▼
                 Final Ranking + Confidence
                          │
                          ▼
                 AI Explanation (explain.py)
                          │
                          ▼
                 Ranked Output File (CSV / JSON)
```

## Semantic matching: embeddings with an offline fallback

`src/semantic.py` will automatically use `sentence-transformers`
(`all-MiniLM-L6-v2`) for higher-quality semantic similarity **if** the
package and model are available. If not (e.g. no internet access at
runtime, which is common in locked-down judging environments), it falls
back to a TF-IDF + cosine-similarity index fit jointly over the JD and
every candidate's profile text — zero external dependencies, fully
deterministic, and still captures shared vocabulary/context rather than
requiring exact keyword matches.

To force the offline backend explicitly: `--no-embeddings`.
To use embeddings, just `pip install sentence-transformers` — no code
changes needed.

## Quick start

```bash
pip install -r requirements.txt

python main.py \
  --jd data/sample_jd.txt \
  --candidates data/sample_candidates.csv \
  --output output/ranked_candidates.csv
```

This runs the included demo JD + 10 synthetic candidates and writes a
fully ranked, explained shortlist to `output/ranked_candidates.csv`.

### Using your own dataset

The loader (`src/data_loader.py`) auto-detects common column-name
variants (e.g. `skills` / `key_skills` / `technical_skills` all map to
the same internal field). If your dataset uses different names, pass an
explicit mapping:

```bash
python main.py --jd my_jd.txt --candidates my_candidates.csv \
  --column-map column_map.json --output output/ranked.csv
```

`column_map.json` example:

```json
{
  "skills": "Skill_Set",
  "years_experience": "Total_Exp_Years",
  "career_history": "Employment_History",
  "github_url": "GitHub_Link"
}
```

### Tuning the scoring weights

```bash
python main.py --jd data/sample_jd.txt --candidates data/sample_candidates.csv \
  --weights '{"skill":0.4,"experience":0.3,"growth":0.15,"behaviour":0.15}' \
  --output output/ranked_candidates.csv
```

## Output format

| Column | Meaning |
|---|---|
| `rank` | Final rank (1 = best fit) |
| `candidate_id`, `name` | Identity |
| `final_score` | 0–100 weighted score after the trust/risk penalty |
| `confidence` | 0–100%, based on agent agreement + data completeness |
| `risk_level` | Low / Medium / High — from the Trust & Risk agent |
| `skill_score`, `experience_score`, `growth_score`, `behaviour_score`, `trust_score` | Per-dimension breakdown ("Hiring DNA") |
| `matched_skills` | Explicit skills overlapping the JD |
| `explanation` | Full grounded, human-readable explanation |

## Project layout

```
recruitmind-ai/
├── main.py                     # CLI entry point
├── requirements.txt
├── data/
│   ├── sample_jd.txt
│   └── sample_candidates.csv
├── output/
│   └── ranked_candidates.csv   # generated
├── tests/
│   └── test_pipeline.py
└── src/
    ├── jd_parser.py            # Requirement extraction
    ├── semantic.py             # Semantic Understanding (embeddings / TF-IDF)
    ├── data_loader.py          # Flexible dataset normalisation
    ├── scoring_engine.py       # AI Decision Engine (hybrid scoring)
    ├── explain.py              # Grounded explanation builder
    ├── pipeline.py             # Orchestrates the full flow
    └── agents/
        ├── skill_agent.py
        ├── experience_agent.py
        ├── growth_agent.py
        ├── behaviour_agent.py
        └── trust_risk_agent.py
```

## Running tests

```bash
python -m pytest tests/ -v
```

## Notes on the provided dataset

Point `--candidates` directly at the dataset from the challenge once
downloaded locally (Google Drive link in the brief — download it, then
pass the local CSV/JSON path). If its column names differ from the
common aliases already handled in `data_loader.py`, supply a
`--column-map` JSON as shown above; no code changes are required.
