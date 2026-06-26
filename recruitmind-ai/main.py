#!/usr/bin/env python3
"""
main.py
--------
Command-line entry point for RecruitMind AI.

Usage:
    python main.py --jd data/sample_jd.txt --candidates data/sample_candidates.csv \
        --output output/ranked_candidates.csv --top-n 10

Run `python main.py --help` for all options.
"""

import argparse
import json
import os
import sys

import pandas as pd

from src import data_loader, pipeline


def parse_args():
    p = argparse.ArgumentParser(
        description="RecruitMind AI — explainable multi-agent candidate ranking."
    )
    p.add_argument("--jd", required=True, help="Path to a job description .txt file")
    p.add_argument(
        "--candidates", required=True, help="Path to candidates dataset (.csv or .json)"
    )
    p.add_argument(
        "--output", default="output/ranked_candidates.csv",
        help="Path to write the ranked output file (.csv or .json)",
    )
    p.add_argument(
        "--column-map", default=None,
        help="Optional JSON file mapping internal field names to your dataset's actual column names",
    )
    p.add_argument("--top-n", type=int, default=None, help="Only output the top N candidates")
    p.add_argument(
        "--no-embeddings", action="store_true",
        help="Force the offline TF-IDF semantic backend even if sentence-transformers is installed",
    )
    p.add_argument(
        "--weights", default=None,
        help='Optional JSON string overriding agent weights, e.g. \'{"skill":0.4,"experience":0.3,"growth":0.15,"behaviour":0.15}\'',
    )
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.jd, "r", encoding="utf-8") as f:
        jd_text = f.read()

    candidates = data_loader.load_candidates(args.candidates, column_map_path=args.column_map)
    if not candidates:
        print("No candidates loaded — check your dataset path/columns.", file=sys.stderr)
        sys.exit(1)

    weights = json.loads(args.weights) if args.weights else None

    results, meta = pipeline.run_pipeline(
        jd_text, candidates, weights=weights, prefer_embeddings=not args.no_embeddings
    )

    if args.top_n:
        results = results[: args.top_n]

    print(f"Role: {meta['role_title']}")
    print(f"Required years: {meta['required_years']}  |  Required skills: {', '.join(meta['required_skills']) or 'n/a'}")
    print(f"Semantic backend: {meta['semantic_backend']}")
    print(f"Candidates scored: {meta['num_candidates']}")
    print("-" * 80)
    for r in results[:10]:
        print(f"#{r['rank']:>2}  {r['name']:<30}  score={r['final_score']:>5}  conf={r['confidence']:>5}%  risk={r['risk_level']}")

    out_df = pd.DataFrame(results)
    columns_order = [
        "rank", "candidate_id", "name", "final_score", "confidence", "risk_level",
        "skill_score", "experience_score", "growth_score", "behaviour_score",
        "trust_score", "matched_skills", "explanation",
    ]
    out_df = out_df[[c for c in columns_order if c in out_df.columns]]

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.output.lower().endswith(".json"):
        out_df.to_json(args.output, orient="records", indent=2)
    else:
        out_df.to_csv(args.output, index=False)

    print("-" * 80)
    print(f"Full ranked output written to: {args.output}")


if __name__ == "__main__":
    main()
