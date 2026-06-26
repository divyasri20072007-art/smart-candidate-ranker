"""
jd_parser.py
------------
Turns a raw Job Description (free text) into structured requirements:
required years of experience, seniority level, role title, and a clean
text blob used for semantic matching downstream.

No external API calls — pure regex / rule based extraction so it always
works offline and is fully explainable (every field traces back to a
specific snippet of the JD).
"""

import re

SENIORITY_LEVELS = [
    "intern", "junior", "associate", "mid-level", "mid level", "senior",
    "staff", "principal", "lead", "manager", "director", "head", "vp",
    "chief",
]

# A reasonably broad, extensible technical/soft skill vocabulary used to
# spot explicit skill mentions in the JD. In production this would be
# loaded from the candidate dataset's own skill taxonomy (see
# build_skill_vocab in semantic.py) and merged with this seed list.
SEED_SKILLS = [
    "python", "java", "javascript", "typescript", "react", "node", "node.js",
    "django", "flask", "fastapi", "spring", "sql", "nosql", "mongodb",
    "postgresql", "mysql", "aws", "azure", "gcp", "docker", "kubernetes",
    "machine learning", "deep learning", "nlp", "computer vision",
    "data engineering", "data science", "pandas", "numpy", "pytorch",
    "tensorflow", "scikit-learn", "rest api", "graphql", "microservices",
    "ci/cd", "git", "agile", "scrum", "leadership", "communication",
    "problem solving", "system design", "etl", "spark", "hadoop",
    "airflow", "kafka", "redis", "html", "css", "c++", "c#", "go", "golang",
    "rust", "linux", "devops", "terraform", "android", "ios", "swift",
    "kotlin", "product management", "ui/ux", "figma",
]

YEARS_PATTERN = re.compile(
    r"(\d{1,2})\s*\+?\s*(?:-|to)?\s*(\d{0,2})\s*\+?\s*years?", re.IGNORECASE
)


def extract_required_years(text: str) -> int:
    """Return the minimum years of experience mentioned in the JD (0 if none)."""
    matches = YEARS_PATTERN.findall(text)
    years = []
    for low, _high in matches:
        if low:
            years.append(int(low))
    return max(years) if years else 0


def extract_seniority(text: str) -> str:
    text_l = text.lower()
    found = [lvl for lvl in SENIORITY_LEVELS if lvl in text_l]
    # Prefer the most senior keyword found, in declared priority order
    priority = list(reversed(SENIORITY_LEVELS))
    for lvl in priority:
        if lvl in found:
            return lvl
    return "unspecified"


def extract_role_title(text: str) -> str:
    """Heuristic: the first non-empty line is usually the role title."""
    for line in text.strip().splitlines():
        line = line.strip(" :-\t")
        if line:
            return line[:120]
    return "Unspecified Role"


def extract_skills(text: str, extra_vocab=None) -> list:
    vocab = set(SEED_SKILLS)
    if extra_vocab:
        vocab.update(s.lower() for s in extra_vocab)
    text_l = text.lower()
    found = sorted({skill for skill in vocab if skill in text_l})
    return found


def extract_responsibilities(text: str) -> list:
    """Pull out bullet / line-based responsibility statements."""
    lines = text.splitlines()
    bullets = []
    for line in lines:
        stripped = line.strip(" \t-*•").strip()
        if len(stripped.split()) >= 4 and ("?" not in stripped[:1]):
            bullets.append(stripped)
    return bullets[:25]


def parse_job_description(text: str, skill_vocab=None) -> dict:
    """Main entry point. Returns a structured requirements dict."""
    return {
        "raw_text": text.strip(),
        "role_title": extract_role_title(text),
        "required_years": extract_required_years(text),
        "seniority": extract_seniority(text),
        "required_skills": extract_skills(text, skill_vocab),
        "responsibilities": extract_responsibilities(text),
    }
