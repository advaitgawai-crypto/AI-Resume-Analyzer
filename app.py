#!/usr/bin/env python3
"""
================================================================================
AI RESUME ANALYZER - LOCAL WEB APP (Flask backend)
================================================================================
Run this to start the local website:

    python app.py

Then open in your browser:

    http://localhost:5000

WHAT IT DOES
------------
1. Serves the frontend (index.html / style.css / script.js) from ./frontend
2. /api/analyze   - accepts an uploaded job posting PDF, runs the same NLP
                     pipeline as resume_analyzer.py, and returns a JSON report
3. /api/search-jobs - accepts skills + city + country, calls the JSearch API
                     (RapidAPI) and returns matching jobs

The heavy stuff (spaCy model, 2,716 resumes, TF-IDF vectorizers, resume
vectors) is loaded ONCE when the server starts, so every analysis after
that is fast.
================================================================================
"""

import os
import re
import sys
import json
import pickle
import spacy
import pandas as pd
import numpy as np
import pdfplumber
import requests
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.absolute()
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
UPLOAD_DIR = PROJECT_ROOT / "data" / "input" / "web_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# We use the NEW model which has 236 comprehensive skills built into the EntityRuler!
NER_MODEL_PATH = PROJECT_ROOT / "models" / "ner_model_v3"
RESUMES_WITH_ENTITIES = DATA_PROCESSED / "resumes_with_entities.csv"

# JSearch API credentials (OpenWeb Ninja)
# NOTE: it's better practice to load this from an environment variable
# (e.g. os.environ.get("API_KEY")) instead of hardcoding it, especially
# before this project ever goes public on GitHub.
OPENWEBNINJA_API_KEY = "ak_03q5qhkasx5xo2e4x6mpu3bc3i4i2nuafrqm9jwle5it0rq"

JSEARCH_URL = "https://api.openwebninja.com/jsearch/search-v2"

WEIGHTS = {
    'SKILL': 0.40,
    'JOB_TITLE': 0.25,
    'DEGREE': 0.20,
    'INSTITUTION': 0.10,
    'CERTIFICATION': 0.05,
    'EXPERIENCE_DURATION': 0.10
}
WEIGHTS_SUM = sum(WEIGHTS.values())
WEIGHTS = {k: v / WEIGHTS_SUM for k, v in WEIGHTS.items()}

TOP_N = 50
IMPROVEMENT_TOP_N = 10
ENTITY_TYPES = ['SKILL', 'JOB_TITLE', 'DEGREE', 'INSTITUTION', 'CERTIFICATION', 'EXPERIENCE_DURATION']

COUNTRY_MAP = {
    "au": "Australia", "at": "Austria", "be": "Belgium", "br": "Brazil",
    "ca": "Canada", "fr": "France", "de": "Germany", "in": "India",
    "it": "Italy", "mx": "Mexico", "nl": "Netherlands", "nz": "New Zealand",
    "pl": "Poland", "sg": "Singapore", "za": "South Africa", "es": "Spain",
    "ch": "Switzerland", "gb": "United Kingdom", "us": "United States",
}


# ============================================================================
# PIPELINE FUNCTIONS (same logic as resume_analyzer.py)
# ============================================================================

def extract_pdf_text(pdf_path: Path) -> str:
    text = ""
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_entities_ner(text: str, nlp_model: spacy.Language) -> Dict[str, List[str]]:
    """Extract entities using the new enhanced spaCy NER model (v3)."""
    doc = nlp_model(text)
    entities = {ent_type: [] for ent_type in ENTITY_TYPES[:-1]}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))
    return entities


def extract_experience_duration(text: str, nlp_model: spacy.Language) -> Optional[int]:
    patterns = [
        r'(\d+)\+?\s+(?:years?|yrs?)',
        r'(\d+)\s*-\s*(\d+)\s+(?:years?|yrs?)',
        r'(\d+)\s+(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(2)) if len(match.groups()) == 2 else int(match.group(1))
    return None


def vectorize_entities(entities: Dict[str, List[str]], vectorizers_dict: Dict) -> Dict:
    job_vectors = {}
    for entity_type, entity_list in entities.items():
        if entity_type in vectorizers_dict:
            entity_text = " ".join(entity_list) if entity_list else ""
            try:
                job_vectors[entity_type] = vectorizers_dict[entity_type].transform([entity_text])
            except Exception:
                job_vectors[entity_type] = None
    return job_vectors


def calculate_similarity_scores(job_vectors: Dict, resume_vectors_dict: Dict) -> Dict:
    similarities = {}
    for entity_type in ENTITY_TYPES[:-1]:
        if job_vectors.get(entity_type) is not None and entity_type in resume_vectors_dict:
            try:
                similarities[entity_type] = cosine_similarity(
                    job_vectors[entity_type], resume_vectors_dict[entity_type])[0]
            except Exception:
                similarities[entity_type] = np.zeros(resume_vectors_dict[entity_type].shape[0])
        else:
            similarities[entity_type] = None
    return similarities


def handle_experience_distance(job_experience: Optional[int], resume_experience: Optional[int]) -> float:
    if job_experience is None or resume_experience is None:
        return 0.5
    diff = abs(job_experience - resume_experience)
    if diff <= 1:
        return 1.0
    elif resume_experience > job_experience:
        return 0.9
    else:
        missing_years = job_experience - resume_experience
        penalty = min(missing_years / max(job_experience, 1), 0.7)
        return max(0.3, 1.0 - penalty)


def rank_resumes_by_category(similarities: Dict, resume_ids_arr: np.ndarray, job_experience: Optional[int],
                              resumes_df: pd.DataFrame, top_n: int = TOP_N) -> Dict:
    rankings = {}
    for entity_type in ENTITY_TYPES:
        if entity_type == 'EXPERIENCE_DURATION':
            exp_scores = []
            for _, row in resumes_df.iterrows():
                resume_exp = None
                if pd.notna(row['EXPERIENCE_DURATION']) and str(row['EXPERIENCE_DURATION']).strip():
                    exp_entries = str(row['EXPERIENCE_DURATION']).split('|')
                    if exp_entries:
                        m = re.search(r'\d+', exp_entries[0])
                        if m:
                            resume_exp = int(m.group())
                exp_scores.append(handle_experience_distance(job_experience, resume_exp))
            scores = np.array(exp_scores)
        elif similarities[entity_type] is not None:
            scores = similarities[entity_type]
        else:
            scores = np.zeros(len(resumes_df))

        ranking_df = pd.DataFrame({
            'resume_id': resume_ids_arr,
            'rank': range(1, len(resume_ids_arr) + 1),
            f'{entity_type}_score': scores
        })
        ranking_df = ranking_df.sort_values(by=f'{entity_type}_score', ascending=False).reset_index(drop=True)
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        rankings[entity_type] = ranking_df.head(top_n).copy()

    return rankings


def generate_improvement_recommendations(job_entities: Dict, rankings: Dict,
                                          resumes_df: pd.DataFrame, top_n: int = IMPROVEMENT_TOP_N) -> Dict:
    """Returns a plain dict (JSON-friendly) with recommendations and a weighted profile score."""

    report = {
        "required_skills": [],
        "strengths": [],
        "strengths_note": "",
        "skills_to_develop": [],
        "action_plan": "",
        "profile_score": 0
    }

    top_resumes = rankings['SKILL'].head(top_n)['resume_id'].tolist()
    job_skills = set(job_entities.get('SKILL', []))

    if not job_skills:
        return report

    report["required_skills"] = sorted(job_skills)

    all_top_skills = []
    for resume_id in top_resumes:
        resume_row = resumes_df[resumes_df['resume_id'] == resume_id].iloc[0]
        skills_str = resume_row.get('SKILL', "")
        skills = set(skills_str.split('|')) if pd.notna(skills_str) else set()
        all_top_skills.extend(skills)

    if top_resumes:
        top_resume_row = resumes_df[resumes_df['resume_id'] == top_resumes[0]].iloc[0]
        top_resume_skills_str = top_resume_row.get('SKILL', "")
        top_resume_skills = set(top_resume_skills_str.split('|')) if pd.notna(top_resume_skills_str) else set()
        matching_skills = top_resume_skills & job_skills
        if matching_skills:
            report["strengths"] = sorted(matching_skills)
            report["strengths_note"] = f"You already have {len(matching_skills)} of the top required skills! Great foundation!"

    skill_counts = Counter(all_top_skills)
    missing_skills = []
    
    total_possible_points = 0
    user_points = 0
    
    # Calculate score based on the top 20 most relevant skills from peer matches
    top_20_skills = skill_counts.most_common(20)
    for skill, count in top_20_skills:
        if count >= top_n - 2:
            priority = "HIGH"
            weight = 3
        elif count >= top_n / 2:
            priority = "MEDIUM"
            weight = 2
        else:
            priority = "LOW"
            weight = 1
            
        total_possible_points += weight
        
        if skill in job_skills:
            user_points += weight
        else:
            missing_skills.append({"skill": skill, "priority": priority})

    if total_possible_points > 0:
        report["profile_score"] = int(round((user_points / total_possible_points) * 100))

    report["skills_to_develop"] = missing_skills[:8]

    high_count = len([s for s in missing_skills if s['priority'] == 'HIGH'])
    med_count = len([s for s in missing_skills if s['priority'] == 'MEDIUM'])
    if high_count > 0 or med_count > 0:
        report["action_plan"] = (
            f"Learn {high_count} HIGH priority skill(s) + {med_count} MEDIUM priority skill(s) "
            f"to match {high_count + med_count} more of the top candidate profiles."
        )

    return report


# ============================================================================
# JOB SEARCH FUNCTIONS (JSearch / RapidAPI version)
# ============================================================================

def search_jobs_jsearch(keywords: List[str], location: str, country_code: str = "in",
                         results_per_page: int = 30) -> List[Dict]:
    """
    Using OpenWeb Ninja's JSearch API (not RapidAPI's JSearh).
    
    JSearch doesn't AND multiple keywords together the way Adzuna does, so we
    keep the same "hardcode a broad query" trick that fixed the multi-skill
    bug: build a simple "<role> jobs in <location>" query instead of jamming
    every extracted skill into the search string.
    """
    query = f"developer jobs in {location}"

    headers = {
        "X-API-Key": OPENWEBNINJA_API_KEY,
    }

    params = {
        "query": query,
    }

    try:
        response = requests.get(JSEARCH_URL, headers=headers, params=params, timeout=15)
        print(f"[DEBUG] OpenWeb Ninja JSearch request URL: {response.url}")
        print(f"[DEBUG] JSearch response status: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        print(f"[DEBUG] Full API response keys: {list(data.keys())}")
        print(f"[DEBUG] data.get('data'): {data.get('data')}")
        data_obj = data.get('data', {})
        print(f"[DEBUG] data_obj type: {type(data_obj)}, keys: {list(data_obj.keys()) if isinstance(data_obj, dict) else 'N/A'}")
        jobs_list = data_obj.get('jobs', []) if isinstance(data_obj, dict) else []
        print(f"[DEBUG] jobs_list type: {type(jobs_list)}, length: {len(jobs_list)}")
        print(f"[DEBUG] JSearch returned {len(jobs_list)} jobs")
    except requests.exceptions.HTTPError as e:
        print(f"[DEBUG] JSearch HTTP Error: {response.status_code} - {response.text}")
        raise

    jobs = []
    for job in jobs_list[:results_per_page]:
        city = job.get('job_city') or ''
        country = job.get('job_country') or ''
        location_str = ", ".join([p for p in [city, country] if p])

        jobs.append({
            'job_title': job.get('job_title', ''),
            'company': job.get('employer_name', ''),
            'location': location_str,
            'salary_min': job.get('job_min_salary', None),
            'salary_max': job.get('job_max_salary', None),
            'salary_currency': job.get('job_salary_currency', 'USD'),
            'description': job.get('job_description', '') or '',
            'posting_date': job.get('job_posted_at_datetime_utc', ''),
            'application_url': job.get('job_apply_link', '')
        })
    return jobs


def score_jobs(jobs: List[Dict], required_skills: List[str]) -> List[Dict]:
    for job in jobs:
        description_lower = job['description'].lower()
        match_count = sum(1 for skill in required_skills if skill.lower() in description_lower)
        job['match_score'] = round((match_count / max(len(required_skills), 1)) * 100)
        job['matching_skills'] = match_count
    return jobs


# ============================================================================
# LOAD HEAVY RESOURCES ONCE AT STARTUP
# ============================================================================

print("=" * 80)
print("Starting AI Resume Analyzer web server...")
print("=" * 80)

print("[Startup] Loading upgraded NER model (v3)...")
nlp = None
try:
    if NER_MODEL_PATH.exists():
        nlp = spacy.load(str(NER_MODEL_PATH))
        print("[OK] NER model loaded (v3 - multi-profession)")
    else:
        print(f"[WARN] NER model path does not exist: {NER_MODEL_PATH}")
        nlp = spacy.load("en_core_web_sm")
        print("[OK] Loaded fallback spaCy model (en_core_web_sm)")
except Exception as e:
    print(f"[WARN] NER model failed to load: {e}")
    nlp = None

print("[Startup] Loading resume database...")
resume_df = None
resume_ids = np.array([])
try:
    if RESUMES_WITH_ENTITIES.exists():
        resume_df = pd.read_csv(RESUMES_WITH_ENTITIES)
        resume_ids = resume_df['resume_id'].values
        print(f"[OK] Loaded {len(resume_df)} resumes")
    else:
        print(f"[WARN] Resume database not found: {RESUMES_WITH_ENTITIES}")
        resume_df = pd.DataFrame()
        print("[WARN] Running with empty resume database (analysis will be limited)")
except Exception as e:
    print(f"[WARN] Failed to load resume database: {e}")
    resume_df = pd.DataFrame()

print("[Startup] Loading vectorizers...")
vectorizers = {}
for entity_type in ENTITY_TYPES[:-1]:
    vectorizer_path = DATA_PROCESSED / f"vectorizer_{entity_type.lower()}.pkl"
    if vectorizer_path.exists():
        try:
            with open(vectorizer_path, 'rb') as f:
                vectorizers[entity_type] = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load vectorizer for {entity_type}: {e}")
print(f"[OK] Loaded {len(vectorizers)} vectorizers")

print("[Startup] Loading resume vectors...")
resume_vectors = {}
for entity_type in ENTITY_TYPES[:-1]:
    vector_path = DATA_PROCESSED / f"resume_vectors_{entity_type.lower()}.npz"
    if vector_path.exists():
        try:
            resume_vectors[entity_type] = load_npz(str(vector_path))
        except Exception as e:
            print(f"[WARN] Failed to load resume vectors for {entity_type}: {e}")
print(f"[OK] Loaded {len(resume_vectors)} resume vector sets")

print("=" * 80)
print("[OK] Server ready! Open http://localhost:5000")
print("=" * 80)


# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__, static_folder="frontend", static_url_path="")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Please upload a PDF file"}), 400

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = secure_filename(file.filename)
        save_path = UPLOAD_DIR / f"{timestamp}_{filename}"
        file.save(save_path)

        # 1. Extract text
        job_text = extract_pdf_text(save_path)

        # 2. Extract entities using our new v3 model
        if nlp is None:
            return jsonify({"error": "NER model unavailable"}), 500
        job_entities = extract_entities_ner(job_text, nlp)

        # 3. Extract experience
        job_experience = extract_experience_duration(job_text, nlp)

        # 4. Vectorize + similarity
        job_vectors = vectorize_entities(job_entities, vectorizers)
        similarities = calculate_similarity_scores(job_vectors, resume_vectors)

        # 5. Rank resumes (handle empty dataframe)
        if resume_df.empty:
            return jsonify({
                "error": "Resume database not loaded. Local testing only."
            }), 503

        rankings = rank_resumes_by_category(similarities, resume_ids, job_experience, resume_df, top_n=TOP_N)

        # 6. Generate recommendations
        rec = generate_improvement_recommendations(job_entities, rankings, resume_df, top_n=IMPROVEMENT_TOP_N)

        report = {
            "job_posting_name": filename,
            "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "required_experience": job_experience,
            "core_skills": job_entities.get('SKILL', []),
            "job_titles": job_entities.get('JOB_TITLE', []),
            "required_skills": rec["required_skills"],
            "strengths": rec["strengths"],
            "strengths_note": rec["strengths_note"],
            "skills_to_develop": rec["skills_to_develop"],
            "action_plan": rec["action_plan"],
            "profile_score": rec["profile_score"],
        }

        return jsonify(report)

    except Exception as e:
        return jsonify({"error": f"Failed to analyze PDF: {str(e)}"}), 500


@app.route("/api/search-jobs", methods=["POST"])
def api_search_jobs():
    data = request.get_json(silent=True) or {}
    skills = data.get('skills', [])
    location = (data.get('location') or '').strip()
    country_code = (data.get('country_code') or 'in').strip().lower()

    if not skills:
        return jsonify({"error": "No skills provided"}), 400
    if not location:
        return jsonify({"error": "No location provided"}), 400

    try:
        jobs = search_jobs_jsearch(skills, location, country_code=country_code, results_per_page=30)

        if not jobs:
            return jsonify({"jobs": [], "country_name": COUNTRY_MAP.get(country_code, country_code)})

        jobs = score_jobs(jobs, skills)
        jobs = sorted(jobs, key=lambda x: x['match_score'], reverse=True)[:20]

        # trim long descriptions before sending to browser
        for j in jobs:
            desc = j.get('description', '')
            j['description'] = (desc[:220] + '…') if len(desc) > 220 else desc

        return jsonify({"jobs": jobs, "country_name": COUNTRY_MAP.get(country_code, country_code)})

    except requests.exceptions.HTTPError as e:
        return jsonify({"error": f"OpenWeb Ninja JSearch API error: {str(e)}"}), 502
    except requests.exceptions.Timeout:
        return jsonify({"error": "OpenWeb Ninja JSearch API timed out. Try again."}), 504
    except Exception as e:
        return jsonify({"error": f"Job search failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)