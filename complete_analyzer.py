#!/usr/bin/env python3
"""
================================================================================
COMPLETE JOB ANALYZER - Analyze Job + Find Matching Jobs (International)
================================================================================
Usage:
  python complete_analyzer.py <job_posting.pdf> <location> <country_code>

Examples:
  python complete_analyzer.py "data\\input\\job.pdf" "Bangalore" "in"      # India
  python complete_analyzer.py "data\\input\\job.pdf" "New York" "us"       # USA
  python complete_analyzer.py "data\\input\\job.pdf" "London" "gb"         # UK
  python complete_analyzer.py "data\\input\\job.pdf" "Toronto" "ca"        # Canada
  python complete_analyzer.py "data\\input\\job.pdf" "Sydney" "au"         # Australia
  python complete_analyzer.py "data\\input\\job.pdf" "Berlin" "de"         # Germany
  python complete_analyzer.py "data\\input\\job.pdf" "Paris" "fr"          # France

Supported Countries:
  in (India), us (USA), gb (UK), ca (Canada), au (Australia)
  de (Germany), fr (France), nl (Netherlands), it (Italy), es (Spain)
  se (Sweden), nz (New Zealand), sg (Singapore)

Pipeline:
1. Parse job posting PDF
2. Extract entities (skills, titles, degrees, etc.)
3. Generate skill development plan
4. Search Adzuna for matching jobs (international)
5. Rank and display results
================================================================================
"""

import os
import sys
import re
import pickle
import spacy
import pandas as pd
import numpy as np
import pdfplumber
import requests
from pathlib import Path
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
from collections import Counter
from typing import Dict, List, Optional


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer")
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "phase5"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESUMES_WITH_ENTITIES = DATA_PROCESSED / "resumes_with_entities.csv"
NER_MODEL_PATH = MODELS_DIR / "ner_model_v2"

# Adzuna API credentials
ADZUNA_APP_ID = "aacca97d"
ADZUNA_APP_KEY = "315d396533cd22919c1d827810601d77"

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


# ============================================================================
# PHASE 5 FUNCTIONS (Resume Analyzer)
# ============================================================================

def load_spacy_model(model_path: Path) -> spacy.Language:
    try:
        nlp = spacy.load(str(model_path))
        print(f"✓ Loaded NER model")
        return nlp
    except Exception as e:
        print(f"✗ Failed to load NER model: {e}")
        sys.exit(1)


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        text = ""
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        print(f"✓ Extracted text from {pdf_path.name}")
        return text
    except Exception as e:
        print(f"✗ Failed to read PDF: {e}")
        sys.exit(1)


def extract_entities_ner(text: str, nlp: spacy.Language) -> Dict[str, List[str]]:
    doc = nlp(text)
    entities = {ent_type: [] for ent_type in ENTITY_TYPES[:-1]}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))
    return entities


def extract_experience_duration(text: str, nlp: spacy.Language) -> Optional[int]:
    patterns = [
        r'(\d+)\+?\s+(?:years?|yrs?)',
        r'(\d+)\s*-\s*(\d+)\s+(?:years?|yrs?)',
        r'(\d+)\s+(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            years = int(match.group(2)) if len(match.groups()) == 2 else int(match.group(1))
            return years
    return None


def vectorize_entities(entities: Dict[str, List[str]], vectorizers: Dict) -> Dict:
    job_vectors = {}
    for entity_type, entity_list in entities.items():
        if entity_type in vectorizers:
            entity_text = " ".join(entity_list) if entity_list else ""
            try:
                vector = vectorizers[entity_type].transform([entity_text])
                job_vectors[entity_type] = vector
            except Exception as e:
                job_vectors[entity_type] = None
    return job_vectors


def load_phase4_vectorizers(data_dir: Path) -> Dict:
    vectorizers = {}
    for entity_type in ENTITY_TYPES[:-1]:
        vectorizer_path = data_dir / f"vectorizer_{entity_type.lower()}.pkl"
        if vectorizer_path.exists():
            try:
                with open(vectorizer_path, 'rb') as f:
                    vectorizers[entity_type] = pickle.load(f)
            except Exception as e:
                pass
    return vectorizers


def load_resume_vectors(data_dir: Path) -> Dict:
    resume_vectors = {}
    for entity_type in ENTITY_TYPES[:-1]:
        vector_path = data_dir / f"resume_vectors_{entity_type.lower()}.npz"
        if vector_path.exists():
            try:
                resume_vectors[entity_type] = load_npz(str(vector_path))
            except Exception as e:
                pass
    return resume_vectors


def calculate_similarity_scores(job_vectors: Dict, resume_vectors: Dict) -> Dict:
    similarities = {}
    for entity_type in ENTITY_TYPES[:-1]:
        if job_vectors.get(entity_type) is not None and entity_type in resume_vectors:
            try:
                sim = cosine_similarity(job_vectors[entity_type], resume_vectors[entity_type])[0]
                similarities[entity_type] = sim
            except Exception as e:
                similarities[entity_type] = np.zeros(resume_vectors[entity_type].shape[0])
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


def rank_resumes_by_category(similarities: Dict, resume_ids: np.ndarray, job_experience: Optional[int],
                              resume_df: pd.DataFrame, top_n: int = TOP_N) -> Dict:
    rankings = {}
    for entity_type in ENTITY_TYPES:
        if entity_type == 'EXPERIENCE_DURATION':
            try:
                exp_scores = []
                for _, row in resume_df.iterrows():
                    resume_exp = None
                    if pd.notna(row['EXPERIENCE_DURATION']) and row['EXPERIENCE_DURATION'].strip():
                        exp_entries = row['EXPERIENCE_DURATION'].split('|')
                        if exp_entries:
                            try:
                                resume_exp = int(re.search(r'\d+', exp_entries[0]).group())
                            except:
                                pass
                    exp_scores.append(handle_experience_distance(job_experience, resume_exp))
                scores = np.array(exp_scores)
            except Exception as e:
                scores = np.zeros(len(resume_df))
        elif similarities[entity_type] is not None:
            scores = similarities[entity_type]
        else:
            scores = np.zeros(len(resume_df))

        ranking_df = pd.DataFrame({
            'resume_id': resume_ids,
            'rank': range(1, len(resume_ids) + 1),
            f'{entity_type}_score': scores
        })
        ranking_df = ranking_df.sort_values(by=f'{entity_type}_score', ascending=False).reset_index(drop=True)
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        rankings[entity_type] = ranking_df.head(top_n).copy()

    return rankings


def calculate_overall_score(rankings: Dict) -> pd.DataFrame:
    overall_df = None
    for entity_type, ranking_df in rankings.items():
        col_name = 'experience_score' if entity_type == 'EXPERIENCE_DURATION' else f'{entity_type.lower()}_score'
        score_col = f'{entity_type}_score'
        temp_df = ranking_df[['resume_id', score_col]].copy()
        temp_df.rename(columns={score_col: col_name}, inplace=True)
        overall_df = temp_df if overall_df is None else overall_df.merge(temp_df, on='resume_id', how='outer')

    overall_df = overall_df.fillna(0.0)
    weights_dict = {
        'skill_score': WEIGHTS.get('SKILL', 0.40),
        'job_title_score': WEIGHTS.get('JOB_TITLE', 0.25),
        'degree_score': WEIGHTS.get('DEGREE', 0.20),
        'institution_score': WEIGHTS.get('INSTITUTION', 0.10),
        'certification_score': WEIGHTS.get('CERTIFICATION', 0.05),
        'experience_score': WEIGHTS.get('EXPERIENCE_DURATION', 0.10),
    }
    overall_df['overall_score'] = 0.0
    for col in [c for c in overall_df.columns if c != 'resume_id']:
        overall_df['overall_score'] += overall_df[col] * weights_dict.get(col, 0.0)

    overall_df = overall_df.sort_values(by='overall_score', ascending=False).reset_index(drop=True)
    overall_df['rank'] = range(1, len(overall_df) + 1)
    return overall_df[['rank', 'resume_id', 'overall_score', 'skill_score', 'job_title_score',
                        'degree_score', 'institution_score', 'certification_score', 'experience_score']]


def generate_improvement_recommendations(job_entities: Dict, rankings: Dict,
                                          resume_df: pd.DataFrame, top_n: int = IMPROVEMENT_TOP_N) -> pd.DataFrame:
    recommendations = []
    top_resumes = rankings['SKILL'].head(top_n)['resume_id'].tolist()
    job_skills = set(job_entities.get('SKILL', []))

    if job_skills:
        recommendations.append({
            'section': '1. REQUIRED SKILLS FOR THIS ROLE',
            'type': 'job_requirement',
            'item': ' | '.join(sorted(job_skills)),
            'priority': 'REQUIRED',
            'analysis': f"This role requires {len(job_skills)} key technical skills"
        })

        all_top_skills = []
        for resume_id in top_resumes:
            resume_row = resume_df[resume_df['resume_id'] == resume_id].iloc[0]
            skills_str = resume_row.get('SKILL', "")
            skills = set(skills_str.split('|')) if pd.notna(skills_str) else set()
            all_top_skills.extend(skills)

        top_resume_id = top_resumes[0] if top_resumes else None
        if top_resume_id:
            top_resume_row = resume_df[resume_df['resume_id'] == top_resume_id].iloc[0]
            top_resume_skills_str = top_resume_row.get('SKILL', "")
            top_resume_skills = set(top_resume_skills_str.split('|')) if pd.notna(top_resume_skills_str) else set()
            your_matching_skills = top_resume_skills & job_skills
            if your_matching_skills:
                recommendations.append({
                    'section': '2. YOUR STRENGTHS 🌟',
                    'type': 'your_skills',
                    'item': ' | '.join(sorted(your_matching_skills)),
                    'priority': 'EXCELLENT',
                    'analysis': f"You already have {len(your_matching_skills)} of the required skills! Great foundation!"
                })

        skill_counts = Counter(all_top_skills)
        missing_skills = []
        for skill, count in skill_counts.most_common():
            if skill not in job_skills:
                priority = "HIGH" if count >= top_n - 2 else "MEDIUM" if count >= top_n / 2 else "LOW"
                missing_skills.append({'skill': skill, 'frequency': count, 'priority': priority})

        if missing_skills:
            recommendations.append({
                'section': '3. SKILLS TO DEVELOP 📚',
                'type': 'missing_skills_header',
                'item': 'Focus on these technical skills to match top candidates:',
                'priority': 'ACTION_ITEMS',
                'analysis': ''
            })
            for skill_info in missing_skills[:8]:
                recommendations.append({
                    'section': '3. SKILLS TO DEVELOP 📚',
                    'type': 'missing_skill',
                    'item': skill_info['skill'],
                    'priority': skill_info['priority'],
                    'frequency': f"{skill_info['frequency']}/{top_n}",
                    'analysis': f"{skill_info['priority']}: Found in {skill_info['frequency']}/{top_n} top candidates"
                })

        high_count = len([s for s in missing_skills if s['priority'] == 'HIGH'])
        med_count = len([s for s in missing_skills if s['priority'] == 'MEDIUM'])
        if high_count > 0 or med_count > 0:
            recommendations.append({
                'section': '4. PERSONALIZED ACTION PLAN',
                'type': 'action_plan',
                'item': f"Learn {high_count} HIGH priority skills + {med_count} MEDIUM priority skills",
                'priority': 'NEXT_STEPS',
                'analysis': f"This will match {high_count + med_count} of the top candidate profiles"
            })

    return pd.DataFrame(recommendations)


def display_report(job_entities: Dict, job_experience: Optional[int], pdf_name: str,
                   recommendations_df: pd.DataFrame) -> None:
    """Display career analysis report with skill development plan"""
    
    print("\n" + "=" * 100)
    print("CAREER MATCH ANALYSIS & SKILL DEVELOPMENT PLAN")
    print("=" * 100)
    print(f"\nJob Posting : {pdf_name}")
    print(f"Analysis    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Job Requirements ──────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("JOB REQUIREMENTS")
    print("=" * 100)
    print(f"\n  Required Experience : {job_experience if job_experience else 'Not specified'} years")

    job_skills = job_entities.get('SKILL', [])
    if job_skills:
        print(f"\n  Core Skills ({len(job_skills)} total):")
        for s in job_skills:
            print(f"    • {s}")

    job_titles = job_entities.get('JOB_TITLE', [])
    if job_titles:
        print(f"\n  Job Titles:")
        for t in job_titles:
            print(f"    • {t}")

    # ── Skill Development Plan ────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("PERSONALIZED SKILL DEVELOPMENT PLAN")
    print("=" * 100)

    sections = recommendations_df.groupby('section')
    for section_name, section_data in sections:
        print(f"\n{section_name}")
        print("-" * 100)

        if section_name == '1. REQUIRED SKILLS FOR THIS ROLE':
            for _, row in section_data.iterrows():
                print(f"  {row['item']}")

        elif section_name == '2. YOUR STRENGTHS 🌟':
            for _, row in section_data.iterrows():
                skills_list = row['item'].split(' | ')
                print(f"\n  ✅ You have these skills:")
                for skill in skills_list:
                    print(f"     • {skill}")
                print(f"\n  💡 {row['analysis']}")

        elif section_name == '3. SKILLS TO DEVELOP 📚':
            for _, row in section_data.iterrows():
                if row['type'] == 'missing_skills_header':
                    print(f"\n  {row['item']}\n")
                elif row['type'] == 'missing_skill':
                    emoji = "🔴" if row['priority'] == 'HIGH' else "🟡" if row['priority'] == 'MEDIUM' else "🟢"
                    print(f"  {emoji} {row['item']} ({row['priority']})")

        elif section_name == '4. PERSONALIZED ACTION PLAN':
            for _, row in section_data.iterrows():
                print(f"\n  🎯 {row['item']}")
                if pd.notna(row.get('analysis')):
                    print(f"  {row['analysis']}")


# ============================================================================
# JOB SEARCH FUNCTIONS (Adzuna Integration)
# ============================================================================

def search_jobs_adzuna(keywords, location, country_code="in", results_per_page=30):
    """
    Search for jobs using Adzuna API (International)
    
    Args:
        keywords: List of skills to search
        location: City name
        country_code: Country code (in, us, gb, ca, au, de, fr, nl, it, es, se, nz, sg)
        results_per_page: Number of results
    """
    
    # Map country codes to country names
    country_map = {
        "in": "India",
        "us": "USA",
        "gb": "UK",
        "ca": "Canada",
        "au": "Australia",
        "de": "Germany",
        "fr": "France",
        "nl": "Netherlands",
        "it": "Italy",
        "es": "Spain",
        "se": "Sweden",
        "nz": "New Zealand",
        "sg": "Singapore",
    }
    
    country_name = country_map.get(country_code.lower(), "Unknown")
    url = f"https://api.adzuna.com/v1/api/jobs/{country_code.lower()}/search/1"
    query = " ".join(keywords)
    
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "what": query,
        "where": location,
        "results_per_page": results_per_page
    }
    
    try:
        print(f"\n🔍 Searching Adzuna for: '{query}' in {location}, {country_name}...")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'results' not in data:
            print(f"✗ No results in API response")
            return []
        
        jobs = []
        for job in data['results']:
            jobs.append({
                'job_title': job.get('title', ''),
                'company': job.get('company', {}).get('display_name', ''),
                'location': job.get('location', {}).get('display_name', ''),
                'salary_min': job.get('salary_min', None),
                'salary_max': job.get('salary_max', None),
                'salary_currency': job.get('salary_currency_code', 'USD'),
                'description': job.get('description', ''),
                'posting_date': job.get('created', ''),
                'application_url': job.get('redirect_url', '')
            })
        
        print(f"✓ Found {len(jobs)} jobs on Adzuna")
        return jobs
    
    except Exception as e:
        print(f"✗ Error searching jobs: {e}")
        return []


def score_jobs(jobs, required_skills):
    """Score each job based on skill match"""
    
    print(f"\n📊 Scoring {len(jobs)} jobs based on skill match...")
    
    for job in jobs:
        description_lower = job['description'].lower()
        match_count = 0
        
        for skill in required_skills:
            if skill.lower() in description_lower:
                match_count += 1
        
        match_score = (match_count / max(len(required_skills), 1)) * 100
        job['match_score'] = match_score
        job['matching_skills'] = match_count
    
    return jobs


def display_jobs(jobs, top_n=15):
    """Display top matching jobs"""
    
    jobs = sorted(jobs, key=lambda x: x['match_score'], reverse=True)
    
    print("\n" + "="*100)
    print("TOP JOBS FOR YOU")
    print("="*100 + "\n")
    
    for i, job in enumerate(jobs[:top_n], 1):
        print(f"{i}. {job['job_title']}")
        print(f"   Company: {job['company']}")
        print(f"   Location: {job['location']}")
        print(f"   Match: {job['match_score']:.0f}% ({job['matching_skills']} skills matched) ⭐")
        
        if job['salary_min'] and job['salary_max']:
            print(f"   Salary: ₹{job['salary_min']:,} - ₹{job['salary_max']:,} {job['salary_currency']}")
        
        print(f"   Apply: {job['application_url']}")
        print()
    
    print("="*100)
    print(f"Displayed top {min(top_n, len(jobs))} of {len(jobs)} jobs")
    print("="*100 + "\n")
    
    return jobs[:top_n]


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run(job_posting_pdf: Path, location: str, country_code: str = "in") -> None:
    """Complete pipeline: analyze job + find matching jobs (international)"""
    
    print("\n" + "=" * 80)
    print("COMPLETE JOB ANALYZER")
    print("=" * 80 + "\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ════════════════════════════════════════════════════════════════════════
    # PART 1: ANALYZE JOB POSTING
    # ════════════════════════════════════════════════════════════════════════
    
    print("[PART 1/2] ANALYZING JOB POSTING\n")

    # Step 1: Load prerequisites
    print("[1/4] Loading prerequisites...")
    nlp = load_spacy_model(NER_MODEL_PATH)
    try:
        resume_df = pd.read_csv(RESUMES_WITH_ENTITIES)
        print(f"✓ Loaded {len(resume_df)} resumes")
    except Exception as e:
        print(f"✗ Failed to load resumes: {e}")
        sys.exit(1)
    resume_ids = resume_df['resume_id'].values

    # Step 2: Extract PDF text
    print("\n[2/4] Extracting text from job posting PDF...")
    job_text = extract_pdf_text(job_posting_pdf)
    print(f"  Extracted {len(job_text)} characters")

    # Step 3: Extract entities
    print("\n[3/4] Extracting entities from job posting...")
    job_entities = extract_entities_ner(job_text, nlp)
    for etype, ents in job_entities.items():
        print(f"  {etype}: {len(ents)} entities")
        if ents[:3]:
            print(f"    Examples: {', '.join(ents[:3])}")

    # Step 4: Extract experience
    print("\n[4/4] Extracting experience duration...")
    job_experience = extract_experience_duration(job_text, nlp)
    print(f"  Required: {job_experience} years" if job_experience else "  No experience requirement found")

    # Load vectorizers and vectors for Phase 5
    print("\n[Loading vectorizers and resume vectors...]")
    vectorizers = load_phase4_vectorizers(DATA_PROCESSED)
    resume_vectors = load_resume_vectors(DATA_PROCESSED)

    # Vectorize job entities
    job_vectors = vectorize_entities(job_entities, vectorizers)

    # Calculate similarities
    similarities = calculate_similarity_scores(job_vectors, resume_vectors)

    # Rank resumes
    rankings = rank_resumes_by_category(similarities, resume_ids, job_experience, resume_df, top_n=TOP_N)

    # Generate recommendations
    recommendations_df = generate_improvement_recommendations(job_entities, rankings, resume_df, top_n=IMPROVEMENT_TOP_N)

    # Display job requirements + skill development plan
    display_report(job_entities, job_experience, job_posting_pdf.name, recommendations_df)

    # ════════════════════════════════════════════════════════════════════════
    # PART 2: SEARCH FOR MATCHING JOBS
    # ════════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 100)
    print("[PART 2/2] SEARCHING FOR MATCHING JOBS\n")

    # Extract skills from job posting
    required_skills = job_entities.get('SKILL', [])
    
    if not required_skills:
        print("⚠️  No skills found in job posting. Skipping job search.")
        return
    
    # Search Adzuna (with country code)
    jobs = search_jobs_adzuna(required_skills, location, country_code=country_code, results_per_page=30)
    
    if not jobs:
        print("✗ No jobs found")
        return
    
    # Score jobs
    jobs = score_jobs(jobs, required_skills)
    
    # Display top jobs
    top_jobs = display_jobs(jobs, top_n=15)
    
    print("="*100)
    print("ANALYSIS COMPLETE ✓")
    print("="*100 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python complete_analyzer.py <job_posting.pdf> <location> <country_code>")
        print(f"\nExamples:")
        print(f"  python complete_analyzer.py \"data\\input\\job.pdf\" \"Bangalore\" \"in\"    # India")
        print(f"  python complete_analyzer.py \"data\\input\\job.pdf\" \"New York\" \"us\"   # USA")
        print(f"  python complete_analyzer.py \"data\\input\\job.pdf\" \"London\" \"gb\"     # UK")
        print(f"  python complete_analyzer.py \"data\\input\\job.pdf\" \"Toronto\" \"ca\"    # Canada")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    location = sys.argv[2] if len(sys.argv) > 2 else "Bangalore"
    country_code = sys.argv[3] if len(sys.argv) > 3 else "in"
    
    if not pdf_path.exists():
        print(f"✗ PDF not found: {pdf_path}")
        sys.exit(1)
    
    run(pdf_path, location, country_code)