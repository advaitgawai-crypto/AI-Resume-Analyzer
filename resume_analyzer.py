#!/usr/bin/env python3
"""
================================================================================
AI RESUME ANALYZER - Single File Runner
================================================================================
Usage:
  python resume_analyzer.py <path_to_pdf>

Example:
  python resume_analyzer.py "data\input\senior_python_engineer.pdf"
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



def load_spacy_model(model_path: Path) -> spacy.Language:
    try:
        nlp = spacy.load(str(model_path))
        print(f"✓ Loaded NER model from {model_path}")
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
            print(f"  Regex found: {years} years experience")
            return years

    doc = nlp(text)
    experience_markers = ['experience', 'years', 'yrs', 'worked', 'employed', 'senior', 'junior', 'lead']
    for ent in doc.ents:
        if ent.label_ in ['DATE', 'CARDINAL']:
            context = text[max(0, ent.start_char - 50):ent.end_char + 50].lower()
            if any(marker in context for marker in experience_markers):
                try:
                    num_match = re.search(r'\d+', ent.text)
                    if num_match:
                        years = int(num_match.group())
                        if 0 < years < 100:
                            print(f"  NLP fallback found: {years} years experience")
                            return years
                except:
                    pass
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
                print(f"  Warning: Failed to vectorize {entity_type}: {e}")
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
                print(f"✓ Loaded vectorizer for {entity_type}")
            except Exception as e:
                print(f"✗ Failed to load vectorizer for {entity_type}: {e}")
        else:
            print(f"⚠ Vectorizer not found: {vectorizer_path}")
    return vectorizers


def load_resume_vectors(data_dir: Path) -> Dict:
    resume_vectors = {}
    for entity_type in ENTITY_TYPES[:-1]:
        vector_path = data_dir / f"resume_vectors_{entity_type.lower()}.npz"
        if vector_path.exists():
            try:
                resume_vectors[entity_type] = load_npz(str(vector_path))
                print(f"✓ Loaded resume vectors for {entity_type}")
            except Exception as e:
                print(f"✗ Failed to load resume vectors for {entity_type}: {e}")
        else:
            print(f"⚠ Resume vectors not found: {vector_path}")
    return resume_vectors


def calculate_similarity_scores(job_vectors: Dict, resume_vectors: Dict) -> Dict:
    similarities = {}
    for entity_type in ENTITY_TYPES[:-1]:
        if job_vectors.get(entity_type) is not None and entity_type in resume_vectors:
            try:
                sim = cosine_similarity(job_vectors[entity_type], resume_vectors[entity_type])[0]
                similarities[entity_type] = sim
            except Exception as e:
                print(f"  Warning: Failed to compute similarity for {entity_type}: {e}")
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
                print(f"  Warning: Failed to compute experience scores: {e}")
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
                'section': '3. SKILLS TO DEVELOP ',
                'type': 'missing_skills_header',
                'item': 'Focus on these technical skills to match top candidates:',
                'priority': 'ACTION_ITEMS',
                'analysis': ''
            })
            for skill_info in missing_skills[:8]:
                recommendations.append({
                    'section': '3. SKILLS TO DEVELOP ',
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


# ============================================================================
# DISPLAY REPORT (from view_phase5_results.py, minus TOP 10 CANDIDATES)
# ============================================================================

def display_report(overall_df: pd.DataFrame, skill_df: pd.DataFrame,
                   recommendations_df: pd.DataFrame, job_entities: Dict,
                   job_experience: Optional[int], pdf_name: str) -> None:

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
                print(f"\n   {row['item']}")
                print(f"   {row['analysis']}")

        elif section_name == '3. SKILLS TO DEVELOP ':
            for _, row in section_data.iterrows():
                if row['type'] == 'missing_skills_header':
                    print(f"\n  {row['item']}\n")
                elif row['type'] == 'missing_skill':
                    emoji = "🔴" if row['priority'] == 'HIGH' else "🟡" if row['priority'] == 'MEDIUM' else "🟢"
                    print(f"  {emoji} {row['item']}")
                    if pd.notna(row.get('frequency')):
                        print(f"     Found in {row.get('frequency')} top candidates")
                    if pd.notna(row.get('analysis')):
                        print(f"     {row['analysis']}\n")

        elif section_name == '4. PERSONALIZED ACTION PLAN':
            for _, row in section_data.iterrows():
                print(f"\n  🎯 {row['item']}")
                if pd.notna(row.get('analysis')):
                    print(f"  {row['analysis']}")

    print("\n" + "=" * 100)
    print("END OF REPORT")
    print("=" * 100 + "\n")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run(job_posting_pdf: Path) -> None:
    print("\n" + "=" * 80)
    print("AI RESUME ANALYZER")
    print("=" * 80 + "\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 1: Load prerequisites
    print("[1/9] Loading prerequisites...")
    nlp = load_spacy_model(NER_MODEL_PATH)
    try:
        resume_df = pd.read_csv(RESUMES_WITH_ENTITIES)
        print(f"✓ Loaded {len(resume_df)} resumes")
    except Exception as e:
        print(f"✗ Failed to load resumes: {e}")
        sys.exit(1)
    resume_ids = resume_df['resume_id'].values

    # Step 2: Extract PDF text
    print("\n[2/9] Extracting text from PDF...")
    job_text = extract_pdf_text(job_posting_pdf)
    print(f"  Extracted {len(job_text)} characters")

    # Step 3: Extract entities
    print("\n[3/9] Extracting entities...")
    job_entities = extract_entities_ner(job_text, nlp)
    for etype, ents in job_entities.items():
        print(f"  {etype}: {len(ents)} entities")
        if ents[:3]:
            print(f"    Examples: {', '.join(ents[:3])}")

    # Step 4: Extract experience
    print("\n[4/9] Extracting experience duration...")
    job_experience = extract_experience_duration(job_text, nlp)
    print(f"  Required: {job_experience} years" if job_experience else "  No experience requirement found")

    # Step 5: Load vectorizers & vectors
    print("\n[5/9] Loading vectorizers and resume vectors...")
    vectorizers = load_phase4_vectorizers(DATA_PROCESSED)
    resume_vectors = load_resume_vectors(DATA_PROCESSED)

    # Step 6: Vectorize job entities
    print("\n[6/9] Vectorizing job posting entities...")
    job_vectors = vectorize_entities(job_entities, vectorizers)
    for etype, vec in job_vectors.items():
        if vec is not None:
            print(f"  ✓ {etype}: vectorized ({vec.shape[1]} dimensions)")

    # Step 7: Calculate similarities
    print("\n[7/9] Calculating similarity scores...")
    similarities = calculate_similarity_scores(job_vectors, resume_vectors)

    # Step 8: Rank resumes
    print("\n[8/9] Ranking resumes...")
    rankings = rank_resumes_by_category(similarities, resume_ids, job_experience, resume_df, top_n=TOP_N)
    for etype, df in rankings.items():
        print(f"  ✓ {etype}: top {len(df)} ranked")

    # Step 9: Generate outputs
    print("\n[9/9] Generating outputs...")
    for etype, df in rankings.items():
        out = OUTPUT_DIR / f"{etype.lower()}_rankings_{timestamp}.csv"
        df.to_csv(out, index=False)

    overall_df = calculate_overall_score(rankings)
    overall_df.to_csv(OUTPUT_DIR / f"overall_rankings_{timestamp}.csv", index=False)

    recommendations_df = generate_improvement_recommendations(job_entities, rankings, resume_df, top_n=IMPROVEMENT_TOP_N)
    recommendations_df.to_csv(OUTPUT_DIR / f"improvement_recommendations_{timestamp}.csv", index=False)

    print(f"  ✓ CSVs saved to: {OUTPUT_DIR}")

    # Display report
    display_report(overall_df, rankings['SKILL'], recommendations_df,
                   job_entities, job_experience, job_posting_pdf.name)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        pdf_path = PROJECT_ROOT / "data" / "input" / "job_posting.pdf"

    if not pdf_path.exists():
        print(f"✗ PDF not found: {pdf_path}")
        print("\nUsage: python resume_analyzer.py <path_to_pdf>")
        sys.exit(1)

    run(pdf_path)
