#!/usr/bin/env python3
"""
================================================================================
PHASE 5: MATCHING ENGINE & RESUME ADVISOR
================================================================================

Purpose:
  - Parse job posting PDF
  - Extract entities (SKILL, JOB_TITLE, DEGREE, INSTITUTION, CERTIFICATION)
  - Extract EXPERIENCE_DURATION (regex + NLP fallback)
  - Rank all 2,716 resumes per category
  - Generate 8 output CSVs with improvement recommendations

Input:  job_posting.pdf
Output: 8 CSVs (7 categories + 1 overall)
        + improvement_recommendations.csv (skill gaps, suggestions)

Time:   ~2–3 seconds per job posting
Author: ThunderLord AI Resume Analyzer
Date:   July 15, 2026
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
from collections import Counter
from typing import Dict, List, Tuple, Optional


# ============================================================================
# CONFIGURATION
# ============================================================================

# Project paths (adjust based on your system)
PROJECT_ROOT = Path(r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer")
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "phase5"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Phase 4 files (inputs)
RESUMES_WITH_ENTITIES = DATA_PROCESSED / "resumes_with_entities.csv"
SIMILARITY_MATRIX_FILE = DATA_PROCESSED / "similarity_matrix.npz"
CATEGORY_SCORES_FILE = DATA_PROCESSED / "category_scores.csv"
NER_MODEL_PATH = MODELS_DIR / "ner_model_v2"

# Weighting for overall score (from Phase 4)
WEIGHTS = {
    'SKILL': 0.40,
    'JOB_TITLE': 0.25,
    'DEGREE': 0.20,
    'INSTITUTION': 0.10,
    'CERTIFICATION': 0.05,
    'EXPERIENCE_DURATION': 0.00  # Placeholder, we'll add it dynamically
}

# After adding EXPERIENCE_DURATION, normalize weights
WEIGHTS['EXPERIENCE_DURATION'] = 0.10
WEIGHTS_SUM = sum(WEIGHTS.values())
WEIGHTS = {k: v / WEIGHTS_SUM for k, v in WEIGHTS.items()}

# Output configuration
TOP_N = 50  # Top 50 resumes per category
IMPROVEMENT_TOP_N = 10  # Top 10 resumes for improvement suggestions

ENTITY_TYPES = ['SKILL', 'JOB_TITLE', 'DEGREE', 'INSTITUTION', 'CERTIFICATION', 'EXPERIENCE_DURATION']

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_spacy_model(model_path: Path) -> spacy.Language:
    """Load trained spaCy NER model."""
    try:
        nlp = spacy.load(str(model_path))
        print(f"✓ Loaded NER model from {model_path}")
        return nlp
    except Exception as e:
        print(f"✗ Failed to load NER model: {e}")
        sys.exit(1)


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF using pdfplumber."""
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
    """Extract entities using trained NER model."""
    doc = nlp(text)
    entities = {ent_type: [] for ent_type in ENTITY_TYPES[:-1]}  # Exclude EXPERIENCE_DURATION
    
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    
    # Remove duplicates, maintain order
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))
    
    return entities


def extract_experience_duration(text: str, nlp: spacy.Language) -> Optional[int]:
    """
    Extract experience duration in years from text.
    Try regex first, then NLP fallback.
    
    Returns:
      - Extracted years as int (e.g., 5, 10, 15)
      - None if not found
    """
    # Regex patterns (attempt 1: simple extraction)
    patterns = [
        r'(\d+)\+?\s+(?:years?|yrs?)',  # "5+ years", "10 years", "3 yrs"
        r'(\d+)\s*-\s*(\d+)\s+(?:years?|yrs?)',  # "5-10 years" → extract max
        r'(\d+)\s+(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)',  # "5 years of experience"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if len(match.groups()) == 2:  # Range like "5-10 years"
                years = int(match.group(2))  # Take max
            else:
                years = int(match.group(1))
            print(f"  Regex found: {years} years experience")
            return years
    
    # NLP fallback: look for DATE entities + context
    doc = nlp(text)
    experience_markers = ['experience', 'years', 'yrs', 'worked', 'employed', 'senior', 'junior', 'lead']
    
    for ent in doc.ents:
        if ent.label_ in ['DATE', 'CARDINAL']:  # DATE or numeric values
            # Check context: is there an experience-related word nearby?
            context = text[max(0, ent.start_char - 50):ent.end_char + 50].lower()
            if any(marker in context for marker in experience_markers):
                try:
                    # Try to extract numeric value from DATE/CARDINAL
                    num_match = re.search(r'\d+', ent.text)
                    if num_match:
                        years = int(num_match.group())
                        if 0 < years < 100:  # Sanity check
                            print(f"  NLP fallback found: {years} years experience")
                            return years
                except:
                    pass
    
    return None


def vectorize_entities(entities: Dict[str, List[str]], vectorizers: Dict) -> Dict[str, np.ndarray]:
    """Vectorize job posting entities using pre-fitted vectorizers from Phase 4."""
    job_vectors = {}
    
    for entity_type, entity_list in entities.items():
        if entity_type in vectorizers:
            # Join entities into single string (same as training)
            entity_text = " ".join(entity_list) if entity_list else ""
            
            try:
                # Transform using pre-fitted vectorizer
                vector = vectorizers[entity_type].transform([entity_text])
                job_vectors[entity_type] = vector
            except Exception as e:
                print(f"  Warning: Failed to vectorize {entity_type}: {e}")
                job_vectors[entity_type] = None
    
    return job_vectors


def load_phase4_vectorizers(data_dir: Path) -> Dict:
    """Load TF-IDF vectorizers from Phase 4 (pickled)."""
    vectorizers = {}
    
    for entity_type in ENTITY_TYPES[:-1]:  # Exclude EXPERIENCE_DURATION (no vectorizer)
        vectorizer_path = data_dir / f"vectorizer_{entity_type.lower()}.pkl"
        
        if vectorizer_path.exists():
            try:
                with open(vectorizer_path, 'rb') as f:
                    vectorizers[entity_type] = pickle.load(f)
                print(f"✓ Loaded vectorizer for {entity_type}")
            except Exception as e:
                print(f"✗ Failed to load vectorizer for {entity_type}: {e}")
        else:
            print(f"⚠ Vectorizer not found for {entity_type}: {vectorizer_path}")
    
    return vectorizers


def load_resume_vectors(data_dir: Path) -> Dict[str, np.ndarray]:
    """Load pre-computed resume vectors from Phase 4."""
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
            print(f"⚠ Resume vectors not found for {entity_type}: {vector_path}")
    
    return resume_vectors


def calculate_similarity_scores(
    job_vectors: Dict[str, np.ndarray],
    resume_vectors: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Calculate cosine similarity between job posting and all resumes per category."""
    similarities = {}
    
    for entity_type in ENTITY_TYPES[:-1]:
        if job_vectors[entity_type] is not None and entity_type in resume_vectors:
            try:
                # Cosine similarity: job_vector vs all resume vectors
                sim = cosine_similarity(job_vectors[entity_type], resume_vectors[entity_type])[0]
                similarities[entity_type] = sim
            except Exception as e:
                print(f"  Warning: Failed to compute similarity for {entity_type}: {e}")
                similarities[entity_type] = np.zeros(len(resume_vectors[entity_type]))
        else:
            print(f"  Skipping {entity_type} (missing vectors)")
            similarities[entity_type] = None
    
    return similarities


def handle_experience_distance(job_experience: Optional[int], resume_experience: Optional[int]) -> float:
    """
    Calculate similarity between job posting experience and resume experience.
    
    Logic:
      - Perfect match (within 1 year): 1.0
      - Overqualified (resume > job): 0.9
      - Underqualified (resume < job): 0.3–0.7 (scales down)
      - No experience data: 0.5 (neutral)
    """
    if job_experience is None or resume_experience is None:
        return 0.5  # Neutral if missing
    
    diff = abs(job_experience - resume_experience)
    
    if diff <= 1:
        return 1.0  # Perfect match
    elif resume_experience > job_experience:
        return 0.9  # Overqualified (good)
    else:
        # Underqualified: penalize by (missing_years / job_experience)
        missing_years = job_experience - resume_experience
        penalty = min(missing_years / max(job_experience, 1), 0.7)
        return max(0.3, 1.0 - penalty)


def rank_resumes_by_category(
    similarities: Dict[str, np.ndarray],
    resume_ids: np.ndarray,
    job_experience: Optional[int],
    resume_df: pd.DataFrame,
    top_n: int = TOP_N
) -> Dict[str, pd.DataFrame]:
    """Rank all resumes per category, return top_n for each."""
    rankings = {}
    
    for entity_type in ENTITY_TYPES:
        if entity_type == 'EXPERIENCE_DURATION':
            # Special handling for experience: parse from resumes
            try:
                exp_scores = []
                for _, row in resume_df.iterrows():
                    resume_exp = None
                    if pd.notna(row['EXPERIENCE_DURATION']) and row['EXPERIENCE_DURATION'].strip():
                        # Try to extract years from pipe-delimited string
                        exp_entries = row['EXPERIENCE_DURATION'].split('|')
                        if exp_entries:
                            try:
                                resume_exp = int(re.search(r'\d+', exp_entries[0]).group())
                            except:
                                pass
                    
                    score = handle_experience_distance(job_experience, resume_exp)
                    exp_scores.append(score)
                
                scores = np.array(exp_scores)
            except Exception as e:
                print(f"  Warning: Failed to compute experience scores: {e}")
                scores = np.zeros(len(resume_df))
        
        elif similarities[entity_type] is not None:
            scores = similarities[entity_type]
        else:
            scores = np.zeros(len(resume_df))
        
        # Create ranking dataframe
        ranking_df = pd.DataFrame({
            'resume_id': resume_ids,
            'rank': range(1, len(resume_ids) + 1),
            f'{entity_type}_score': scores
        })
        
        # Sort by score descending
        ranking_df = ranking_df.sort_values(by=f'{entity_type}_score', ascending=False).reset_index(drop=True)
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        
        # Keep top_n
        rankings[entity_type] = ranking_df.head(top_n).copy()
    
    return rankings


def calculate_overall_score(rankings: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate weighted overall score from all categories."""
    # Merge all ranking dataframes
    overall_df = None
    
    for entity_type, ranking_df in rankings.items():
        if entity_type == 'EXPERIENCE_DURATION':
            col_name = 'experience_score'
        else:
            col_name = f'{entity_type.lower()}_score'
        
        # Get the score column (it's named {entity_type}_score in the ranking_df)
        score_col = f'{entity_type}_score'
        temp_df = ranking_df[['resume_id', score_col]].copy()
        temp_df.rename(columns={score_col: col_name}, inplace=True)
        
        if overall_df is None:
            overall_df = temp_df
        else:
            overall_df = overall_df.merge(temp_df, on='resume_id', how='outer')
    
    # Fill NaNs with 0 (resume not in top-50 for that category)
    overall_df = overall_df.fillna(0.0)
    
    # Calculate weighted overall score
    score_cols = [col for col in overall_df.columns if col != 'resume_id']
    weights_dict = {
        'skill_score': WEIGHTS.get('SKILL', 0.40),
        'job_title_score': WEIGHTS.get('JOB_TITLE', 0.25),
        'degree_score': WEIGHTS.get('DEGREE', 0.20),
        'institution_score': WEIGHTS.get('INSTITUTION', 0.10),
        'certification_score': WEIGHTS.get('CERTIFICATION', 0.05),
        'experience_score': WEIGHTS.get('EXPERIENCE_DURATION', 0.10),
    }
    
    overall_df['overall_score'] = 0.0
    for col in score_cols:
        weight = weights_dict.get(col, 0.0)
        overall_df['overall_score'] += overall_df[col] * weight
    
    # Rank by overall score
    overall_df = overall_df.sort_values(by='overall_score', ascending=False).reset_index(drop=True)
    overall_df['rank'] = range(1, len(overall_df) + 1)
    
    return overall_df[['rank', 'resume_id', 'overall_score', 'skill_score', 'job_title_score', 
                        'degree_score', 'institution_score', 'certification_score', 'experience_score']]


def generate_improvement_recommendations(
    job_entities: Dict[str, List[str]],
    rankings: Dict[str, pd.DataFrame],
    resume_df: pd.DataFrame,
    top_n: int = IMPROVEMENT_TOP_N
) -> pd.DataFrame:
    """
    Generate improvement recommendations for the job seeker.
    
    IMPROVED VERSION:
    - Only recommends SKILLS (not degrees, job titles, certifications)
    - Shows what skills the job posting requires
    - Shows what skills top candidates have
    - Identifies missing skills with priority levels
    - Provides praise for strong skills
    """
    recommendations = []
    
    # Get top-10 resumes (from SKILL rankings, as most comprehensive)
    top_resumes = rankings['SKILL'].head(top_n)['resume_id'].tolist()
    
    # ========================================================================
    # SECTION 1: SKILLS ANALYSIS (Only SKILLS, not degrees/titles)
    # ========================================================================
    
    job_skills = set(job_entities.get('SKILL', []))
    
    if job_skills:
        # PRAISE: Show what skills the job posting requires
        recommendations.append({
            'section': '1. REQUIRED SKILLS FOR THIS ROLE',
            'type': 'job_requirement',
            'item': ' | '.join(sorted(job_skills)),
            'priority': 'REQUIRED',
            'analysis': f"This role requires {len(job_skills)} key technical skills"
        })
        
        # Get all skills from top resumes
        all_top_skills = []
        for resume_id in top_resumes:
            resume_row = resume_df[resume_df['resume_id'] == resume_id].iloc[0]
            skills_str = resume_row.get('SKILL', "")
            skills = set(skills_str.split('|')) if pd.notna(skills_str) else set()
            all_top_skills.extend(skills)
        
        # ====================================================================
        # SECTION 2: WHAT YOU HAVE (Praise!)
        # ====================================================================
        # Assuming the top-ranked resume is closest to what we're looking for
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
        
        # ====================================================================
        # SECTION 3: MISSING SKILLS (With Priority)
        # ====================================================================
        all_top_skills_list = list(all_top_skills)
        skill_counts = Counter(all_top_skills_list)
        
        # Find skills required but not in your profile (approximated from top resumes)
        # We'll use frequency to identify critical missing skills
        missing_skills = []
        for skill, count in skill_counts.most_common():
            if skill not in job_skills:  # Only if not already mentioned in job posting
                priority = "HIGH" if count >= top_n - 2 else "MEDIUM" if count >= top_n / 2 else "LOW"
                missing_skills.append({
                    'skill': skill,
                    'frequency': count,
                    'priority': priority
                })
        
        # Add recommendations for HIGH and MEDIUM priority skills only
        if missing_skills:
            recommendations.append({
                'section': '3. SKILLS TO DEVELOP 📚',
                'type': 'missing_skills_header',
                'item': 'Focus on these technical skills to match top candidates:',
                'priority': 'ACTION_ITEMS',
                'analysis': ''
            })
            
            for skill_info in missing_skills[:8]:  # Top 8 missing skills
                recommendations.append({
                    'section': '3. SKILLS TO DEVELOP 📚',
                    'type': 'missing_skill',
                    'item': skill_info['skill'],
                    'priority': skill_info['priority'],
                    'frequency': f"{skill_info['frequency']}/{top_n}",
                    'analysis': f"{skill_info['priority']}: Found in {skill_info['frequency']}/{top_n} top candidates"
                })
        
        # ====================================================================
        # SECTION 4: ACTION PLAN
        # ====================================================================
        high_priority_count = len([s for s in missing_skills if s['priority'] == 'HIGH'])
        medium_priority_count = len([s for s in missing_skills if s['priority'] == 'MEDIUM'])
        
        if high_priority_count > 0 or medium_priority_count > 0:
            recommendations.append({
                'section': '4. PERSONALIZED ACTION PLAN',
                'type': 'action_plan',
                'item': f"Learn {high_priority_count} HIGH priority skills + {medium_priority_count} MEDIUM priority skills",
                'priority': 'NEXT_STEPS',
                'analysis': f"This will match {high_priority_count + medium_priority_count} of the top candidate profiles"
            })
    
    return pd.DataFrame(recommendations)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_phase5(job_posting_pdf: Path) -> None:
    """
    Main Phase 5 pipeline.
    
    Args:
      job_posting_pdf: Path to job posting PDF file
    """
    print("\n" + "="*80)
    print("PHASE 5: MATCHING ENGINE & RESUME ADVISOR")
    print("="*80 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ========================================================================
    # STEP 1: Load prerequisites
    # ========================================================================
    print("[1/9] Loading prerequisites...")
    
    # Load NER model
    nlp = load_spacy_model(NER_MODEL_PATH)
    
    # Load resumes with entities
    try:
        resume_df = pd.read_csv(RESUMES_WITH_ENTITIES)
        print(f"✓ Loaded {len(resume_df)} resumes with entities")
    except Exception as e:
        print(f"✗ Failed to load resumes: {e}")
        sys.exit(1)
    
    resume_ids = resume_df['resume_id'].values
    
    # ========================================================================
    # STEP 2: Extract text from job posting PDF
    # ========================================================================
    print("\n[2/9] Extracting text from job posting PDF...")
    job_text = extract_pdf_text(job_posting_pdf)
    print(f"  Extracted {len(job_text)} characters")
    
    # ========================================================================
    # STEP 3: Extract entities from job posting (NER)
    # ========================================================================
    print("\n[3/9] Extracting entities from job posting...")
    job_entities = extract_entities_ner(job_text, nlp)
    for entity_type, entities in job_entities.items():
        print(f"  {entity_type}: {len(entities)} entities")
        if entities[:3]:
            print(f"    Examples: {', '.join(entities[:3])}")
    
    # ========================================================================
    # STEP 4: Extract experience duration
    # ========================================================================
    print("\n[4/9] Extracting experience duration...")
    job_experience = extract_experience_duration(job_text, nlp)
    print(f"  Job posting requires: {job_experience} years" if job_experience else "  No experience requirement found")
    
    # ========================================================================
    # STEP 5: Load Phase 4 vectorizers and resume vectors
    # ========================================================================
    print("\n[5/9] Loading Phase 4 vectorizers and resume vectors...")
    vectorizers = load_phase4_vectorizers(DATA_PROCESSED)
    resume_vectors = load_resume_vectors(DATA_PROCESSED)
    
    # ========================================================================
    # STEP 6: Vectorize job posting entities
    # ========================================================================
    print("\n[6/9] Vectorizing job posting entities...")
    job_vectors = vectorize_entities(job_entities, vectorizers)
    for entity_type, vector in job_vectors.items():
        if vector is not None:
            print(f"  ✓ {entity_type}: vectorized ({vector.shape[1]} dimensions)")
    
    # ========================================================================
    # STEP 7: Calculate similarity scores
    # ========================================================================
    print("\n[7/9] Calculating similarity scores...")
    similarities = calculate_similarity_scores(job_vectors, resume_vectors)
    
    # ========================================================================
    # STEP 8: Rank resumes per category
    # ========================================================================
    print("\n[8/9] Ranking resumes per category...")
    rankings = rank_resumes_by_category(similarities, resume_ids, job_experience, resume_df, top_n=TOP_N)
    
    for entity_type, ranking_df in rankings.items():
        print(f"  ✓ {entity_type}: top {len(ranking_df)} resumes ranked")
    
    # ========================================================================
    # STEP 9: Output results and recommendations
    # ========================================================================
    print("\n[9/9] Generating outputs...")
    
    # Output individual category rankings
    for entity_type, ranking_df in rankings.items():
        output_file = OUTPUT_DIR / f"{entity_type.lower()}_rankings_{timestamp}.csv"
        ranking_df.to_csv(output_file, index=False)
        print(f"  ✓ {entity_type}: {output_file.name}")
    
    # Calculate and output overall ranking
    overall_df = calculate_overall_score(rankings)
    overall_file = OUTPUT_DIR / f"overall_rankings_{timestamp}.csv"
    overall_df.to_csv(overall_file, index=False)
    print(f"  ✓ Overall: {overall_file.name}")
    
    # Generate improvement recommendations
    recommendations_df = generate_improvement_recommendations(job_entities, rankings, resume_df, top_n=IMPROVEMENT_TOP_N)
    recommendations_file = OUTPUT_DIR / f"improvement_recommendations_{timestamp}.csv"
    recommendations_df.to_csv(recommendations_file, index=False)
    print(f"  ✓ Recommendations: {recommendations_file.name}")
    
    # Summary report (user-friendly version)
    summary_file = OUTPUT_DIR / f"phase5_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CAREER MATCH ANALYSIS & SKILL DEVELOPMENT PLAN\n")
        f.write("="*80 + "\n\n")
        f.write(f"Job Posting: {job_posting_pdf.name}\n")
        f.write(f"Analysis Date: {timestamp}\n\n")
        
        # ====================================================================
        f.write("JOB REQUIREMENTS ANALYSIS\n")
        f.write("-"*80 + "\n")
        f.write(f"Required Experience: {job_experience} years\n\n")
        
        job_skills = job_entities.get('SKILL', [])
        if job_skills:
            f.write(f"Core Skills Required ({len(job_skills)} total):\n")
            for skill in job_skills:
                f.write(f"  • {skill}\n")
        
        job_titles = job_entities.get('JOB_TITLE', [])
        if job_titles:
            f.write(f"\nDesired Job Titles:\n")
            for title in job_titles:
                f.write(f"  • {title}\n")
        
        f.write("\n")
        
        # ====================================================================
        f.write("="*80 + "\n")
        f.write("TOP 10 MATCHED CANDIDATES\n")
        f.write("="*80 + "\n\n")
        f.write("These candidates are the closest matches for this role:\n\n")
        for idx, row in overall_df.head(10).iterrows():
            f.write(f"  {int(row['rank'])}. Resume {int(row['resume_id'])}\n")
            f.write(f"     Overall Match: {row['overall_score']:.1%}\n")
            f.write(f"     Skill Match: {row['skill_score']:.1%}\n\n")
        
        # ====================================================================
        f.write("="*80 + "\n")
        f.write("YOUR PERSONALIZED SKILL DEVELOPMENT PLAN\n")
        f.write("="*80 + "\n\n")
        
        # Show sections from recommendations
        sections = recommendations_df.groupby('section')
        for section_name, section_data in sections:
            # Remove emojis for file writing (encoding issues)
            clean_section_name = section_name.replace('🌟', '').replace('📚', '').strip()
            f.write(f"\n{clean_section_name}\n")
            f.write("-"*80 + "\n")
            
            if section_name == '1. REQUIRED SKILLS FOR THIS ROLE':
                for _, row in section_data.iterrows():
                    f.write(f"  {row['item']}\n")
            
            elif '2. YOUR STRENGTHS' in section_name:
                for _, row in section_data.iterrows():
                    f.write(f"  [STRENGTH] {row['item']}\n")
                    f.write(f"  {row['analysis']}\n")
            
            elif '3. SKILLS TO DEVELOP' in section_name:
                for _, row in section_data.iterrows():
                    if row['type'] == 'missing_skills_header':
                        f.write(f"\n  {row['item']}\n\n")
                    elif row['type'] == 'missing_skill':
                        priority_symbol = "[HIGH]" if row['priority'] == 'HIGH' else "[MEDIUM]" if row['priority'] == 'MEDIUM' else "[LOW]"
                        f.write(f"  {priority_symbol} {row['item']}\n")
                        f.write(f"     Priority: {row['priority']} (found in {row.get('frequency', 'N/A')} top candidates)\n")
            
            elif '4. PERSONALIZED ACTION PLAN' in section_name:
                for _, row in section_data.iterrows():
                    f.write(f"  {row['item']}\n")
                    f.write(f"  {row['analysis']}\n")
            
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"  ✓ Summary: {summary_file.name}")
    
    print("\n" + "="*80)
    print(f"PHASE 5 COMPLETE ✓")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("="*80 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Example usage (you can also run from command line):
    # python phase5_matching_engine.py path/to/job_posting.pdf
    
    if len(sys.argv) > 1:
        job_posting_path = Path(sys.argv[1])
    else:
        # Default test path
        job_posting_path = PROJECT_ROOT / "data" / "input" / "job_posting.pdf"
    
    if not job_posting_path.exists():
        print(f"✗ Job posting PDF not found: {job_posting_path}")
        print("\nUsage: python phase5_matching_engine.py <path_to_job_posting.pdf>")
        sys.exit(1)
    
    run_phase5(job_posting_path)