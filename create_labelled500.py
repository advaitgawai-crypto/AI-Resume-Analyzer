"""
Phase 2.5: Create labelled500.csv
Simple approach — load resumes_unified.csv, sample 500, auto-extract entities
"""

import pandas as pd
import re
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Update this path to your actual location
RESUMES_UNIFIED = r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer\data\processed\resumes_unified.csv"
OUTPUT_CSV = r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer\data\processed\labelled500.csv"

# ============================================================================
# KEYWORD LISTS
# ============================================================================

TECH_SKILLS = {
    'python', 'java', 'c++', 'c#', 'javascript', 'typescript', 'go', 'rust', 
    'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css',
    'django', 'flask', 'fastapi', 'react', 'vue', 'angular', 'next.js', 'svelte',
    'spring', 'springboot', 'hibernate', 'express', 'node.js', 'nodejs', 'laravel',
    'pytorch', 'tensorflow', 'keras', 'scikit-learn', 'pandas', 'numpy', 'scipy',
    'matplotlib', 'seaborn', 'plotly', 'bokeh', 'dash',
    'mysql', 'postgresql', 'mongodb', 'cassandra', 'redis', 'elasticsearch',
    'dynamodb', 'firestore', 'oracle', 'sqlite', 'mariadb', 'neo4j',
    'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'jenkins',
    'gitlab', 'github', 'bitbucket', 'terraform', 'ansible', 'vagrant',
    'cloudformation', 'openstack', 'heroku',
    'hadoop', 'spark', 'hive', 'pig', 'kafka', 'airflow', 'etl', 'bigquery',
    'snowflake', 'redshift', 'dbt', 'databricks',
    'machine learning', 'deep learning', 'nlp', 'computer vision', 'cv',
    'bert', 'gpt', 'transformers', 'lstm', 'cnn', 'rnn', 'gan',
    'git', 'jira', 'confluence', 'slack', 'linux', 'unix', 'windows',
    'macos', 'rest api', 'graphql', 'soap', 'grpc', 'agile', 'scrum',
    'junit', 'pytest', 'mocha', 'jest', 'selenium', 'cypress',
}

DEGREE_PATTERNS = [
    r'\bb\.?tech\.?\b',
    r'\bb\.?(?:a|s|sc|com|eng)\.?\b',
    r'\bm\.?tech\.?\b',
    r'\bm\.?(?:a|s|sc|com|eng)\.?\b',
    r'\bm\.?\.?b\.?\.?a\.?\b',
    r'\bphd\.?\b',
    r'\bbachelor(?:\'s)?(?:\s+(?:of|in))?\b',
    r'\bmaster(?:\'s)?(?:\s+(?:of|in))?\b',
    r'\bpostgraduate\b',
    r'\bdiplomate?\b',
]

DURATION_PATTERNS = [
    r'\b\d{4}\s*[-–]\s*(?:\d{4}|present|current)\b',
    r'\b\d{1,2}\s*(?:years?|yrs?|months?|mons?)\b',
]

# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_skills(text):
    """Extract SKILL entities"""
    if not isinstance(text, str):
        return ''
    
    text_lower = text.lower()
    found = set()
    
    for skill in TECH_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.add(skill.title())
    
    return '|'.join(sorted(found)) if found else ''

def extract_degrees(text):
    """Extract DEGREE entities"""
    if not isinstance(text, str):
        return ''
    
    found = set()
    
    for pattern in DEGREE_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            found.add(match.group(0).strip())
    
    return '|'.join(sorted(found)) if found else ''

def extract_durations(text):
    """Extract EXPERIENCE_DURATION entities"""
    if not isinstance(text, str):
        return ''
    
    found = set()
    
    for pattern in DURATION_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            found.add(match.group(0).strip())
    
    return '|'.join(sorted(found)) if found else ''

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PHASE 2.5: CREATE LABELLED500.CSV")
    print("=" * 70)
    print()
    
    # Load
    print("[1/3] Loading resumes_unified.csv...")
    try:
        df = pd.read_csv(RESUMES_UNIFIED, encoding='utf-8')
        print(f"  ✓ Loaded {len(df)} resumes")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return
    
    print()
    
    # Sample 500
    print("[2/3] Sampling 500 resumes...")
    sample_df = df.sample(n=min(500, len(df)), random_state=42).reset_index(drop=True)
    print(f"  ✓ Sampled {len(sample_df)} resumes")
    print()
    
    # Extract entities
    print("[3/3] Auto-extracting entities...")
    
    # Find text column
    text_col = None
    for col in ['resume_text', 'text', 'resume', 'content', 'description']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        text_col = df.columns[0]
    
    results = []
    for idx, row in sample_df.iterrows():
        text = str(row[text_col]) if text_col in row else ''
        resume_id = str(row.get('id', row.get('resume_id', f'resume_{idx}')))
        
        skills = extract_skills(text)
        degrees = extract_degrees(text)
        durations = extract_durations(text)
        
        results.append({
            'resume_id': resume_id,
            'SKILL': skills,
            'JOB_TITLE': '',
            'EXPERIENCE_DURATION': durations,
            'DEGREE': degrees,
            'INSTITUTION': '',
            'CERTIFICATION': ''
        })
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(sample_df)}...")
    
    print(f"  ✓ Extracted from {len(results)} resumes")
    print()
    
    # Save
    print("Saving to CSV...")
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"  ✓ Saved to: {OUTPUT_CSV}")
    print()
    
    # Stats
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total resumes:               {len(results)}")
    print(f"With SKILL entities:         {sum(1 for r in results if r['SKILL'])}")
    print(f"With DEGREE entities:        {sum(1 for r in results if r['DEGREE'])}")
    print(f"With EXPERIENCE_DURATION:    {sum(1 for r in results if r['EXPERIENCE_DURATION'])}")
    print()
    print("NEXT: Open labelled500.csv and fill in:")
    print("  - JOB_TITLE")
    print("  - INSTITUTION")
    print("  - CERTIFICATION")
    print()

if __name__ == '__main__':
    main()
