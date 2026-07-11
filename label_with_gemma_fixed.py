"""
Phase 2.5: Use Gemma 7B via Ollama to label 500 resumes
Reads resume_sample_500.csv, sends each resume to Gemma for entity extraction
Outputs labelled500_gemma.csv with extracted entities
"""

import pandas as pd
import requests
import json
import re
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

RESUME_SAMPLE_PATH = r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer\data\processed\resume_sample_500.csv"
OUTPUT_PATH = r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer\data\processed\labelled500_gemma.csv"
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma:7b"

EXTRACTION_PROMPT = """You are a resume entity extraction expert. Extract exactly these 6 entities:

1. SKILL: Technical skills, languages, tools (pipe-separated)
2. JOB_TITLE: Job titles/positions (pipe-separated)
3. EXPERIENCE_DURATION: Work duration, date ranges (pipe-separated)
4. DEGREE: Educational degrees (pipe-separated)
5. INSTITUTION: Universities/schools (pipe-separated)
6. CERTIFICATION: Professional certifications (pipe-separated)

Output ONLY a CSV row in this format (no other text):
resume_id,SKILL,JOB_TITLE,EXPERIENCE_DURATION,DEGREE,INSTITUTION,CERTIFICATION

Use empty string ("") if entity not found.
Use pipe (|) to separate multiple values.

Resume text:
{resume_text}

Output CSV row:"""

# ============================================================================
# FUNCTIONS
# ============================================================================

def call_gemma(resume_text, resume_id):
    """Call Gemma 7B via Ollama to extract entities"""
    prompt = EXTRACTION_PROMPT.format(resume_text=resume_text[:2000])  # Limit to 2000 chars
    
    try:
        response = requests.post(
            OLLAMA_API,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1  # Lower temp for consistent extraction
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            print(f"  ❌ API error for {resume_id}: {response.status_code}")
            return None
    except Exception as e:
        print(f"  ❌ Error calling Gemma for {resume_id}: {e}")
        return None

def parse_csv_row(csv_row):
    """Parse CSV row from Gemma output"""
    try:
        # Try to parse as CSV
        parts = csv_row.split(',', 6)  # Split into 7 parts max
        if len(parts) >= 7:
            return {
                'resume_id': parts[0].strip().strip('"'),
                'SKILL': parts[1].strip().strip('"'),
                'JOB_TITLE': parts[2].strip().strip('"'),
                'EXPERIENCE_DURATION': parts[3].strip().strip('"'),
                'DEGREE': parts[4].strip().strip('"'),
                'INSTITUTION': parts[5].strip().strip('"'),
                'CERTIFICATION': parts[6].strip().strip('"')
            }
    except:
        pass
    return None

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PHASE 2.5: GEMMA 7B LABELING")
    print("=" * 70)
    print()
    
    # Load resume_sample_500.csv
    print("[1/3] Loading resume_sample_500.csv...")
    try:
        df = pd.read_csv(RESUME_SAMPLE_PATH)
        print(f"  ✓ Loaded {len(df)} resumes")
    except Exception as e:
        print(f"  ❌ Error loading CSV: {e}")
        return
    
    print()
    
    # Check Ollama connection
    print("[2/3] Checking Ollama connection...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print(f"  ✓ Ollama is running")
        else:
            print(f"  ❌ Ollama error: {response.status_code}")
            return
    except Exception as e:
        print(f"  ❌ Ollama not running: {e}")
        print("  Make sure Ollama is started: ollama serve")
        return
    
    print()
    
    # Label resumes
    print("[3/3] Labeling resumes with Gemma 7B...")
    print()
    
    results = []
    errors = 0
    
    for idx, row in df.iterrows():
        resume_id = row['resume_id']
        
        # Get resume text (try different column names)
        resume_text = None
        for col in ['resume_text', 'text', 'resume', 'content']:
            if col in df.columns:
                resume_text = str(row[col])
                break
        
        if not resume_text:
            print(f"  ⚠ {resume_id}: No text found, skipping")
            errors += 1
            continue
        
        # Call Gemma
        gemma_output = call_gemma(resume_text, resume_id)
        
        if gemma_output:
            parsed = parse_csv_row(gemma_output)
            if parsed:
                results.append(parsed)
                print(f"  ✓ {resume_id} ({idx+1}/{len(df)})")
            else:
                print(f"  ⚠ {resume_id}: Could not parse output")
                errors += 1
        else:
            errors += 1
        
        # Progress indicator
        if (idx + 1) % 50 == 0:
            print(f"    ... {idx+1}/{len(df)} completed ({errors} errors)")
    
    print()
    print(f"✓ Labeled {len(results)} resumes ({errors} errors)")
    print()
    
    # Save results
    print("Saving to CSV...")
    if results:
        output_df = pd.DataFrame(results)
        output_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
        print(f"  ✓ Saved to: {OUTPUT_PATH}")
    else:
        print(f"  ❌ No results to save")
        return
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total resumes labeled: {len(results)}")
    print(f"Output file: {OUTPUT_PATH}")
    print()
    print("Next step:")
    print("  python convert_to_spacy_v2.py data/labelled500_gemma.csv data/train_v2.json data/test_v2.json")
    print()

if __name__ == '__main__':
    main()
