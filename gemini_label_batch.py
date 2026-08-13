"""
Gemini-based NER labeling for resume text
Uses the same workflow as Phase 2.5 to label 6 entity types:
SKILL, JOB_TITLE, EXPERIENCE_DURATION, DEGREE, INSTITUTION, CERTIFICATION

Output format: pipe-delimited entity lists (same as Phase 2.5)
"""

import pandas as pd
import os
import json
import time
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    print("❌ google-generativeai not installed. Run: pip install google-generativeai")
    exit(1)

# ============================================================================
# CONFIG
# ============================================================================

INPUT_CSV = r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer\data\labeling\test_batch_50_reference.csv"
OUTPUT_CSV = r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer\data\labeling\test_batch_50_labeled.csv"

# Get your Gemini API key from environment or hardcode
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Set env var or paste here
if not GEMINI_API_KEY:
    print("⚠️  GEMINI_API_KEY not set. Paste your API key here:")
    GEMINI_API_KEY = input(">>> ")

# ============================================================================
# PROMPTS
# ============================================================================

SYSTEM_PROMPT = """You are an expert NER (Named Entity Recognition) labeler for resumes.
Extract entities from the resume text and return them in this exact JSON format:

{
  "SKILL": ["skill1", "skill2", ...],
  "JOB_TITLE": ["title1", "title2", ...],
  "EXPERIENCE_DURATION": ["duration1", "duration2", ...],
  "DEGREE": ["degree1", "degree2", ...],
  "INSTITUTION": ["institution1", "institution2", ...],
  "CERTIFICATION": ["cert1", "cert2", ...]
}

RULES:
- SKILL: Technical skills, programming languages, tools, soft skills (e.g., "Python", "Leadership", "Agile")
- JOB_TITLE: Job titles or roles (e.g., "Senior Software Engineer", "Data Analyst")
- EXPERIENCE_DURATION: Time periods, durations (e.g., "3 years", "2019-2021", "Jan 2020 - Jun 2022")
- DEGREE: Academic degrees (e.g., "B.S. in Computer Science", "MBA", "Bachelor's")
- INSTITUTION: Schools, universities, companies (e.g., "MIT", "Google", "Stanford University")
- CERTIFICATION: Professional certifications, licenses (e.g., "AWS Certified Solutions Architect", "CPA")

Return ONLY valid JSON, no markdown, no preamble."""

# ============================================================================
# FUNCTIONS
# ============================================================================

def label_resume_with_gemini(resume_text, model_name="gemini-1.5-flash"):
    """Label a single resume using Gemini"""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        response = client.messages.create(
            model=model_name,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Resume text:\n\n{resume_text}"}
            ]
        )
        
        # Extract JSON from response
        response_text = response.content[0].text
        
        # Parse JSON
        try:
            entities = json.loads(response_text)
        except json.JSONDecodeError:
            print(f"   ⚠️  JSON parse error, trying to extract JSON...")
            # Try to extract JSON from response
            import re
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                entities = json.loads(match.group())
            else:
                print(f"   ❌ Could not parse response: {response_text[:100]}")
                entities = {k: [] for k in ['SKILL', 'JOB_TITLE', 'EXPERIENCE_DURATION', 'DEGREE', 'INSTITUTION', 'CERTIFICATION']}
        
        return entities
    
    except Exception as e:
        print(f"   ❌ API error: {e}")
        return {k: [] for k in ['SKILL', 'JOB_TITLE', 'EXPERIENCE_DURATION', 'DEGREE', 'INSTITUTION', 'CERTIFICATION']}

def format_entities_as_pipe_delimited(entities):
    """Convert entity dict to pipe-delimited strings (same as Phase 2.5)"""
    formatted = {}
    for key, values in entities.items():
        if isinstance(values, list):
            formatted[key] = " | ".join(values) if values else ""
        else:
            formatted[key] = values
    return formatted

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("🚀 Starting Gemini-based NER labeling...\n")
    
    # Setup API
    genai.configure(api_key=GEMINI_API_KEY)
    print(f"✅ Gemini API configured\n")
    
    # Read input CSV
    print(f"📖 Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"   Total resumes to label: {len(df)}\n")
    
    # Prepare output dataframe
    output_df = df[['resume_id', 'category', 'resume_text']].copy()
    entity_columns = ['SKILL', 'JOB_TITLE', 'EXPERIENCE_DURATION', 'DEGREE', 'INSTITUTION', 'CERTIFICATION']
    for col in entity_columns:
        output_df[col] = ""
    
    # Label each resume
    print("🏷️  Labeling resumes...\n")
    for idx, row in df.iterrows():
        resume_id = row['resume_id']
        category = row['category']
        resume_text = row['resume_text']
        
        print(f"[{idx+1}/{len(df)}] {resume_id} ({category})")
        
        # Call Gemini
        entities = label_resume_with_gemini(resume_text)
        formatted = format_entities_as_pipe_delimited(entities)
        
        # Store in output df
        for col in entity_columns:
            output_df.loc[idx, col] = formatted.get(col, "")
        
        # Print sample
        print(f"    SKILLs: {formatted.get('SKILL', '')[:60]}...")
        print(f"    JOB_TITLEs: {formatted.get('JOB_TITLE', '')[:60]}...")
        
        # Rate limit: Gemini has limits, add small delay
        time.sleep(0.5)
    
    # Save labeled CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False, quoting=1)
    print(f"\n✅ Saved labeled CSV: {OUTPUT_CSV}")
    print(f"   Shape: {output_df.shape}")
    
    # Show sample
    print(f"\n📋 Sample labeled resume:")
    sample_row = output_df.iloc[0]
    print(f"   resume_id: {sample_row['resume_id']}")
    print(f"   category: {sample_row['category']}")
    print(f"   SKILL: {sample_row['SKILL'][:80]}...")
    print(f"   JOB_TITLE: {sample_row['JOB_TITLE'][:80]}...")
    print(f"   DEGREE: {sample_row['DEGREE'][:80]}...")
    
    print(f"\n🎓 Next steps:")
    print(f"   1. Review the labeled data in {OUTPUT_CSV}")
    print(f"   2. Fix any obvious Gemini mistakes manually if needed")
    print(f"   3. Convert to spaCy format using your existing Phase 2.5 pipeline")
    print(f"   4. Train fine-tuned model on this labeled data")

if __name__ == "__main__":
    main()