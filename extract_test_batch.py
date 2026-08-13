"""
Extract stratified 50-resume test batch from resumes_unified.csv
Prepares CSV for Gemini labeling with 6 entity types:
SKILL, JOB_TITLE, EXPERIENCE_DURATION, DEGREE, INSTITUTION, CERTIFICATION
"""

import pandas as pd
import os
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

CSV_PATH = r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer\data\processed\resumes_unified.csv"
OUTPUT_DIR = r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer\data\labeling"
SAMPLE_SIZE = 50
RESUMES_PER_CATEGORY = 2  # 50 / 24 ≈ 2 per category

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # Read full CSV
    print(f"\n📖 Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"   Total resumes: {len(df)}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Check category distribution
    print(f"\n📊 Category distribution:")
    cat_counts = df['Category'].value_counts()
    print(cat_counts)
    
    # Stratified sampling: 2 resumes per category
    print(f"\n🎯 Extracting stratified sample ({SAMPLE_SIZE} resumes, {RESUMES_PER_CATEGORY} per category)...")
    
    sample_df = df.groupby('Category', group_keys=False).apply(
        lambda x: x.sample(n=min(RESUMES_PER_CATEGORY, len(x)), random_state=42)
    ).reset_index(drop=True)
    
    print(f"   Sample size: {len(sample_df)}")
    print(f"\n   Sample category distribution:")
    print(sample_df['Category'].value_counts().sort_index())
    
    # Create labeling CSV with structure for annotation
    # Columns: resume_id, category, resume_text, SKILL, JOB_TITLE, EXPERIENCE_DURATION, DEGREE, INSTITUTION, CERTIFICATION
    labeling_df = sample_df[['ID', 'Category', 'Resume_text']].copy()
    labeling_df.rename(columns={'ID': 'resume_id', 'Category': 'category', 'Resume_text': 'resume_text'}, inplace=True)
    
    # Add empty entity columns (to be filled by Gemini)
    entity_columns = ['SKILL', 'JOB_TITLE', 'EXPERIENCE_DURATION', 'DEGREE', 'INSTITUTION', 'CERTIFICATION']
    for col in entity_columns:
        labeling_df[col] = ""
    
    # Reorder columns for clarity
    labeling_df = labeling_df[['resume_id', 'category', 'resume_text'] + entity_columns]
    
    # Save labeling CSV
    output_csv = os.path.join(OUTPUT_DIR, "test_batch_50_unlabeled.csv")
    labeling_df.to_csv(output_csv, index=False, quoting=1)  # quoting=1 for QUOTE_ALL to handle newlines
    print(f"\n✅ Saved unlabeled batch: {output_csv}")
    print(f"   Shape: {labeling_df.shape}")
    
    # Show sample
    print(f"\n📋 Sample (first 2 rows):")
    print(labeling_df[['resume_id', 'category', 'resume_text']].head(2))
    
    # Create a reference CSV with just resume_id, category, text (for Gemini labeling script)
    reference_df = labeling_df[['resume_id', 'category', 'resume_text']].copy()
    reference_csv = os.path.join(OUTPUT_DIR, "test_batch_50_reference.csv")
    reference_df.to_csv(reference_csv, index=False, quoting=1)
    print(f"\n✅ Saved reference CSV: {reference_csv}")
    
    print(f"\n🎓 Next steps:")
    print(f"   1. Use test_batch_50_unlabeled.csv as the base for manual labeling")
    print(f"   2. OR run the Gemini labeling script to auto-label using test_batch_50_reference.csv")
    print(f"   3. Save labeled results back to test_batch_50_labeled.csv")
    print(f"   4. Run fine-tuning on the labeled data")

if __name__ == "__main__":
    main()