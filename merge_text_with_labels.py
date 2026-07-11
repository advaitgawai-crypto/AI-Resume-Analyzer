"""
Phase 2.5: Merge labeled entities with real resume text
Joins cleaned500.csv (entities) with resumes_unified.csv (actual text)
"""

import csv
import pandas as pd
from pathlib import Path

CLEANED_CSV = r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer\data\processed\cleaned500.csv"
RESUMES_UNIFIED = r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer\data\processed\resumes_unified.csv"
OUTPUT_CSV = r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer\data\processed\labelled500_with_text.csv"

print("=" * 70)
print("MERGE LABELED ENTITIES WITH REAL RESUME TEXT")
print("=" * 70)
print()

# Step 1: Load cleaned labels
print("[1/3] Loading cleaned500.csv (labeled entities)...")
try:
    labels_df = pd.read_csv(CLEANED_CSV)
    print(f"  ✓ Loaded {len(labels_df)} labeled resumes")
except Exception as e:
    print(f"  ❌ Error: {e}")
    exit(1)

print()

# Step 2: Load resume text
print("[2/3] Loading resumes_unified.csv (real text)...")
try:
    resumes_df = pd.read_csv(RESUMES_UNIFIED)
    print(f"  ✓ Loaded {len(resumes_df)} resumes")
    print(f"  ✓ Columns: {list(resumes_df.columns)}")
except Exception as e:
    print(f"  ❌ Error: {e}")
    exit(1)

print()

# Step 3: Identify text column
print("[3/3] Merging data...")
text_column = None
for col in ['resume_text', 'Resume_text', 'text', 'resume', 'content', 'description']:
    if col in resumes_df.columns:
        text_column = col
        print(f"  ✓ Found text column: '{col}'")
        break

if not text_column:
    print(f"  ❌ Could not find text column in resumes_unified.csv")
    print(f"  Available columns: {list(resumes_df.columns)}")
    exit(1)

# Merge on ID (resumes_unified uses 'ID', cleaned500 uses 'resume_id')
# First rename ID to resume_id in resumes_df for consistency
resumes_df_renamed = resumes_df.rename(columns={'ID': 'resume_id'})

merged_df = labels_df.merge(
    resumes_df_renamed[['resume_id', text_column]],
    on='resume_id',
    how='left'
)

# Check for missing text
missing_text = merged_df[text_column].isna().sum()
print(f"  ✓ Merged {len(merged_df)} rows")
if missing_text > 0:
    print(f"  ⚠ Warning: {missing_text} rows missing resume text")
    merged_df = merged_df[merged_df[text_column].notna()]
    print(f"  ✓ Kept {len(merged_df)} rows with text")

print()

# Rename text column to 'resume_text'
merged_df.rename(columns={text_column: 'resume_text'}, inplace=True)

# Reorder columns: resume_id, resume_text, then entities
cols = ['resume_id', 'resume_text', 'SKILL', 'JOB_TITLE', 'EXPERIENCE_DURATION', 'DEGREE', 'INSTITUTION', 'CERTIFICATION']
merged_df = merged_df[cols]

# Save
merged_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"✓ Saved merged data to: {OUTPUT_CSV}")
print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total rows: {len(merged_df)}")
print(f"Columns: {list(merged_df.columns)}")
print()
print("Sample row 1:")
row = merged_df.iloc[0]
print(f"  resume_id: {row['resume_id']}")
print(f"  resume_text: {str(row['resume_text'])[:100]}...")
print(f"  SKILL: {row['SKILL'][:50]}")
print(f"  JOB_TITLE: {row['JOB_TITLE'][:50]}")
print()
print("NEXT STEP:")
print("  python convert_to_spacy_with_text.py data/processed/labelled500_with_text.csv")
print()