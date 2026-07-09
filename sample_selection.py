"""
Phase 2: Sample Selection for NER Training
Stratified sampling of 500 resumes proportionally by profession
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
UNIFIED_CSV = r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer\data\processed\resumes_unified.csv"
OUTPUT_DIR = r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer\data\processed"
SAMPLE_SIZE = 500
RANDOM_SEED = 42

# ============================================================================
# STEP 1: LOAD UNIFIED CORPUS
# ============================================================================
print("[1/4] Loading unified corpus...")
df = pd.read_csv(UNIFIED_CSV)
print(f"✓ Loaded {len(df):,} resumes")
print(f"  Columns: {list(df.columns)}")

# ============================================================================
# STEP 2: ANALYZE PROFESSION DISTRIBUTION
# ============================================================================
print("\n[2/4] Analyzing profession distribution...")
profession_counts = df['Category'].value_counts().sort_values(ascending=False)
print(f"✓ Found {len(profession_counts)} professions:\n")
print(profession_counts)

# ============================================================================
# STEP 3: STRATIFIED SAMPLING
# ============================================================================
print(f"\n[3/4] Stratified sampling of {SAMPLE_SIZE} resumes...")
np.random.seed(RANDOM_SEED)

# Calculate proportion for each profession
profession_props = profession_counts / len(df)
sample_per_profession = (profession_props * SAMPLE_SIZE).round().astype(int)

# Adjust if rounding errors cause total != SAMPLE_SIZE
diff = SAMPLE_SIZE - sample_per_profession.sum()
if diff != 0:
    # Adjust the largest professions to match total
    sample_per_profession.iloc[0] += diff

print("\nSamples per profession:")
print(sample_per_profession)

# Perform stratified sampling
sampled_list = []
for profession in df['Category'].unique():
    profession_df = df[df['Category'] == profession]
    n_samples = min(sample_per_profession[profession], len(profession_df))
    sampled = profession_df.sample(n=n_samples, random_state=RANDOM_SEED)
    sampled_list.append(sampled)

sampled_df = pd.concat(sampled_list, ignore_index=True)

print(f"\n✓ Sampled {len(sampled_df)} resumes")
print(f"  Profession distribution in sample:")
print(sampled_df['Category'].value_counts().sort_values(ascending=False))

# ============================================================================
# STEP 4: CREATE ANNOTATION TEMPLATE
# ============================================================================
print("\n[4/4] Creating annotation template...")

# Create template with columns for each entity type
annotation_df = pd.DataFrame({
    'ID': sampled_df['ID'].values,
    'Category': sampled_df['Category'].values,
    'Resume_text': sampled_df['Resume_text'].values,
    'SKILL': [''] * len(sampled_df),
    'JOB_TITLE': [''] * len(sampled_df),
    'COMPANY': [''] * len(sampled_df),
    'EXPERIENCE_DURATION': [''] * len(sampled_df),
    'DEGREE': [''] * len(sampled_df),
    'INSTITUTION': [''] * len(sampled_df),
    'CERTIFICATION': [''] * len(sampled_df),
    'NOTES': [''] * len(sampled_df),
})

# Save to CSV
output_path = Path(OUTPUT_DIR) / "resumes_sample_500.csv"
annotation_df.to_csv(output_path, index=False, encoding='utf-8')
print(f"✓ Saved sample to: {output_path}")
print(f"  File size: {output_path.stat().st_size / (1024**2):.2f} MB")

# ============================================================================
# STEP 5: SAVE METADATA
# ============================================================================
metadata = {
    'total_resumes': len(df),
    'sample_size': len(sampled_df),
    'professions': len(profession_counts),
    'train_test_split': '80-20 (400 train, 100 test)',
    'entity_types': [
        'SKILL',
        'JOB_TITLE',
        'COMPANY',
        'EXPERIENCE_DURATION',
        'DEGREE',
        'INSTITUTION',
        'CERTIFICATION',
    ],
    'sampling_seed': RANDOM_SEED,
    'profession_distribution': profession_counts.to_dict(),
}

import json
metadata_path = Path(OUTPUT_DIR) / "sample_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"\n✓ Saved metadata to: {metadata_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SAMPLE SELECTION COMPLETE ✅")
print("="*70)
print(f"\nDeliverables:")
print(f"  1. resumes_sample_500.csv - 500 resumes ready for labeling")
print(f"  2. sample_metadata.json - Sampling statistics")
print(f"\nNext steps:")
print(f"  1. Open resumes_sample_500.csv in Excel/LibreOffice")
print(f"  2. For each resume, fill entity columns with extracted text")
print(f"  3. Entity format: pipe-separated if multiple values")
print(f"     Example: SKILL column = 'Python|Java|AWS|Machine Learning'")
print(f"  4. Save completed file as resumes_labeled_500.csv")
print(f"\nEstimated labeling time: ~2-3 weeks (5-10 mins per resume)")
print("="*70)