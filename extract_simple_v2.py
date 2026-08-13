import pandas as pd
import os

CSV_PATH = r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer\data\processed\resumes_unified.csv"
OUTPUT_DIR = r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer\data\labeling"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📁 Output directory: {OUTPUT_DIR}")

print(f"\n📖 Reading CSV...")
df = pd.read_csv(CSV_PATH)
print(f"   Total resumes: {len(df)}")

print(f"\n🎯 Extracting stratified sample (2 per category)...")

# Stratified sampling - preserve all columns
sample_dfs = []
for category in df['Category'].unique():
    category_df = df[df['Category'] == category]
    sample_size = min(2, len(category_df))
    sample_dfs.append(category_df.sample(n=sample_size, random_state=42))

sample_df = pd.concat(sample_dfs, ignore_index=True)

print(f"   Sample size: {len(sample_df)}")
print(f"   Columns in sample: {sample_df.columns.tolist()}")

# Create labeling dataframe
labeling_df = sample_df[['ID', 'Category', 'Resume_text']].copy()
labeling_df.columns = ['resume_id', 'category', 'resume_text']

# Add empty entity columns
for col in ['SKILL', 'JOB_TITLE', 'EXPERIENCE_DURATION', 'DEGREE', 'INSTITUTION', 'CERTIFICATION']:
    labeling_df[col] = ""

# Reorder columns
labeling_df = labeling_df[['resume_id', 'category', 'resume_text', 'SKILL', 'JOB_TITLE', 'EXPERIENCE_DURATION', 'DEGREE', 'INSTITUTION', 'CERTIFICATION']]

# Save unlabeled CSV
output_csv = os.path.join(OUTPUT_DIR, "test_batch_50_unlabeled.csv")
labeling_df.to_csv(output_csv, index=False, quoting=1)
print(f"\n✅ Saved: {output_csv}")

# Save reference CSV (for Gemini)
reference_csv = os.path.join(OUTPUT_DIR, "test_batch_50_reference.csv")
labeling_df[['resume_id', 'category', 'resume_text']].to_csv(reference_csv, index=False, quoting=1)
print(f"✅ Saved: {reference_csv}")

# Show category distribution
print(f"\n📊 Sample category distribution:")
print(labeling_df['category'].value_counts().sort_index())

print(f"\n✅ Done! Both CSVs ready for labeling.")
