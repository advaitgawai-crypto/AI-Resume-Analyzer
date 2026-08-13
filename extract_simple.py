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
sample_df = df.groupby('Category', group_keys=False).apply(
    lambda x: x.sample(n=min(2, len(x)), random_state=42), include_groups=False
).reset_index(drop=True)

print(f"   Sample size: {len(sample_df)}")
print(f"   Columns in sample: {sample_df.columns.tolist()}")

labeling_df = sample_df[['ID', 'Category', 'Resume_text']].copy()
labeling_df.columns = ['resume_id', 'category', 'resume_text']

for col in ['SKILL', 'JOB_TITLE', 'EXPERIENCE_DURATION', 'DEGREE', 'INSTITUTION', 'CERTIFICATION']:
    labeling_df[col] = ""

labeling_df = labeling_df[['resume_id', 'category', 'resume_text', 'SKILL', 'JOB_TITLE', 'EXPERIENCE_DURATION', 'DEGREE', 'INSTITUTION', 'CERTIFICATION']]

output_csv = os.path.join(OUTPUT_DIR, "test_batch_50_unlabeled.csv")
labeling_df.to_csv(output_csv, index=False, quoting=1)
print(f"\n✅ Saved: {output_csv}")

reference_csv = os.path.join(OUTPUT_DIR, "test_batch_50_reference.csv")
labeling_df[['resume_id', 'category', 'resume_text']].to_csv(reference_csv, index=False, quoting=1)
print(f"✅ Saved: {reference_csv}")

print(f"\n📊 Sample category distribution:")
print(labeling_df['category'].value_counts().sort_index())

print(f"\n✅ Done! Both CSVs ready for labeling.")