import pandas as pd

df = pd.read_csv('data/processed/category_scores.csv')

print("=" * 80)
print("CATEGORY SCORES - DATA PREVIEW")
print("=" * 80)
print("\nFirst 10 resumes:")
print(df.head(10).to_string())

print(f"\n\nStatistics:")
print(f"  Total resumes: {len(df)}")
print(f"  Min overall_score: {df['overall_score'].min():.3f}")
print(f"  Max overall_score: {df['overall_score'].max():.3f}")
print(f"  Mean overall_score: {df['overall_score'].mean():.3f}")
print(f"  Median overall_score: {df['overall_score'].median():.3f}")

print(f"\n\nTop 10 highest-quality resumes:")
top_10 = df.nlargest(10, 'overall_score')[['resume_id', 'skill_score', 'job_title_score', 'degree_score', 'institution_score', 'certification_score', 'overall_score']]
print(top_10.to_string())

print(f"\n\nBottom 10 lowest-quality resumes:")
bottom_10 = df.nsmallest(10, 'overall_score')[['resume_id', 'skill_score', 'job_title_score', 'degree_score', 'institution_score', 'certification_score', 'overall_score']]
print(bottom_10.to_string())

print("\n" + "=" * 80)