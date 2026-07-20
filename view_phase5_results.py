#!/usr/bin/env python3
"""
View Phase 5 Results
Run this in your project directory to display:
  - Top 10 matched resumes
  - Top 15 improvement recommendations
"""

import pandas as pd
from pathlib import Path

output_dir = Path("data/output/phase5")

if not output_dir.exists():
    print(f"✗ Output directory not found: {output_dir}")
    exit(1)

# Find latest files
csv_files = sorted(output_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
txt_files = sorted(output_dir.glob("*.txt"), key=lambda x: x.stat().st_mtime, reverse=True)

if not csv_files:
    print("✗ No CSV files found!")
    exit(1)

# Get latest timestamp
latest_timestamp = csv_files[0].name.split('_')[-1].replace('.csv', '')
print(f"Latest run: {latest_timestamp}\n")

# ========================================================================
# OVERALL RANKINGS
# ========================================================================
overall_file = [f for f in csv_files if 'overall_rankings' in f.name][0]
overall_df = pd.read_csv(overall_file)

print("=" * 100)
print("TOP 10 MATCHED RESUMES (Overall)")
print("=" * 100)
print()

display_cols = ['rank', 'resume_id', 'overall_score', 'skill_score', 'job_title_score', 'degree_score', 'experience_score']
print(overall_df[display_cols].head(10).to_string(index=False))

print(f"\n✓ Total resumes ranked: {len(overall_df)}")
print(f"✓ Overall score range: {overall_df['overall_score'].min():.3f} - {overall_df['overall_score'].max():.3f}")
print(f"✓ Skill match range: {overall_df['skill_score'].min():.3f} - {overall_df['skill_score'].max():.3f}")

# ========================================================================
# SKILL RANKINGS
# ========================================================================
skill_file = [f for f in csv_files if 'skill_rankings' in f.name][0]
skill_df = pd.read_csv(skill_file)

print("\n" + "=" * 100)
print("TOP 10 BY SKILL MATCH")
print("=" * 100)
print()
print(skill_df[['rank', 'resume_id', 'SKILL_score']].head(10).to_string(index=False))

# ========================================================================
# IMPROVEMENT RECOMMENDATIONS
# ========================================================================
rec_file = [f for f in csv_files if 'improvement_recommendations' in f.name][0]
rec_df = pd.read_csv(rec_file)

print("\n" + "=" * 100)
print("PERSONALIZED SKILL DEVELOPMENT PLAN")
print("=" * 100)
print()

# Group by section
sections = rec_df.groupby('section')

for section_name, section_data in sections:
    print(f"\n{section_name}")
    print("-" * 100)
    
    if section_name == '1. REQUIRED SKILLS FOR THIS ROLE':
        for _, row in section_data.iterrows():
            print(f"  {row['item']}")
    
    elif section_name == '2. YOUR STRENGTHS 🌟':
        for _, row in section_data.iterrows():
            print(f"\n  ✅ {row['item']}")
            print(f"  💡 {row['analysis']}")
    
    elif section_name == '3. SKILLS TO DEVELOP 📚':
        for _, row in section_data.iterrows():
            if row['type'] == 'missing_skills_header':
                print(f"\n  {row['item']}\n")
            elif row['type'] == 'missing_skill':
                priority_emoji = "🔴" if row['priority'] == 'HIGH' else "🟡" if row['priority'] == 'MEDIUM' else "🟢"
                print(f"  {priority_emoji} {row['item']}")
                if pd.notna(row.get('frequency')):
                    print(f"     Found in {row.get('frequency')} top candidates")
                if pd.notna(row.get('analysis')):
                    print(f"     {row['analysis']}\n")
    
    elif section_name == '4. PERSONALIZED ACTION PLAN':
        for _, row in section_data.iterrows():
            print(f"\n  🎯 {row['item']}")
            if pd.notna(row.get('analysis')):
                print(f"  {row['analysis']}")

# ========================================================================
# SUMMARY REPORT
# ========================================================================
if txt_files:
    summary_file = [f for f in txt_files if 'phase5_summary' in f.name][0]
    print("\n" + "=" * 100)
    print("FULL SUMMARY REPORT")
    print("=" * 100)
    with open(summary_file) as f:
        print(f.read())