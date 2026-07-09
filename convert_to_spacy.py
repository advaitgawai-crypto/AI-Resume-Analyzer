"""
Phase 2, Step 3: Convert labeled CSV to spaCy JSON format
Input: labelled200.csv (labeled entities in CSV columns)
Output: train.json (80%), test.json (20%)
"""

import pandas as pd
import random
from pathlib import Path
import json

# ============================================================================
# CONFIG
# ============================================================================
CSV_PATH = Path("data/processed/labelled200.csv")  # Your renamed CSV file
OUTPUT_DIR = Path("data/processed")
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# Entity types we labeled
ENTITY_TYPES = ["SKILL", "JOB_TITLE", "EXPERIENCE_DURATION", "DEGREE", "INSTITUTION", "CERTIFICATION"]

# ============================================================================
# STEP 1: LOAD CSV FILE
# ============================================================================
print("[1/5] Loading CSV file...")
try:
    df = pd.read_csv(CSV_PATH, keep_default_na=False, na_values=[])
    print(f"✓ Loaded {len(df)} resumes from {CSV_PATH}")
except FileNotFoundError:
    print(f"❌ ERROR: Could not find {CSV_PATH}")
    print(f"   Please ensure the file exists at: C:\\Users\\Advait Gawai\\OneDrive\\Desktop\\AI resume Analyzer\\data\\processed\\labelled200.csv")
    exit(1)

print(f"  Columns: {list(df.columns)}")

# ============================================================================
# STEP 2: BUILD TRAINING DATA (find character offsets)
# ============================================================================
print("\n[2/5] Finding entity character offsets in resume text...")

training_data = []
skipped_count = 0

for idx, row in df.iterrows():
    resume_text = row['Resume_text']
    entities = []
    
    # For each entity type, find its position in the text
    for entity_type in ENTITY_TYPES:
        if entity_type not in df.columns:
            continue
            
        entity_value = row[entity_type]
        
        # Skip empty entities
        if not entity_value or pd.isna(entity_value) or str(entity_value).strip() == '':
            continue
        
        entity_value = str(entity_value).strip()  # Convert to string
        
        # Handle pipe-separated multiple values (e.g., "Python|Java|AWS")
        if '|' in entity_value:
            values = [v.strip() for v in entity_value.split('|')]
        else:
            values = [entity_value]
        
        # Find each value in the resume text
        for value in values:
            if not value:  # Skip empty values
                continue
            # Case-insensitive search
            value_lower = value.lower()
            text_lower = resume_text.lower()
            
            start_pos = text_lower.find(value_lower)
            
            if start_pos != -1:
                # Found the entity
                end_pos = start_pos + len(value)
                entities.append((start_pos, end_pos, entity_type))
            # If not found, skip silently (entity may not appear exactly in text)
    
    # Only add to training data if we found at least one entity
    if entities:
        training_data.append((resume_text, {"entities": entities}))
    else:
        skipped_count += 1

print(f"✓ Found entities in {len(training_data)} resumes")
if skipped_count > 0:
    print(f"  (Skipped {skipped_count} resumes with no detectable entities)")

# ============================================================================
# STEP 3: SPLIT TRAIN/TEST
# ============================================================================
print("\n[3/5] Splitting train/test (80/20)...")
random.seed(RANDOM_SEED)
random.shuffle(training_data)

split_idx = int(len(training_data) * TRAIN_SPLIT)
train_data = training_data[:split_idx]
test_data = training_data[split_idx:]

print(f"✓ Train set: {len(train_data)} resumes")
print(f"✓ Test set: {len(test_data)} resumes")

# ============================================================================
# STEP 4: CONVERT TO JSON & SAVE
# ============================================================================
print("\n[4/5] Converting to JSON format...")

train_json_fixed = []
for text, annot in train_data:
    train_json_fixed.append({
        "text": text,
        "entities": annot["entities"]
    })

test_json_fixed = []
for text, annot in test_data:
    test_json_fixed.append({
        "text": text,
        "entities": annot["entities"]
    })

# Save as JSON
train_json_path = OUTPUT_DIR / "train.json"
test_json_path = OUTPUT_DIR / "test.json"

with open(train_json_path, 'w', encoding='utf-8') as f:
    json.dump(train_json_fixed, f, ensure_ascii=False, indent=2)

with open(test_json_path, 'w', encoding='utf-8') as f:
    json.dump(test_json_fixed, f, ensure_ascii=False, indent=2)

print(f"✓ Saved: {train_json_path} ({len(train_json_fixed)} records)")
print(f"✓ Saved: {test_json_path} ({len(test_json_fixed)} records)")

# ============================================================================
# STEP 5: SUMMARY
# ============================================================================
print("\n[5/5] Summary")
print("="*70)
print(f"Training data: {len(train_data)} resumes")
print(f"Test data: {len(test_data)} resumes")
print(f"Entity types: {', '.join(ENTITY_TYPES)}")
print()
print("✓ STEP 3 COMPLETE!")
print()
print("NEXT STEPS (Phase 2, Step 4):")
print("  1. Train spaCy NER model on train.json")
print("  2. Evaluate on test.json")
print("  3. Extract entities from all 2,484 resumes")
print()
print(f"Files ready for training:")
print(f"  {train_json_path}")
print(f"  {test_json_path}")
print("="*70)