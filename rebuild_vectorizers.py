#!/usr/bin/env python3
"""
Rebuild TF-IDF vectorizers from the re-labeled resume database.

After running relabel_all_resumes.py, run this to:
1. Rebuild TF-IDF vectorizers with richer, multi-profession vocabulary
2. Re-vectorize all resumes
3. Save new vectorizer .pkl and resume vector .npz files

Usage:
    python rebuild_vectorizers.py
"""

import pickle
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = Path(r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer")
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_CSV = DATA_PROCESSED / "resumes_with_entities_v3.csv"
BACKUP_DIR = DATA_PROCESSED / "backup"

ENTITY_TYPES = ['SKILL', 'JOB_TITLE', 'DEGREE', 'INSTITUTION', 'CERTIFICATION']

# More generous TF-IDF settings to capture diverse profession skills
TFIDF_SETTINGS = {
    'max_features': 10000,  # was 5000
    'min_df': 1,            # was 2 — capture rare profession-specific skills
    'max_df': 0.95,         # same
    'ngram_range': (1, 2),  # add bigrams for multi-word skills like "machine learning"
    'sublinear_tf': True,   # better for diverse vocabularies
}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("REBUILD TF-IDF VECTORIZERS")
    print("=" * 70)
    
    # Check input
    if not INPUT_CSV.exists():
        print(f"ERROR Input file not found: {INPUT_CSV}")
        print(f"   Run relabel_all_resumes.py first!")
        return
    
    # Load data
    print(f"\n[1/4] Loading re-labeled resume data...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  OK Loaded {len(df)} resumes")
    
    # Backup old files
    print(f"\n[2/4] Backing up existing vectorizers...")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    backed_up = 0
    for entity_type in ENTITY_TYPES:
        for pattern in [f"vectorizer_{entity_type.lower()}.pkl", 
                        f"resume_vectors_{entity_type.lower()}.npz"]:
            src = DATA_PROCESSED / pattern
            if src.exists():
                dst = BACKUP_DIR / f"{timestamp}_{pattern}"
                shutil.copy2(src, dst)
                backed_up += 1
    
    # Also backup the old entities CSV
    old_entities = DATA_PROCESSED / "resumes_with_entities.csv"
    if old_entities.exists():
        shutil.copy2(old_entities, BACKUP_DIR / f"{timestamp}_resumes_with_entities.csv")
        backed_up += 1
    
    print(f"  OK Backed up {backed_up} files to {BACKUP_DIR.name}/")
    
    # Build vectorizers
    print(f"\n[3/4] Building TF-IDF vectorizers...")
    print(f"  Settings: max_features={TFIDF_SETTINGS['max_features']}, "
          f"min_df={TFIDF_SETTINGS['min_df']}, ngram_range={TFIDF_SETTINGS['ngram_range']}")
    
    vectorizers = {}
    resume_vectors = {}
    
    for entity_type in ENTITY_TYPES:
        col = entity_type
        
        # Prepare text: join pipe-separated entities with spaces
        texts = []
        for _, row in df.iterrows():
            val = row.get(col, "")
            if pd.notna(val) and str(val).strip():
                # Convert pipe-separated to space-separated
                text = str(val).replace("|", " ")
                texts.append(text)
            else:
                texts.append("")
        
        # Count non-empty
        non_empty = sum(1 for t in texts if t.strip())
        
        if non_empty < 5:
            print(f"  WARN {entity_type}: only {non_empty} non-empty entries, skipping")
            continue
        
        # Fit TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_SETTINGS['max_features'],
            min_df=TFIDF_SETTINGS['min_df'],
            max_df=TFIDF_SETTINGS['max_df'],
            ngram_range=TFIDF_SETTINGS['ngram_range'],
            sublinear_tf=TFIDF_SETTINGS['sublinear_tf'],
            lowercase=True,
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z0-9+#.]*\b',  # better token pattern for tech skills
        )
        
        vectors = vectorizer.fit_transform(texts)
        vocab_size = len(vectorizer.vocabulary_)
        
        vectorizers[entity_type] = vectorizer
        resume_vectors[entity_type] = vectors
        
        print(f"  OK {entity_type:20s}: vocab={vocab_size:5d}, non-empty={non_empty}/{len(df)}, "
              f"matrix={vectors.shape}")
    
    # Save
    print(f"\n[4/4] Saving vectorizers and vectors...")
    
    for entity_type in vectorizers:
        # Save vectorizer
        vectorizer_path = DATA_PROCESSED / f"vectorizer_{entity_type.lower()}.pkl"
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizers[entity_type], f)
        
        # Save vectors
        vector_path = DATA_PROCESSED / f"resume_vectors_{entity_type.lower()}.npz"
        save_npz(str(vector_path), resume_vectors[entity_type])
        
        print(f"  OK {entity_type}: saved vectorizer + vectors")
    
    # Also copy the v2 entities file to the standard name for app.py
    print(f"\n  Updating resumes_with_entities.csv...")
    shutil.copy2(INPUT_CSV, old_entities)
    print(f"  OK Copied v2 entities to {old_entities.name}")
    
    # Print vocabulary samples
    print(f"\n{'=' * 70}")
    print("VOCABULARY SAMPLES")
    print("=" * 70)
    
    for entity_type in vectorizers:
        vocab = sorted(vectorizers[entity_type].vocabulary_.keys())
        sample = vocab[:30]
        print(f"\n  {entity_type} ({len(vocab)} terms):")
        print(f"    {', '.join(sample)}...")
    
    # Save metadata
    metadata_path = DATA_PROCESSED / "vectorization_metadata_v2.txt"
    with open(metadata_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("VECTORIZATION V2 - REBUILT WITH GEMINI-LABELED DATA\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input: {INPUT_CSV.name}\n")
        f.write(f"Resumes: {len(df)}\n\n")
        f.write("TF-IDF Settings:\n")
        for k, v in TFIDF_SETTINGS.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nVocabulary sizes:\n")
        for entity_type in vectorizers:
            f.write(f"  {entity_type}: {len(vectorizers[entity_type].vocabulary_)}\n")
    
    print(f"\nOK Saved metadata to {metadata_path.name}")
    print(f"\n{'=' * 70}")
    print("OK VECTORIZER REBUILD COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
