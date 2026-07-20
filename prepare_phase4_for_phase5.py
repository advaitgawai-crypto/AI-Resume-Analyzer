#!/usr/bin/env python3
"""
================================================================================
PREPARE PHASE 4 OUTPUTS FOR PHASE 5
================================================================================

Purpose:
  - Recompute TF-IDF vectorizers from Phase 4 entities
  - Save vectorizers as pickle files
  - Save resume vectors as sparse NPZ files
  
This bridges Phase 4 → Phase 5 by providing the vectorizers and vectors
that Phase 5 needs to vectorize the job posting.

Run this ONCE after Phase 4 is complete and before running Phase 5.

Author: ThunderLord
Date:   July 15, 2026
================================================================================
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz, csr_matrix


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer")
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

RESUMES_WITH_ENTITIES = DATA_PROCESSED / "resumes_with_entities.csv"
ENTITY_TYPES = ['SKILL', 'JOB_TITLE', 'DEGREE', 'INSTITUTION', 'CERTIFICATION']

# TF-IDF Hyperparameters (same as Phase 4)
TFIDF_PARAMS = {
    'max_features': 5000,
    'min_df': 2,
    'max_df': 0.95,
    'lowercase': True,
}


def prepare_phase4_for_phase5():
    """Main pipeline: load entities, train vectorizers, save vectors."""
    
    print("\n" + "="*80)
    print("PREPARING PHASE 4 OUTPUTS FOR PHASE 5")
    print("="*80 + "\n")
    
    # ========================================================================
    # STEP 1: Load resume entities
    # ========================================================================
    print("[1/4] Loading resume entities...")
    
    try:
        resume_df = pd.read_csv(RESUMES_WITH_ENTITIES)
        print(f"✓ Loaded {len(resume_df)} resumes")
    except Exception as e:
        print(f"✗ Failed to load resumes: {e}")
        return
    
    # ========================================================================
    # STEP 2: Train TF-IDF vectorizers for each category
    # ========================================================================
    print("\n[2/4] Training TF-IDF vectorizers for each category...")
    
    vectorizers = {}
    for entity_type in ENTITY_TYPES:
        print(f"\n  {entity_type}:")
        
        # Get pipe-delimited entities and split into list
        entities_list = []
        for idx, row in resume_df.iterrows():
            entity_str = row.get(entity_type, "")
            if pd.notna(entity_str) and entity_str.strip():
                # Join pipe-separated entities into single string for vectorization
                entities_list.append(entity_str.replace('|', ' '))
            else:
                entities_list.append("")
        
        print(f"    - Processing {len([e for e in entities_list if e])} non-empty entries")
        
        # Train TF-IDF vectorizer
        vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        vectorizer.fit(entities_list)
        
        vectorizers[entity_type] = vectorizer
        print(f"    - Vocabulary size: {len(vectorizer.get_feature_names_out())} features")
        
        # Save vectorizer
        vectorizer_path = DATA_PROCESSED / f"vectorizer_{entity_type.lower()}.pkl"
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        print(f"    - Saved: {vectorizer_path.name}")
    
    # ========================================================================
    # STEP 3: Transform resumes to vectors and save as sparse NPZ
    # ========================================================================
    print("\n[3/4] Transforming resumes to vectors and saving...")
    
    for entity_type in ENTITY_TYPES:
        print(f"\n  {entity_type}:")
        
        # Get entities
        entities_list = []
        for idx, row in resume_df.iterrows():
            entity_str = row.get(entity_type, "")
            if pd.notna(entity_str) and entity_str.strip():
                entities_list.append(entity_str.replace('|', ' '))
            else:
                entities_list.append("")
        
        # Transform to vectors
        vectors = vectorizers[entity_type].transform(entities_list)
        
        print(f"    - Shape: {vectors.shape}")
        print(f"    - Non-zero values: {vectors.nnz:,}")
        
        # Save as sparse NPZ
        vector_path = DATA_PROCESSED / f"resume_vectors_{entity_type.lower()}.npz"
        save_npz(str(vector_path), vectors)
        file_size_mb = vector_path.stat().st_size / (1024 * 1024)
        print(f"    - Saved: {vector_path.name} ({file_size_mb:.1f} MB)")
    
    # ========================================================================
    # STEP 4: Summary
    # ========================================================================
    print("\n[4/4] Summary")
    print("\n✓ All Phase 4 outputs prepared for Phase 5:")
    print("  - Vectorizers saved as .pkl files")
    print("  - Resume vectors saved as .npz (sparse) files")
    print("  - Ready to run Phase 5: phase5_matching_engine.py")
    
    print("\n" + "="*80)
    print("PREPARATION COMPLETE ✓")
    print("="*80 + "\n")


if __name__ == "__main__":
    prepare_phase4_for_phase5()
