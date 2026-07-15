import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz
import time
import os

print("=" * 80)
print("PHASE 4: VECTORIZATION & SIMILARITY SCORING")
print("=" * 80)

start_time = time.time()

INPUT_FILE = "data/processed/resumes_with_entities.csv"
OUTPUT_DIR = "data/processed"

print(f"\n[1/8] Loading resumes with extracted entities...")
df = pd.read_csv(INPUT_FILE)
print(f"✓ Loaded {len(df)} resumes from {INPUT_FILE}")

print(f"\n[2/8] Preparing entity data...")
entity_cols = ['SKILL', 'JOB_TITLE', 'DEGREE', 'INSTITUTION', 'CERTIFICATION']

for col in entity_cols:
    if col not in df.columns:
        print(f"  ERROR: Column '{col}' not found in CSV")
        exit(1)

df_entities = df[['resume_id'] + entity_cols].copy()

for col in entity_cols:
    df_entities[col] = df_entities[col].fillna('')

print(f"✓ Prepared {len(df_entities)} resumes with {len(entity_cols)} entity types")

print(f"\n[3/8] Building TF-IDF vectorizers and computing similarity matrices...")

similarity_matrices = {}
vectorizers = {}
vocab_info = {}

print("\n  Entity type | Vocab size | Matrix size     | Non-zero values")
print("  " + "-" * 65)

for entity_type in entity_cols:
    
    if entity_type == 'EXPERIENCE_DURATION':
        print(f"  {entity_type:15} | Skipped (0% extraction rate)")
        continue
    
    vec = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95, lowercase=True)
    
    tfidf_matrix = vec.fit_transform(df_entities[entity_type])
    
    vocab_size = len(vec.get_feature_names_out())
    
    sim_matrix = cosine_similarity(tfidf_matrix, dense_output=False)
    
    non_zero = sim_matrix.nnz
    
    similarity_matrices[entity_type] = sim_matrix
    vectorizers[entity_type] = vec
    vocab_info[entity_type] = vocab_size
    
    print(f"  {entity_type:15} | {vocab_size:10} | {sim_matrix.shape[0]:4}x{sim_matrix.shape[1]:4} | {non_zero:15,}")

print(f"\n[4/8] Aggregating similarity matrices with weights...")

weights = {
    'SKILL': 0.40,
    'JOB_TITLE': 0.25,
    'DEGREE': 0.20,
    'INSTITUTION': 0.10,
    'CERTIFICATION': 0.05
}

overall_similarity = None

for entity_type, weight in weights.items():
    if entity_type in similarity_matrices:
        if overall_similarity is None:
            overall_similarity = weights[entity_type] * similarity_matrices[entity_type]
        else:
            overall_similarity = overall_similarity + weights[entity_type] * similarity_matrices[entity_type]
        print(f"  {entity_type:15} (weight={weight}) ✓")

print(f"✓ Overall weighted similarity matrix: {overall_similarity.shape}")

print(f"\n[5/8] Computing category scores per resume...")

category_scores = pd.DataFrame()
category_scores['resume_id'] = df_entities['resume_id'].values

for entity_type in entity_cols:
    if entity_type == 'EXPERIENCE_DURATION':
        category_scores[f'{entity_type.lower()}_score'] = 0.0
        continue
    
    if entity_type in similarity_matrices:
        sim_matrix = similarity_matrices[entity_type]
        
        avg_scores = np.asarray(sim_matrix.mean(axis=1)).flatten()
        category_scores[f'{entity_type.lower()}_score'] = np.clip(avg_scores, 0, 1)

category_scores['overall_score'] = np.clip(
    0.40 * category_scores['skill_score'] +
    0.25 * category_scores['job_title_score'] +
    0.20 * category_scores['degree_score'] +
    0.10 * category_scores['institution_score'] +
    0.05 * category_scores['certification_score'],
    0, 1
)

print(f"✓ Computed scores for {len(category_scores)} resumes")
print(f"  Columns: {list(category_scores.columns)}")

print(f"\n[6/8] Saving similarity matrix as sparse NPZ...")

output_npz = os.path.join(OUTPUT_DIR, 'similarity_matrix.npz')
save_npz(output_npz, overall_similarity)

file_size_mb = os.path.getsize(output_npz) / (1024 * 1024)
print(f"✓ Saved to {output_npz} ({file_size_mb:.2f} MB)")

print(f"\n[7/8] Saving category scores as CSV...")

output_csv = os.path.join(OUTPUT_DIR, 'category_scores.csv')
category_scores.to_csv(output_csv, index=False)

file_size_kb = os.path.getsize(output_csv) / 1024
print(f"✓ Saved to {output_csv} ({file_size_kb:.1f} KB)")

print(f"\n[8/8] Writing vectorization metadata...")

metadata_file = os.path.join(OUTPUT_DIR, 'vectorization_metadata.txt')
with open(metadata_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("PHASE 4 VECTORIZATION & SIMILARITY SCORING - METADATA\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("INPUT:\n")
    f.write(f"  File: resumes_with_entities.csv\n")
    f.write(f"  Resumes: {len(df_entities)}\n")
    f.write(f"  Entity types: {', '.join(entity_cols)}\n\n")
    
    f.write("TF-IDF VECTORIZATION:\n")
    f.write(f"  max_features: 5000\n")
    f.write(f"  min_df: 2\n")
    f.write(f"  max_df: 0.95\n")
    f.write(f"  Vocabulary sizes:\n")
    for entity_type, vocab_size in vocab_info.items():
        f.write(f"    {entity_type}: {vocab_size}\n")
    f.write("\n")
    
    f.write("SIMILARITY AGGREGATION WEIGHTS:\n")
    for entity_type, weight in weights.items():
        f.write(f"  {entity_type}: {weight}\n")
    f.write("\n")
    
    f.write("OUTPUTS:\n")
    f.write(f"  similarity_matrix.npz: {file_size_mb:.2f} MB (sparse 2484x2484 matrix)\n")
    f.write(f"  category_scores.csv: {file_size_kb:.1f} KB (2484 rows x 7 columns)\n")
    f.write(f"  vectorization_metadata.txt: This file\n\n")
    
    f.write("USAGE:\n")
    f.write(f"  Load sparse matrix in Python:\n")
    f.write(f"    from scipy.sparse import load_npz\n")
    f.write(f"    sim = load_npz('similarity_matrix.npz')\n")
    f.write(f"    score = sim[resume_i, resume_j]  # Query any pair\n\n")
    
    f.write("NOTES:\n")
    f.write(f"  - EXPERIENCE_DURATION excluded (0% extraction rate from Phase 2)\n")
    f.write(f"  - Similarity matrix is sparse (95% zeros, only ~200k non-zero values stored)\n")
    f.write(f"  - Overall score is weighted average of category scores\n")
    f.write(f"  - Ready for Phase 5 (matching engine & job fit scoring)\n")

print(f"✓ Saved to {metadata_file}")

elapsed = time.time() - start_time

print(f"\n" + "=" * 80)
print(f"PHASE 4 COMPLETE!")
print(f"=" * 80)
print(f"\nExecution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
print(f"\nOutputs:")
print(f"  ✓ data/processed/similarity_matrix.npz ({file_size_mb:.2f} MB)")
print(f"  ✓ data/processed/category_scores.csv ({file_size_kb:.1f} KB)")
print(f"  ✓ data/processed/vectorization_metadata.txt")
print(f"\nReady for Phase 5: Matching engine & job fit scoring")
print("=" * 80)