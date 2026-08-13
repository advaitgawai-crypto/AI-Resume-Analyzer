import pandas as pd
import spacy
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = Path(r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer")
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "resumes_unified.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "resumes_with_entities_v3.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "ner_model_v3"

ENTITY_TYPES = ['SKILL', 'JOB_TITLE', 'DEGREE', 'INSTITUTION', 'CERTIFICATION']

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("LOCAL RESUME DATABASE RE-LABELING WITH ENHANCED SPACY")
    print("=" * 70)
    
    print(f"\n[1/3] Loading upgraded model from {MODEL_PATH.name}...")
    nlp = spacy.load(str(MODEL_PATH))
    print("  ✓ Model loaded")
    
    print(f"\n[2/3] Loading resumes from {INPUT_CSV.name}...")
    df = pd.read_csv(INPUT_CSV)
    total = len(df)
    print(f"  ✓ Loaded {total} resumes")
    
    output_rows = []
    
    print(f"\n[3/3] Processing resumes locally (lightning fast)...")
    
    for idx, row in df.iterrows():
        resume_id = row['ID']
        category = row.get('Category', 'UNKNOWN')
        text = str(row.get('Resume_text', ''))
        
        if idx % 100 == 0:
            print(f"  Processed {idx}/{total} resumes...")
            
        if not text or len(text) < 50:
            formatted = {col: "" for col in ENTITY_TYPES}
            formatted['EXPERIENCE_DURATION'] = ""
        else:
            # Run spacy model
            doc = nlp(text)
            
            entities = {ent_type: [] for ent_type in ENTITY_TYPES}
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text)
            
            # Format to pipe-separated strings and deduplicate
            formatted = {}
            for k in ENTITY_TYPES:
                # deduplicate while preserving order
                unique_vals = list(dict.fromkeys(entities[k]))
                formatted[k] = "|".join(unique_vals) if unique_vals else ""
                
            formatted['EXPERIENCE_DURATION'] = "" # Kept empty as before unless we add regex back
            
        output_rows.append({
            'resume_id': resume_id,
            'text': text,
            'Category': category,
            **formatted
        })
        
    print(f"  Processed {total}/{total} resumes!")
    
    print(f"\nSaving final output...")
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(OUTPUT_CSV, index=False, quoting=1)
    
    print(f"  ✓ Saved: {OUTPUT_CSV}")
    
    # Stats
    print(f"\nEntity coverage:")
    for col in ENTITY_TYPES:
        non_empty = (output_df[col].fillna('').str.len() > 0).sum()
        print(f"  {col:25s}: {non_empty}/{total} ({100*non_empty/total:.1f}%)")
        
    print("\n✓ LOCAL RE-LABELING COMPLETE!")
    print("=" * 70)

if __name__ == '__main__':
    main()
