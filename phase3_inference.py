import csv
import spacy
import os
from pathlib import Path

def run_phase3_inference(resumes_csv, model_path, output_csv):
    """
    Phase 3: Extract entities from all 2,484 resumes using trained NER model.
    
    Input: resumes_unified.csv (resume_id, text, ...)
    Output: resumes_with_entities.csv (resume_id, text, SKILL, JOB_TITLE, EXPERIENCE_DURATION, DEGREE, INSTITUTION, CERTIFICATION)
    
    Args:
        resumes_csv: Path to resumes_unified.csv
        model_path: Path to trained ner_model_v2 directory
        output_csv: Path to output CSV file
    """
    
    # Load trained model
    print(f"Loading model from: {model_path}")
    try:
        nlp = spacy.load(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    # Read resumes
    print(f"\nReading resumes from: {resumes_csv}")
    resumes = []
    try:
        with open(resumes_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, 1):
                resumes.append(row)
                if i % 500 == 0:
                    print(f"  Loaded {i} resumes...")
        
        total_resumes = len(resumes)
        print(f"✓ Loaded {total_resumes} resumes")
    except Exception as e:
        print(f"✗ Error reading CSV: {e}")
        return False
    
    # Entity types we're extracting
    entity_types = ['SKILL', 'JOB_TITLE', 'EXPERIENCE_DURATION', 'DEGREE', 'INSTITUTION', 'CERTIFICATION']
    
    # Run inference
    print(f"\nRunning inference on {total_resumes} resumes...")
    results = []
    
    for i, resume in enumerate(resumes, 1):
        resume_id = resume.get('ID', '')
        resume_text = resume.get('Resume_text', '')
        
        if not resume_text or not resume_text.strip():
            # Empty resume
            result = {'resume_id': resume_id, 'text': resume_text}
            for entity_type in entity_types:
                result[entity_type] = ''
            results.append(result)
            continue
        
        # Process with spaCy
        doc = nlp(resume_text)
        
        # Extract entities
        entities_by_type = {entity_type: [] for entity_type in entity_types}
        
        for ent in doc.ents:
            if ent.label_ in entities_by_type:
                entities_by_type[ent.label_].append(ent.text)
        
        # Format output: pipe-delimited unique entities
        result = {'resume_id': resume_id, 'text': resume_text}
        for entity_type in entity_types:
            # Deduplicate and join
            unique_entities = list(dict.fromkeys(entities_by_type[entity_type]))
            result[entity_type] = '|'.join(unique_entities)
        
        results.append(result)
        
        # Progress indicator
        if i % 500 == 0:
            print(f"  Processed {i}/{total_resumes} resumes...")
    
    # Write output CSV
    print(f"\nWriting output to: {output_csv}")
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['resume_id', 'text'] + entity_types
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"✓ Output CSV created: {output_csv}")
        print(f"✓ Extracted entities from {total_resumes} resumes")
        return True
    
    except Exception as e:
        print(f"✗ Error writing output: {e}")
        return False

if __name__ == '__main__':
    import sys
    
    # Paths (adjust as needed)
    resumes_csv = r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer\data\processed\resumes_unified.csv"
    model_path = r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer\models\ner_model_v2"
    output_csv = r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer\data\processed\resumes_with_entities.csv"
    
    # Allow command-line override
    if len(sys.argv) > 1:
        resumes_csv = sys.argv[1]
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_csv = sys.argv[3]
    
    print("=" * 80)
    print("PHASE 3: FULL RESUME INFERENCE & ENTITY EXTRACTION")
    print("=" * 80)
    print(f"\nResumes CSV: {resumes_csv}")
    print(f"Model path: {model_path}")
    print(f"Output CSV: {output_csv}\n")
    
    # Check paths exist
    if not os.path.exists(resumes_csv):
        print(f"✗ Error: {resumes_csv} not found")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"✗ Error: {model_path} not found")
        sys.exit(1)
    
    # Run inference
    success = run_phase3_inference(resumes_csv, model_path, output_csv)
    
    if success:
        print("\n" + "=" * 80)
        print("PHASE 3 COMPLETE")
        print("=" * 80)
        print(f"\nNext steps:")
        print("  1. Verify output: resumes_with_entities.csv")
        print("  2. Check entity extraction quality")
        print("  3. Proceed to Phase 4: Vectorization & Similarity")
    else:
        print("\n✗ Phase 3 failed")
        sys.exit(1)
