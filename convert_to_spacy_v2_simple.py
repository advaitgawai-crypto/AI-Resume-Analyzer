"""
Convert labeled CSV to spaCy NER Training Format (Simple Version)
Phase 2.5: Improved version - no overlapping entity bugs
"""

import json
import csv
from pathlib import Path

def csv_to_spacy_json(csv_file, output_train, output_test, train_split=0.8):
    """
    Convert labeled CSV to spaCy format.
    Simple, straightforward implementation.
    """
    
    training_examples = []
    test_examples = []
    
    stats = {
        'total_resumes': 0,
        'valid_examples': 0,
        'skipped': 0,
    }
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            stats['total_resumes'] += 1
            
            resume_id = row['resume_id']
            
            # Extract entities from CSV
            entity_dict = {
                'SKILL': row.get('SKILL', '').strip(),
                'DEGREE': row.get('DEGREE', '').strip(),
                'EXPERIENCE_DURATION': row.get('EXPERIENCE_DURATION', '').strip(),
                'JOB_TITLE': row.get('JOB_TITLE', '').strip(),
                'INSTITUTION': row.get('INSTITUTION', '').strip(),
                'CERTIFICATION': row.get('CERTIFICATION', '').strip(),
            }
            
            # Skip if no entities found
            if not any(entity_dict.values()):
                stats['skipped'] += 1
                continue
            
            # Create synthetic resume text from the data
            resume_text = f"Resume {resume_id}: " + " | ".join([
                f"{k}: {v}" for k, v in entity_dict.items() if v
            ])
            
            # Extract all entity spans
            entities = []
            for ent_type, ent_values in entity_dict.items():
                if ent_values:
                    for value in ent_values.split('|'):
                        value = value.strip()
                        if value:
                            # Find entity in text (case-insensitive)
                            text_lower = resume_text.lower()
                            value_lower = value.lower()
                            
                            start = text_lower.find(value_lower)
                            if start != -1:
                                end = start + len(value)
                                entities.append((start, end, ent_type))
            
            # Remove overlapping entities (keep longest)
            if entities:
                # Sort by length (descending)
                entities = sorted(entities, key=lambda x: -(x[1] - x[0]))
                
                # Remove overlaps
                cleaned = []
                for start, end, ent_type in entities:
                    has_conflict = False
                    for s2, e2, _ in cleaned:
                        # Check overlap
                        if not (end <= s2 or start >= e2):
                            has_conflict = True
                            break
                    if not has_conflict:
                        cleaned.append((start, end, ent_type))
                
                entities = cleaned
            
            if not entities:
                stats['skipped'] += 1
                continue
            
            stats['valid_examples'] += 1
            
            # Format for spaCy
            example = {
                'text': resume_text,
                'entities': [(s, e, t) for s, e, t in entities]
            }
            
            # Split train/test
            if len(training_examples) / (stats['valid_examples'] or 1) < train_split:
                training_examples.append(example)
            else:
                test_examples.append(example)
    
    # Write spaCy format
    with open(output_train, 'w', encoding='utf-8') as f:
        json.dump(training_examples, f, indent=2, ensure_ascii=False)
    
    with open(output_test, 'w', encoding='utf-8') as f:
        json.dump(test_examples, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"Total resumes processed:      {stats['total_resumes']}")
    print(f"Valid examples created:       {stats['valid_examples']}")
    print(f"Skipped (no entities):        {stats['skipped']}")
    print()
    print(f"Training examples: {len(training_examples)} ({int(train_split*100)}%)")
    print(f"Test examples:     {len(test_examples)} ({int((1-train_split)*100)}%)")
    print()
    print(f"✓ Training data saved to: {output_train}")
    print(f"✓ Test data saved to:     {output_test}")
    
    return training_examples, test_examples, stats

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python convert_to_spacy_v2.py <csv_file> [output_train] [output_test]")
        print("Example: python convert_to_spacy_v2.py data/cleaned500.csv data/train_v2.json data/test_v2.json")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_train = sys.argv[2] if len(sys.argv) > 2 else 'train_v2.json'
    output_test = sys.argv[3] if len(sys.argv) > 3 else 'test_v2.json'
    
    csv_to_spacy_json(csv_file, output_train, output_test)
