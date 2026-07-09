"""
Phase 2, Step 4: Train spaCy NER Model
Input: train.json, test.json (from Step 3)
Output: Trained NER model at /models/ner_model/
"""

import json
import random
from pathlib import Path
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

# ============================================================================
# CONFIG
# ============================================================================
TRAIN_JSON_PATH = Path("data/processed/train.json")
TEST_JSON_PATH = Path("data/processed/test.json")
MODEL_OUTPUT_DIR = Path("models/ner_model")
ITERATIONS = 30
DROP = 0.5  # Dropout

# ============================================================================
# FUNCTION: REMOVE OVERLAPPING ENTITIES
# ============================================================================
def remove_overlapping_entities(entities):
    """
    Remove overlapping entities, keeping the longest one for each position.
    entities: list of (start, end, label) tuples
    """
    if not entities:
        return []
    
    # Sort by start position, then by end position (descending)
    sorted_ents = sorted(entities, key=lambda x: (x[0], -x[1]))
    
    filtered = []
    for start, end, label in sorted_ents:
        # Check if this entity overlaps with any already-filtered entity
        overlaps = False
        for f_start, f_end, f_label in filtered:
            # Check for overlap
            if not (end <= f_start or start >= f_end):
                overlaps = True
                break
        
        if not overlaps:
            filtered.append((start, end, label))
    
    return filtered

# ============================================================================
# STEP 1: LOAD TRAINING DATA
# ============================================================================
print("[1/6] Loading training data...")
try:
    with open(TRAIN_JSON_PATH, 'r', encoding='utf-8') as f:
        train_data_list = json.load(f)
    print(f"✓ Loaded {len(train_data_list)} training examples")
except FileNotFoundError:
    print(f"❌ ERROR: Could not find {TRAIN_JSON_PATH}")
    exit(1)

try:
    with open(TEST_JSON_PATH, 'r', encoding='utf-8') as f:
        test_data_list = json.load(f)
    print(f"✓ Loaded {len(test_data_list)} test examples")
except FileNotFoundError:
    print(f"❌ ERROR: Could not find {TEST_JSON_PATH}")
    exit(1)

# ============================================================================
# STEP 2: CONVERT TO SPACY EXAMPLE FORMAT (with overlap removal)
# ============================================================================
print("\n[2/6] Converting to spaCy Example format...")

train_examples = []
overlaps_removed_train = 0

for entry in train_data_list:
    text = entry["text"]
    entities = entry["entities"]
    
    # Remove overlapping entities
    clean_entities = remove_overlapping_entities(entities)
    if len(clean_entities) < len(entities):
        overlaps_removed_train += len(entities) - len(clean_entities)
    
    # Create a Doc object
    nlp_blank = spacy.blank("en")
    doc = nlp_blank.make_doc(text)
    
    # Create an Example
    example = Example.from_dict(doc, {"entities": clean_entities})
    train_examples.append(example)

test_examples = []
overlaps_removed_test = 0

for entry in test_data_list:
    text = entry["text"]
    entities = entry["entities"]
    
    # Remove overlapping entities
    clean_entities = remove_overlapping_entities(entities)
    if len(clean_entities) < len(entities):
        overlaps_removed_test += len(entities) - len(clean_entities)
    
    nlp_blank = spacy.blank("en")
    doc = nlp_blank.make_doc(text)
    
    example = Example.from_dict(doc, {"entities": clean_entities})
    test_examples.append(example)

print(f"✓ Converted {len(train_examples)} training examples")
if overlaps_removed_train > 0:
    print(f"  (Removed {overlaps_removed_train} overlapping entities from training)")
print(f"✓ Converted {len(test_examples)} test examples")
if overlaps_removed_test > 0:
    print(f"  (Removed {overlaps_removed_test} overlapping entities from test)")

# ============================================================================
# STEP 3: CREATE BLANK MODEL & ADD NER COMPONENT
# ============================================================================
print("\n[3/6] Creating blank spaCy model...")
nlp = spacy.blank("en")
print("✓ Created blank English model")

# Add NER component
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
    print("✓ Added NER component")
else:
    ner = nlp.get_pipe("ner")
    print("✓ NER component already exists")

# Add labels
labels = ["SKILL", "JOB_TITLE", "EXPERIENCE_DURATION", "DEGREE", "INSTITUTION", "CERTIFICATION"]
for label in labels:
    ner.add_label(label)

print(f"✓ Added entity labels: {', '.join(labels)}")

# Initialize the NER component with the training examples
print("\n[3.5/6] Initializing NER component...")
nlp.initialize(lambda: train_examples)
print("✓ NER component initialized")

# ============================================================================
# STEP 4: TRAIN MODEL
# ============================================================================
print("\n[4/6] Training NER model ({} iterations)...".format(ITERATIONS))
print("  (This may take 1-5 minutes)\n")

# Training loop
for iteration in range(ITERATIONS):
    random.shuffle(train_examples)
    
    losses = {}
    
    for example in train_examples:
        nlp.update(
            [example],
            drop=DROP,
            losses=losses
        )
    
    # Print progress every 5 iterations
    if (iteration + 1) % 5 == 0:
        print(f"  Iteration {iteration + 1}/{ITERATIONS} — Loss: {losses.get('ner', 0):.4f}")

print(f"\n✓ Training complete!")

# ============================================================================
# STEP 5: EVALUATE ON TEST SET
# ============================================================================
print("\n[5/6] Evaluating on test set...")

tp = 0  # true positives
fp = 0  # false positives
fn = 0  # false negatives

for example in test_examples:
    pred_doc = nlp(example.text)
    
    # Extract predicted and gold entities
    pred_ents = set((ent.start_char, ent.end_char, ent.label_) for ent in pred_doc.ents)
    gold_ents = set((ent.start_char, ent.end_char, ent.label_) for ent in example.reference.ents)
    
    # Calculate TP, FP, FN
    tp += len(pred_ents & gold_ents)
    fp += len(pred_ents - gold_ents)
    fn += len(gold_ents - pred_ents)

# Calculate metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"✓ Test Set Performance:")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1:.4f}")
print(f"  TP: {tp}, FP: {fp}, FN: {fn}")

# ============================================================================
# STEP 6: SAVE MODEL
# ============================================================================
print("\n[6/6] Saving trained model...")

# Create output directory
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save model
nlp.to_disk(MODEL_OUTPUT_DIR)
print(f"✓ Saved model to: {MODEL_OUTPUT_DIR}")

# Save metrics
metrics = {
    "iterations": ITERATIONS,
    "training_examples": len(train_examples),
    "test_examples": len(test_examples),
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "entity_labels": labels,
    "overlaps_removed_train": overlaps_removed_train,
    "overlaps_removed_test": overlaps_removed_test
}

metrics_path = MODEL_OUTPUT_DIR / "metrics.json"
with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2)

print(f"✓ Saved metrics to: {metrics_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✓ PHASE 2, STEP 4 COMPLETE!")
print("="*70)
print()
print("MODEL SUMMARY:")
print(f"  Training examples: {len(train_examples)}")
print(f"  Test examples: {len(test_examples)}")
print(f"  Iterations: {ITERATIONS}")
print(f"  Entity labels: {len(labels)}")
print()
print("TEST SET PERFORMANCE:")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1:.4f}")
print()
print("MODEL LOCATION:")
print(f"  {MODEL_OUTPUT_DIR}/")
print()
print("✓ Model ready for inference!")
print("="*70)