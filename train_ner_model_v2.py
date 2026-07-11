"""
Phase 2.5: Retrain NER Model with Improvements
- Pretrained model: en_core_web_sm (instead of blank)
- Iterations: 50-100 (instead of 30)
- Dataset: 500 resumes (instead of 203)
- Better evaluation: per-entity metrics
"""

import json
import spacy
from pathlib import Path
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import sys

# ============================================================================
# SETUP
# ============================================================================

def load_training_data(train_json, test_json):
    """Load spaCy JSON format data"""
    with open(train_json, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    return train_data, test_data

def create_examples(data, nlp):
    """Convert data to spaCy Example objects"""
    examples = []
    for item in data:
        doc = nlp.make_doc(item['text'])
        ents = []
        for start, end, label in item['entities']:
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)
        doc.ents = ents
        example = Example.from_dict(doc, {"entities": item['entities']})
        examples.append(example)
    return examples

# ============================================================================
# TRAINING
# ============================================================================

def train_ner_model(train_json, test_json, model_output_dir, n_iterations=50, use_pretrained=True):
    """
    Train NER model with improved settings
    
    Args:
        train_json: Path to training data (spaCy JSON format)
        test_json: Path to test data
        model_output_dir: Where to save trained model
        n_iterations: Number of training iterations (50-100)
        use_pretrained: Use en_core_web_sm (True) or blank model (False)
    """
    
    print("=" * 70)
    print("PHASE 2.5: NER MODEL RETRAINING")
    print("=" * 70)
    print(f"Pretrained model: {use_pretrained}")
    print(f"Iterations: {n_iterations}")
    print()
    
    # Load data
    print("[1/5] Loading training data...")
    train_data, test_data = load_training_data(train_json, test_json)
    print(f"  ✓ Training examples: {len(train_data)}")
    print(f"  ✓ Test examples: {len(test_data)}")
    print()
    
    # Load or create model
    print("[2/5] Loading language model...")
    if use_pretrained:
        print("  → Using pretrained model: en_core_web_sm")
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("  ⚠ en_core_web_sm not found. Downloading...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            nlp = spacy.load('en_core_web_sm')
    else:
        print("  → Using blank model")
        nlp = spacy.blank('en')
    
    # Add NER component if not present
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner')
    else:
        ner = nlp.get_pipe('ner')
    
    # Add entity labels
    entity_labels = set()
    for item in train_data + test_data:
        for _, _, label in item['entities']:
            entity_labels.add(label)
    
    for label in entity_labels:
        ner.add_label(label)
    
    print(f"  ✓ Entity labels: {sorted(entity_labels)}")
    print()
    
    # Create training examples
    print("[3/5] Converting data to spaCy format...")
    train_examples = create_examples(train_data, nlp)
    test_examples = create_examples(test_data, nlp)
    print(f"  ✓ Created {len(train_examples)} training examples")
    print(f"  ✓ Created {len(test_examples)} test examples")
    print()
    
    # Training
    print(f"[4/5] Training NER model ({n_iterations} iterations)...")
    print("  (This may take 3-10 minutes)")
    print()
    
    # Disable other components for faster training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        # Initialize the NER component
        optimizer = nlp.create_optimizer()
        
        # Training loop
        for iteration in range(1, n_iterations + 1):
            random.shuffle(train_examples)
            losses = {}
            
            # Mini-batch training
            for batch in minibatch(train_examples, size=8):
                nlp.update(
                    batch,
                    drop=0.5,  # Dropout rate
                    sgd=optimizer,
                    losses=losses
                )
            
            if iteration % 5 == 0 or iteration == 1:
                print(f"  Iteration {iteration}/{n_iterations} — Loss: {losses.get('ner', 0):.4f}")
    
    print("  ✓ Training complete!")
    print()
    
    # Evaluation
    print("[5/5] Evaluating on test set...")
    metrics = evaluate_model(nlp, test_examples)
    print_evaluation_metrics(metrics)
    
    # Save model
    print()
    print("Saving trained model...")
    output_path = Path(model_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_path)
    print(f"  ✓ Saved model to: {output_path}")
    
    # Save metrics
    metrics_file = output_path / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Saved metrics to: {metrics_file}")
    
    print()
    print("=" * 70)
    print("✓ PHASE 2.5 TRAINING COMPLETE!")
    print("=" * 70)
    
    return nlp, metrics

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(nlp, examples):
    """
    Evaluate model on test set.
    Returns per-entity and overall metrics.
    """
    tp = {}  # True Positives per entity type
    fp = {}  # False Positives per entity type
    fn = {}  # False Negatives per entity type
    
    all_tp = 0
    all_fp = 0
    all_fn = 0
    
    for example in examples:
        pred_ents = nlp(example.text).ents
        gold_ents = example.reference.ents
        
        # Convert to sets of (start, end, label) for comparison
        pred_set = {(ent.start_char, ent.end_char, ent.label_) for ent in pred_ents}
        gold_set = {(ent.start_char, ent.end_char, ent.label_) for ent in gold_ents}
        
        # Calculate per-entity metrics
        for label in set([e[2] for e in pred_set | gold_set]):
            pred_label = {e for e in pred_set if e[2] == label}
            gold_label = {e for e in gold_set if e[2] == label}
            
            tp_label = len(pred_label & gold_label)
            fp_label = len(pred_label - gold_label)
            fn_label = len(gold_label - pred_label)
            
            tp[label] = tp.get(label, 0) + tp_label
            fp[label] = fp.get(label, 0) + fp_label
            fn[label] = fn.get(label, 0) + fn_label
            
            all_tp += tp_label
            all_fp += fp_label
            all_fn += fn_label
    
    # Calculate metrics
    metrics = {
        'overall': calculate_metrics(all_tp, all_fp, all_fn),
        'per_entity': {}
    }
    
    for label in tp.keys() | fp.keys() | fn.keys():
        metrics['per_entity'][label] = calculate_metrics(
            tp.get(label, 0),
            fp.get(label, 0),
            fn.get(label, 0)
        )
    
    return metrics

def calculate_metrics(tp, fp, fn):
    """Calculate Precision, Recall, F1"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def print_evaluation_metrics(metrics):
    """Pretty print evaluation metrics"""
    print()
    print("  OVERALL METRICS:")
    overall = metrics['overall']
    print(f"    Precision: {overall['precision']:.4f}")
    print(f"    Recall:    {overall['recall']:.4f}")
    print(f"    F1-Score:  {overall['f1']:.4f}")
    print(f"    (TP: {overall['tp']}, FP: {overall['fp']}, FN: {overall['fn']})")
    
    if metrics['per_entity']:
        print()
        print("  PER-ENTITY METRICS:")
        for label in sorted(metrics['per_entity'].keys()):
            m = metrics['per_entity'][label]
            print(f"    {label:20} P:{m['precision']:.4f} R:{m['recall']:.4f} F1:{m['f1']:.4f}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Default paths
    train_json = 'data/train_v2.json'
    test_json = 'data/test_v2.json'
    model_output = 'models/ner_model_v2'
    n_iterations = 50
    use_pretrained = True
    
    # Parse arguments
    if len(sys.argv) > 1:
        n_iterations = int(sys.argv[1])
    if len(sys.argv) > 2:
        use_pretrained = sys.argv[2].lower() == 'true'
    
    # Train
    nlp, metrics = train_ner_model(
        train_json,
        test_json,
        model_output,
        n_iterations=n_iterations,
        use_pretrained=use_pretrained
    )
