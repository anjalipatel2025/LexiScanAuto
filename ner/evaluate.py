import os
from pathlib import Path
import spacy
from spacy.training import Example

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import configure_logger
from ner.train import get_train_val_split, validate_and_format_data

logger = configure_logger("LexiScanAuto.NER.Evaluate")

def evaluate_model(model_dir, test_data):
    if not os.path.exists(model_dir):
        logger.error(f"Model directory {model_dir} does not exist. Please run training first.")
        return

    logger.info(f"Loading trained model from {model_dir}...")
    try:
        nlp = spacy.load(model_dir)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    logger.info("Validating validation data offsets...")
    clean_test_data, skipped_test, test_ent_counts = validate_and_format_data(nlp, test_data)
    logger.info(f"Validation data validation complete. Skipped {skipped_test} misaligned samples.")
    logger.info(f"Validation Entity Distribution: {test_ent_counts}")

    if not clean_test_data:
        logger.error("No valid validation data remaining.")
        return

    examples = []
    for text, annotations in clean_test_data:
        doc = nlp.make_doc(text)
        try:
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        except Exception:
            pass

    logger.info("Evaluating model against validation dataset...")
    scorer = nlp.evaluate(examples)
    
    logger.info("=== Overall Performance ===")
    logger.info(f"Precision : {scorer['ents_p']:.4f}")
    logger.info(f"Recall    : {scorer['ents_r']:.4f}")
    logger.info(f"F1-Score  : {scorer['ents_f']:.4f}")
    
    logger.info("=== Per-Entity Evaluation ===")
    target_entities = ['DATE', 'AMOUNT', 'PARTY', 'JURISDICTION']
    
    for ent_type in target_entities:
        if ent_type in scorer['ents_per_type']:
            metrics = scorer['ents_per_type'][ent_type]
            logger.info(f"{ent_type:12s} - Precision: {metrics['p']:.4f}  |  Recall: {metrics['r']:.4f}  |  F1: {metrics['f']:.4f}")
        else:
            logger.warning(f"{ent_type:12s} - No evaluation data found for this entity.")


def run_evaluation():
    base_dir = Path(__file__).resolve().parent.parent
    annotations_dir = base_dir / "data" / "annotations"
    model_dir = base_dir / "models" / "lexiscan_ner"
    
    # Evaluate ONLY on the validation set to prevent leakage
    train_data, val_data = get_train_val_split(annotations_dir)
                
    if not val_data:
        logger.error("No valid data found to evaluate model against.")
        return
        
    evaluate_model(model_dir, val_data)

if __name__ == "__main__":
    run_evaluation()
