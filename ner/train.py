import json
import os
import random
from pathlib import Path

import spacy
from spacy.training import Example, offsets_to_biluo_tags
from spacy.util import minibatch, compounding

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import configure_logger

logger = configure_logger("LexiScanAuto.NER.Train")

def load_data(filepath):
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                
                noise = item.get("ocr_noise_ratio", 0)
                if noise > 0.2:
                    logger.warning(f"Skipping document {item.get('document_id')} due to high OCR noise: {noise}")
                    continue
                    
                text = item.get("text", "")
                labels = item.get("label", [])
                
                if labels:
                    data.append((text, {"entities": labels}))
    except Exception as e:
        logger.error(f"Failed to read {filepath}: {e}")
        
    return data

def get_train_val_split(annotations_dir):
    all_data = []
    for filename in os.listdir(annotations_dir):
        if filename.endswith(".jsonl"):
            filepath = annotations_dir / filename
            all_data.extend(load_data(filepath))
            
    if not all_data:
        return [], []
        
    random.seed(42)
    random.shuffle(all_data)
    split_idx = max(1, int(len(all_data) * 0.8))
    
    return all_data[:split_idx], all_data[split_idx:]


def validate_and_format_data(nlp, data):
    valid_data = []
    skipped = 0
    entity_counts = {}
    
    for text, annotations in data:
        doc = nlp.make_doc(text)
        entities = annotations.get("entities", [])
        valid_entities = []
        
        for start, end, label in entities:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                valid_entities.append((span.start_char, span.end_char, label))
                entity_counts[label] = entity_counts.get(label, 0) + 1
            else:
                skipped += 1
                
        if valid_entities:
            valid_data.append((text, {"entities": valid_entities}))
            
    return valid_data, skipped, entity_counts


def train_ner(train_data, val_data, model_dir, n_iter=30):
    if not train_data:
        logger.error("No training data provided!")
        return

    logger.info("Setting up a blank English model for NER...")
    nlp = spacy.blank("en")
    
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Validate data and handle misalignments
    logger.info("Validating training data offsets...")
    clean_train_data, skipped_train, train_ent_counts = validate_and_format_data(nlp, train_data)
    
    logger.info(f"Training data validation complete. Skipped {skipped_train} misaligned samples.")
    logger.info(f"Training Entity Distribution: {train_ent_counts}")
    
    if len(clean_train_data) < 10:
        logger.warning("Dataset size is very small. Model may not generalize well.")

    if not clean_train_data:
        logger.error("No valid training data remaining after alignment check.")
        return

    # Add all unique labels from our dataset
    for _, annotations in clean_train_data:
        for ent in annotations.get("entities", []):
            ner.add_label(ent[2])

    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    
    logger.info(f"Beginning training with {len(clean_train_data)} samples for {n_iter} epochs...")
    optimizer = nlp.begin_training()
    
    with nlp.disable_pipes(*other_pipes):
        for itn in range(n_iter):
            random.shuffle(clean_train_data)
            losses = {}
            batches = minibatch(clean_train_data, size=compounding(4.0, 32.0, 1.001))
            
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    try:
                        example = Example.from_dict(doc, annotations)
                        examples.append(example)
                    except Exception as e:
                        logger.warning(f"Error creating example: {e}")
                nlp.update(examples, sgd=optimizer, drop=0.4, losses=losses)
                
            logger.info(f"Epoch {itn + 1}/{n_iter} | Loss: {losses.get('ner', 0):.4f}")

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_dir)
    logger.info(f"Successfully saved trained NER model to: {output_dir}")

def run_training():
    base_dir = Path(__file__).resolve().parent.parent
    annotations_dir = base_dir / "data" / "annotations"
    model_output_dir = base_dir / "models" / "lexiscan_ner"
    
    if not annotations_dir.exists():
        logger.error(f"Annotations directory not found: {annotations_dir}")
        return
        
    train_data, val_data = get_train_val_split(annotations_dir)
            
    if not train_data:
        logger.error("No valid annotated training data found.")
        return
        
    logger.info(f"Loaded {len(train_data)} training and {len(val_data)} validation documents.")
    
    train_ner(train_data, val_data, model_output_dir, n_iter=20)

if __name__ == "__main__":
    run_training()
