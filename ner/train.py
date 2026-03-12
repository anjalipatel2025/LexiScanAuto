"""
LexiScan Auto — NER Training Pipeline
=======================================
Trains a custom SpaCy NER model to recognise four legal entity types:

    PARTY · DATE · AMOUNT · JURISDICTION

The module supports:
* Loading JSONL annotation files (Doccano / SpaCy format).
* Automatic train / validation split (80 / 20).
* Token-level alignment validation to avoid misaligned spans.
* Configurable epoch count, dropout, and batch sizing.
* Model serialisation to ``models/lexiscan_ner/``.
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import configure_logger

logger = configure_logger("LexiScanAuto.NER.Train")

# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Load a single JSONL annotation file and return SpaCy training tuples.

    Each JSONL record must contain:
    * ``text`` — the document text.
    * ``label`` — a list of ``[start, end, LABEL]`` triples.

    Records with ``ocr_noise_ratio > 0.20`` are automatically skipped.
    """
    data: List[Tuple[str, Dict[str, Any]]] = []

    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(f"{filepath}:{line_no} — invalid JSON: {exc}")
                    continue

                noise = item.get("ocr_noise_ratio", 0)
                if noise > 0.20:
                    logger.warning(
                        f"Skipping doc {item.get('document_id', '?')} "
                        f"(noise={noise:.2f})"
                    )
                    continue

                text = item.get("text", "")
                labels = item.get("label", [])

                if text and labels:
                    entities = [
                        (int(s), int(e), lbl) for s, e, lbl in labels
                    ]
                    data.append((text, {"entities": entities}))

    except FileNotFoundError:
        logger.error(f"Annotation file not found: {filepath}")
    except Exception as exc:
        logger.error(f"Failed to read {filepath}: {exc}")

    return data


def get_train_val_split(
    annotations_dir: Path,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List, List]:
    """Load all ``.jsonl`` files from *annotations_dir* and split 80/20."""
    all_data: List[Tuple[str, Dict[str, Any]]] = []

    for filename in sorted(os.listdir(annotations_dir)):
        if filename.endswith(".jsonl"):
            filepath = annotations_dir / filename
            all_data.extend(load_data(str(filepath)))

    if not all_data:
        logger.warning("No annotated documents found.")
        return [], []

    random.seed(seed)
    random.shuffle(all_data)
    split_idx = max(1, int(len(all_data) * (1 - val_ratio)))

    logger.info(
        f"Split: {split_idx} train / {len(all_data) - split_idx} val "
        f"(total={len(all_data)})"
    )
    return all_data[:split_idx], all_data[split_idx:]


# ---------------------------------------------------------------------------
#  Alignment validation
# ---------------------------------------------------------------------------

def validate_and_format_data(
    nlp: spacy.Language,
    data: List[Tuple[str, Dict[str, Any]]],
) -> Tuple[List, int, Dict[str, int]]:
    """Filter out misaligned spans and return clean training examples.

    Returns
    -------
    tuple
        ``(clean_data, skipped_span_count, entity_distribution_dict)``
    """
    valid_data: List[Tuple[str, Dict[str, Any]]] = []
    skipped = 0
    entity_counts: Dict[str, int] = {}

    for text, annotations in data:
        doc = nlp.make_doc(text)
        entities = annotations.get("entities", [])
        valid_entities: List[Tuple[int, int, str]] = []

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


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

def train_ner(
    train_data: List[Tuple[str, Dict[str, Any]]],
    val_data: List[Tuple[str, Dict[str, Any]]],
    model_dir: str,
    n_iter: int = 30,
    dropout: float = 0.40,
):
    """Train a SpaCy English model with a custom NER component."""
    if not train_data:
        logger.error("No training data provided — aborting.")
        return

    logger.info("Initialising base English SpaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.blank("en")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # ── Validate alignments ──────────────────────────────────────────
    logger.info("Validating training-data offsets...")
    clean_train, skipped_train, train_ent_counts = validate_and_format_data(
        nlp, train_data
    )
    logger.info(
        f"Training validation done — kept {len(clean_train)} docs, "
        f"skipped {skipped_train} spans."
    )
    logger.info(f"Training entity distribution: {train_ent_counts}")

    if not clean_train:
        logger.error("No valid training data after alignment — aborting.")
        return

    if len(clean_train) < 10:
        logger.warning("Very small dataset (<10 docs) — generalisation may be poor.")

    # ── Register labels ──────────────────────────────────────────────
    for _, annotations in clean_train:
        for ent in annotations.get("entities", []):
            ner.add_label(ent[2])

    # ── Freeze non-NER pipes ─────────────────────────────────────────
    pipe_exceptions = {"ner"}
    other_pipes = [p for p in nlp.pipe_names if p not in pipe_exceptions]

    logger.info(
        f"Training for {n_iter} epochs on {len(clean_train)} documents..."
    )
    # Resume training rather than resetting
    optimizer = nlp.resume_training()

    with nlp.disable_pipes(*other_pipes):
        for epoch in range(1, n_iter + 1):
            random.shuffle(clean_train)
            losses: Dict[str, float] = {}
            batches = minibatch(
                clean_train, size=compounding(4.0, 32.0, 1.001)
            )

            for batch in batches:
                examples = []
                for text, anns in batch:
                    doc = nlp.make_doc(text)
                    try:
                        example = Example.from_dict(doc, anns)
                        examples.append(example)
                    except Exception as exc:
                        logger.debug(f"Example creation error: {exc}")

                if examples:
                    nlp.update(examples, sgd=optimizer, drop=dropout, losses=losses)

            ner_loss = losses.get("ner", 0.0)
            logger.info(f"Epoch {epoch:>3}/{n_iter} | NER loss: {ner_loss:.4f}")

    # ── Persist ──────────────────────────────────────────────────────
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_dir)
    logger.info(f"Model saved → {output_dir}")


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

def run_training():
    """Convenience wrapper — loads data from the default paths and trains."""
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

    logger.info(
        f"Loaded {len(train_data)} training and {len(val_data)} validation docs."
    )
    train_ner(train_data, val_data, str(model_output_dir), n_iter=20)


if __name__ == "__main__":
    run_training()
