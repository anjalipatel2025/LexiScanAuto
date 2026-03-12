"""
LexiScan Auto — NER Model Evaluation
======================================
Evaluates a trained SpaCy NER model on a held-out validation set and
reports:

* **Overall** Precision, Recall, F1-Score.
* **Per-entity** metrics focused on DATE, AMOUNT, PARTY, JURISDICTION.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import spacy
from spacy.training import Example

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import configure_logger
from ner.train import get_train_val_split, validate_and_format_data

logger = configure_logger("LexiScanAuto.NER.Evaluate")


def evaluate_model(
    model_dir: str,
    test_data: List[Tuple[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Load a trained model and evaluate it against *test_data*.

    Parameters
    ----------
    model_dir : str | Path
        Path to the serialised SpaCy model.
    test_data : list
        SpaCy-format examples ``(text, {"entities": [...]})``

    Returns
    -------
    dict
        The full SpaCy ``Scorer`` results dictionary.
    """
    if not os.path.exists(model_dir):
        logger.error(
            f"Model directory {model_dir} does not exist. "
            "Please run training first."
        )
        return {}

    logger.info(f"Loading trained model from {model_dir}...")
    try:
        nlp = spacy.load(model_dir)
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        return {}

    # ── Validate alignments in the test set ──────────────────────────
    logger.info("Validating test-data offsets...")
    clean_test, skipped, ent_counts = validate_and_format_data(nlp, test_data)
    logger.info(
        f"Test-data validation: kept {len(clean_test)} docs, "
        f"skipped {skipped} misaligned spans."
    )
    logger.info(f"Test entity distribution: {ent_counts}")

    if not clean_test:
        logger.error("No valid test data remaining after alignment check.")
        return {}

    # ── Build Example objects ────────────────────────────────────────
    examples = []
    for text, annotations in clean_test:
        doc = nlp.make_doc(text)
        try:
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        except Exception:
            pass

    # ── Evaluate ─────────────────────────────────────────────────────
    logger.info(f"Evaluating model on {len(examples)} examples...")
    scores = nlp.evaluate(examples)

    logger.info("═══ Overall Performance ═══")
    logger.info(f"  Precision : {scores.get('ents_p', 0):.4f}")
    logger.info(f"  Recall    : {scores.get('ents_r', 0):.4f}")
    logger.info(f"  F1-Score  : {scores.get('ents_f', 0):.4f}")

    logger.info("═══ Per-Entity Evaluation ═══")
    target_entities = ["DATE", "AMOUNT", "PARTY", "JURISDICTION"]
    per_type = scores.get("ents_per_type", {})

    for ent_type in target_entities:
        if ent_type in per_type:
            m = per_type[ent_type]
            logger.info(
                f"  {ent_type:12s}  P={m['p']:.4f}  R={m['r']:.4f}  F1={m['f']:.4f}"
            )
        else:
            logger.warning(f"  {ent_type:12s}  — no evaluation data.")

    return scores


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

def run_evaluation():
    """Convenience wrapper — evaluates the default model on the val split."""
    base_dir = Path(__file__).resolve().parent.parent
    annotations_dir = base_dir / "data" / "annotations"
    model_dir = base_dir / "models" / "lexiscan_ner"

    _, val_data = get_train_val_split(annotations_dir)

    if not val_data:
        logger.error("No validation data found to evaluate against.")
        return

    evaluate_model(str(model_dir), val_data)


if __name__ == "__main__":
    run_evaluation()