"""
LexiScan Auto — NER Inference Engine
======================================
Performs production inference on extracted OCR text using a trained SpaCy
NER model.  Integrates the rule-based post-processing layer from
``rules.validators`` to return clean, validated entities.
"""

import os
import string
from pathlib import Path
from typing import Any, Dict, List

import spacy

from utils.logger import configure_logger
from rules.validators import apply_all_rules, group_entities

logger = configure_logger("LexiScanAuto.NER.Inference")


class NERInference:
    """Wraps SpaCy models for production entity extraction."""

    def __init__(self, model_dir: str = None):
        """Load the serialised NER model and a general English fallback."""
        if model_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
            model_dir = str(base_dir / "models" / "lexiscan_ner")

        logger.info("Initializing Hybrid NER Engines...")
        
        # 1. Try to load the custom fine-tuned model
        self.custom_nlp = None
        if os.path.exists(model_dir):
            try:
                self.custom_nlp = spacy.load(model_dir)
                logger.info(f"Custom LexiScan model loaded from {model_dir}")
            except Exception as e:
                logger.warning(f"Could not load custom model: {e}")

        # 2. Always load the base English model for Zero-Shot/Out-Of-Box mapping
        try:
            self.base_nlp = spacy.load("en_core_web_sm")
            logger.info("Base English model (en_core_web_sm) loaded successfully.")
        except OSError:
            logger.error("en_core_web_sm is not installed. Pipeline will be severely degraded.")
            self.base_nlp = None

    def _is_valid_entity(self, text: str, ent_label: str) -> bool:
        """Validates a predicted span to reject noise and punctuation entities."""
        text = text.strip()
        if not text:
            return False
        if len(text) <= 1 and text in string.punctuation:
            return False
        return True

    def extract_entities_raw(self, text: str) -> List[Dict[str, Any]]:
        """Run NER and return raw entity dicts mapping base entities to target ontology."""
        entities: List[Dict[str, Any]] = []
        found_spans = set()

        # Target valid entities
        valid_custom_labels = {"DATE", "PARTY", "AMOUNT", "JURISDICTION"}

        # Custom Model Priority
        if self.custom_nlp:
            doc_custom = self.custom_nlp(text)
            for ent in doc_custom.ents:
                val = ent.text.strip()
                if self._is_valid_entity(val, ent.label_) and ent.label_ in valid_custom_labels:
                    entities.append({
                        "entity": ent.label_,
                        "value": val,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                    })
                    found_spans.add((ent.start_char, ent.end_char))

        # Base Model Fallback with Ontology Mapping
        if self.base_nlp:
            doc_base = self.base_nlp(text)
            
            # Map SpaCy's default ontology to our legal constraints
            label_map = {
                "ORG": "PARTY",
                "PERSON": "PARTY",
                "MONEY": "AMOUNT",
                "DATE": "DATE",
                "GPE": "JURISDICTION"
            }
            
            for ent in doc_base.ents:
                mapped_label = label_map.get(ent.label_)
                if mapped_label:
                    val = ent.text.strip()
                    # Only add if we haven't already extracted something at this exact location
                    if self._is_valid_entity(val, mapped_label) and (ent.start_char, ent.end_char) not in found_spans:
                        entities.append({
                            "entity": mapped_label,
                            "value": val,
                            "start_char": ent.start_char,
                            "end_char": ent.end_char,
                        })

        return entities

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Run NER **and** rule-based post-processing.

        This is the primary method consumed by the API layer.
        """
        raw = self.extract_entities_raw(text)
        validated = apply_all_rules(raw)
        return validated

    def extract_grouped(self, text: str) -> Dict[str, List[str]]:
        """Run NER + rules and return grouped output.

        Returns
        -------
        dict
            ``{"DATE": [...], "PARTY": [...], "AMOUNT": [...], "JURISDICTION": [...]}``
        """
        validated = self.extract_entities(text)
        return group_entities(validated)


# ──────────────────────────────────────────────────────────────────────────
#  CLI quick-test
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    try:
        engine = NERInference()

        sample_text = (
            "This Confidentiality Agreement is entered into on October 12, 2023, "
            "between Acme Corp (the 'Disclosing Party') and John Doe (the "
            "'Receiving Party'). The penalty amount for breach is $50,000.00 and "
            "falls within the jurisdiction of the state of New York. The agreement "
            "shall terminate on December 31, 2025."
        )

        logger.info(f"Analysing sample text:\n{sample_text}")

        # Detailed output
        entities = engine.extract_entities(sample_text)
        logger.info("═══ Validated Entities ═══")
        for ent in entities:
            logger.info(
                f"  [{ent['entity']:12s}] {ent['value']}  "
                f"(chars {ent['start_char']}–{ent['end_char']})"
            )

        # Grouped output
        grouped = engine.extract_grouped(sample_text)
        logger.info("═══ Grouped Output ═══")
        print(json.dumps(grouped, indent=2))

    except FileNotFoundError as exc:
        logger.error(str(exc))
