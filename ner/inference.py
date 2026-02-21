import os
import string
from pathlib import Path
import spacy

from utils.logger import configure_logger

logger = configure_logger("LexiScanAuto.NER.Inference")

class NERInference:
    """Class to perform production inference on extracted OCR text."""
    
    def __init__(self, model_dir=None):
        if model_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
            model_dir = base_dir / "models" / "lexiscan_ner"
            
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory missing: {model_dir}. Please train the model first.")
        
        logger.info(f"Loading LexiScan NER model from: {model_dir}...")
        self.nlp = spacy.load(model_dir)
        logger.info("Model securely loaded and ready for inference.")
        
    def _is_valid_entity(self, text: str, ent_label: str) -> bool:
        """Validates a predicted span to reject noise and punctuation entities."""
        text = text.strip()
        
        # Reject empty strings
        if not text:
            return False
            
        # Reject single character punctuation entities (like ")", "-", etc.)
        if len(text) <= 1 and text in string.punctuation:
            return False
            
        return True

    def extract_entities(self, text: str):
        """Passes text through the NLP pipeline and returns structured entities."""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if self._is_valid_entity(ent.text, ent.label_):
                entities.append({
                    "entity": ent.label_,
                    "value": ent.text.strip(),
                    "start_char": ent.start_char,
                    "end_char": ent.end_char
                })
            else:
                logger.debug(f"Rejected invalid entity prediction: '{ent.text}' [{ent.label_}]")
            
        return entities

if __name__ == "__main__":
    try:
        inference_engine = NERInference()
        
        sample_text = """
        This Confidentiality Agreement is entered into on October 12, 2023, 
        between Acme Corp (the "PARTY") and John Doe (the "PARTY"). 
        The penalty amount for breach is $50,000.00 and falls within the 
        jurisdiction of the state of New York.
        """
        
        logger.info(f"Analyzing sample text:\n{sample_text.strip()}")
        results = inference_engine.extract_entities(sample_text)
        
        logger.info("=== Extracted Entities ===")
        if not results:
            logger.info("No entities found.")
        else:
            for item in results:
                logger.info(f"[{item['entity']:12s}] {item['value']} (Ch: {item['start_char']}-{item['end_char']})")
                
    except FileNotFoundError as e:
        logger.error(str(e))
