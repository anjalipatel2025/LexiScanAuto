import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """
    Takes text input and returns extracted named entities
    """
    doc = nlp(text)

    entities = []

    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })

    return entities


