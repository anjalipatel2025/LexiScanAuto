import re
from datetime import datetime

def standardize_date(date_text):
    try:
        parsed = datetime.strptime(date_text, "%d/%m/%Y")
        return parsed.strftime("%Y-%m-%d")
    except:
        return date_text

def clean_amount(amount_text):
    cleaned = re.sub(r"[^\d.]", "", amount_text)
    return cleaned

def apply_rules(entities):
    processed = []

    for ent in entities:
        if ent["label"] == "DATE":
            ent["text"] = standardize_date(ent["text"])
        elif ent["label"] == "MONEY":
            ent["text"] = clean_amount(ent["text"])

        processed.append(ent)

    return processed


if __name__ == "__main__":
    sample_entities = [
        {"text": "12/03/2023", "label": "DATE"},
        {"text": "$5,000", "label": "MONEY"}
    ]

    result = apply_rules(sample_entities)
    print(result)
