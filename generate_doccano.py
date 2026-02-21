import json
import uuid
import random

# A more robust simulated Doccano JSONL export with enough variation for spaCy to learn.
templates = [
    ("This Confidentiality Agreement is entered into on {date}, between {party1} and {party2}. The penalty amount for breach is {amount} and falls within the jurisdiction of the state of {jurisdiction}."),
    ("This Agreement, dated {date}, is between {party1} and {party2}. The total investment is {amount} to be governed by the laws of {jurisdiction}."),
    ("Signed on {date}, {party1} agrees to pay {amount} to {party2} under the jurisdiction of {jurisdiction}."),
    ("The contract was executed on {date} by {party1} and {party2}. Total damages are capped at {amount} subject to the courts of {jurisdiction}."),
    ("On {date}, an agreement was formed between {party1} and {party2} involving {amount}. Dispute resolution shall occur in {jurisdiction}."),
    ("Effective as of {date}, {party1} shall deliver to {party2} the sum of {amount}. This document is governed by {jurisdiction} laws."),
    ("Between {party1} and {party2}, signed {date}, for the value of {amount}. Jurisdiction is set in {jurisdiction}."),
    ("Made on {date}. {party1} will compensate {party2} with {amount}, compliant with {jurisdiction} state laws.")
]

dates = ["October 12, 2023", "January 1, 2024", "15th March 2022", "2021-05-20", "November 5, 2019", "2nd of July, 2020", "12/12/2022", "01-10-2023"]
parties = ["Acme Corp", "John Doe", "Tech Solutions Inc.", "Global Ventures LLC", "Beta Corp", "Alpha Inc.", "Cyberdyne Systems", "Skynet", "Omega LLC", "Jane Smith", "Initech", "Umbrella Corp", "Stark Industries", "Wayne Enterprises"]
amounts = ["$50,000.00", "€5,000,000", "$10,000", "£2,000,000", "$1,000", "$100,500", "€35,000", "500,000 GBP"]
jurisdictions = ["New York", "California", "Texas", "London", "Delaware", "Nevada", "Florida", "Ontario"]

def generate_record():
    template = random.choice(templates)
    date = random.choice(dates)
    party1 = random.choice(parties)
    party2 = random.choice([p for p in parties if p != party1])
    amount = random.choice(amounts)
    jurisdiction = random.choice(jurisdictions)

    text = template.format(date=date, party1=party1, party2=party2, amount=amount, jurisdiction=jurisdiction)
    
    entities = [
        (date, "DATE"),
        (party1, "PARTY"),
        (party2, "PARTY"),
        (amount, "AMOUNT"),
        (jurisdiction, "JURISDICTION")
    ]
    
    labels = []
    for ent_text, label in entities:
        start = text.find(ent_text)
        if start != -1:
            end = start + len(ent_text)
            labels.append([start, end, label])

    # Sort labels by start index for valid spacy example
    labels = sorted(labels, key=lambda x: x[0])

    return {
        "document_id": str(uuid.uuid4()),
        "text": text,
        "label": labels,
        "ocr_noise_ratio": round(random.uniform(0.01, 0.05), 4)
    }

print("Generating 50 robust mock annotated documents...")
with open('data/annotations/doccano_export.jsonl', 'w', encoding='utf-8') as f:
    for _ in range(50):
        f.write(json.dumps(generate_record()) + '\n')
print("doccano_export.jsonl created.")
