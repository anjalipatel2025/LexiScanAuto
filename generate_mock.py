import json

mock_texts = [
    {
        "text": "This Confidentiality Agreement is entered into on October 12, 2023, between Acme Corp and John Doe. The penalty amount for breach is $50,000.00 and falls within the jurisdiction of the state of New York.",
        "entities": [
            ("October 12, 2023", "DATE"),
            ("Acme Corp", "PARTY"),
            ("John Doe", "PARTY"),
            ("$50,000.00", "AMOUNT"),
            ("New York", "JURISDICTION")
        ]
    },
    {
        "text": "This Agreement, dated January 1, 2024, is between Tech Solutions Inc. and Global Ventures LLC. The total investment is €5,000,000 to be governed by the laws of California.",
        "entities": [
            ("January 1, 2024", "DATE"),
            ("Tech Solutions Inc.", "PARTY"),
            ("Global Ventures LLC", "PARTY"),
            ("€5,000,000", "AMOUNT"),
            ("California", "JURISDICTION")
        ]
    },
    {
        "text": "Signed on 15th March 2022, Beta Corp agrees to pay $10,000 to Alpha Inc. under the jurisdiction of Texas.",
        "entities": [
            ("15th March 2022", "DATE"),
            ("Beta Corp", "PARTY"),
            ("$10,000", "AMOUNT"),
            ("Alpha Inc.", "PARTY"),
            ("Texas", "JURISDICTION")
        ]
    },
    {
        "text": "The contract was executed on 2021-05-20 by Cyberdyne Systems and Skynet. Total damages are capped at £2,000,000 subject to the courts of London.",
        "entities": [
            ("2021-05-20", "DATE"),
            ("Cyberdyne Systems", "PARTY"),
            ("Skynet", "PARTY"),
            ("£2,000,000", "AMOUNT"),
            ("London", "JURISDICTION")
        ]
    }
]

import uuid

with open('data/annotations/mock_train.jsonl', 'w') as f:
    for item in mock_texts:
        text = item["text"]
        labels = []
        for ent_text, label in item["entities"]:
            start = text.find(ent_text)
            if start != -1:
                end = start + len(ent_text)
                labels.append([start, end, label])
        
        record = {
            "document_id": str(uuid.uuid4()),
            "text": text,
            "label": labels,
            "ocr_noise_ratio": 0.05
        }
        f.write(json.dumps(record) + '\n')
