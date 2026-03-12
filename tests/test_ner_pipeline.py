from rules.validators import apply_all_rules, group_entities, normalize_amount, normalize_date

def test_normalize_amount():
    assert normalize_amount("$50,000.00") == "50000.00"
    assert normalize_amount("USD 1,234.56") == "1234.56"
    assert normalize_amount("1,000,000") == "1000000"

def test_normalize_date():
    assert normalize_date("Oct 12 2023") == "2023-10-12"
    assert normalize_date("October 12, 2023") == "2023-10-12"
    assert normalize_date("10/12/2023") == "2023-10-12"

def test_apply_all_rules():
    raw_entities = [
        {"entity": "DATE", "value": "Oct 12 2023", "start_char": 0, "end_char": 11},
        {"entity": "AMOUNT", "value": "$50,000.00", "start_char": 20, "end_char": 30},
        {"entity": "PARTY", "value": ")", "start_char": 40, "end_char": 41}, # Noise
    ]
    
    validated = apply_all_rules(raw_entities)
    assert len(validated) == 2
    
    date_ent = next(e for e in validated if e["entity"] == "DATE")
    amount_ent = next(e for e in validated if e["entity"] == "AMOUNT")
    
    assert date_ent["value"] == "2023-10-12"
    assert amount_ent["value"] == "50000.00"

def test_group_entities():
    validated_entities = [
        {"entity": "DATE", "value": "2023-10-12"},
        {"entity": "PARTY", "value": "Acme Corp"},
        {"entity": "PARTY", "value": "John Doe"},
        {"entity": "AMOUNT", "value": "50000.00"},
        {"entity": "JURISDICTION", "value": "New York"}
    ]
    
    grouped = group_entities(validated_entities)
    
    assert "DATE" in grouped
    assert "PARTY" in grouped
    assert "AMOUNT" in grouped
    assert "JURISDICTION" in grouped
    
    assert grouped["DATE"] == ["2023-10-12"]
    assert len(grouped["PARTY"]) == 2
    assert "Acme Corp" in grouped["PARTY"]
    assert grouped["AMOUNT"] == ["50000.00"]
    assert grouped["JURISDICTION"] == ["New York"]
