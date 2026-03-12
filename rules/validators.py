"""
LexiScan Auto — Rule-Based Post-Processing & Validation
=========================================================
Applies deterministic business rules **after** the NER model has
predicted entity spans.  This layer:

* Normalises dates to ISO-8601.
* Validates and normalises monetary amounts.
* Sanitises entity predictions (removes noise, punctuation-only spans).
* Validates date logic (termination ≥ effective).

All functions are pure (no side-effects) and individually testable.
"""

import re
import string
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import configure_logger

logger = configure_logger("LexiScanAuto.Rules")

# ───────────────────────────────────────────────────────────────────────────
#  Date normalisation
# ───────────────────────────────────────────────────────────────────────────

# Common date patterns found in legal contracts
_DATE_PATTERNS: List[Tuple[str, str]] = [
    # Month DD, YYYY  — "October 12, 2023"
    (r"(?:January|February|March|April|May|June|July|August|September|"
     r"October|November|December)\s+\d{1,2},?\s+\d{4}", "%B %d, %Y"),
    # Mon DD YYYY     — "Oct 12 2023"
    (r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}",
     "%b %d %Y"),
    # Mon DD, YYYY    — "Oct 12, 2023"
    (r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}",
     "%b %d, %Y"),
    # DD/MM/YYYY or MM/DD/YYYY — "12/10/2023"
    (r"\d{1,2}/\d{1,2}/\d{4}", "%m/%d/%Y"),
    # YYYY-MM-DD      — already ISO
    (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
    # DD-MM-YYYY
    (r"\d{1,2}-\d{1,2}-\d{4}", "%d-%m-%Y"),
    # DD.MM.YYYY
    (r"\d{1,2}\.\d{1,2}\.\d{4}", "%d.%m.%Y"),
]


def _try_parse_date(raw: str) -> Optional[datetime]:
    """Attempt to parse *raw* into a ``datetime`` using known patterns."""
    raw = raw.strip().replace(",", ", ").replace("  ", " ")
    # Remove ordinal suffixes: 12th, 1st, 2nd, 3rd
    raw = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", raw)

    for pattern, fmt in _DATE_PATTERNS:
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            try:
                return datetime.strptime(match.group(), fmt)
            except ValueError:
                continue

    # Last resort — dateutil
    try:
        from dateutil import parser as dateutil_parser
        return dateutil_parser.parse(raw, dayfirst=False)
    except Exception:
        return None


def normalize_date(raw: str) -> str:
    """Convert a date string to ISO-8601 format (``YYYY-MM-DD``).

    If parsing fails, the original string is returned unchanged.

    Examples
    --------
    >>> normalize_date("Oct 12 2023")
    '2023-10-12'
    >>> normalize_date("October 12, 2023")
    '2023-10-12'
    """
    dt = _try_parse_date(raw)
    if dt:
        return dt.strftime("%Y-%m-%d")
    return raw.strip()


def validate_dates(
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalise all DATE entities to ISO-8601 and flag logical errors.

    Business rule: *termination date* must not precede *effective date*.
    If both named dates are present we log a warning (but still keep the
    entities — the caller decides whether to drop them).
    """
    effective: Optional[datetime] = None
    termination: Optional[datetime] = None

    for ent in entities:
        if ent.get("entity") != "DATE":
            continue

        raw = ent["value"]
        iso = normalize_date(raw)
        ent["value"] = iso

        low = raw.lower()
        dt = _try_parse_date(iso)

        if "effective" in low or "commence" in low or "start" in low:
            effective = dt
        elif "terminat" in low or "expir" in low or "end" in low:
            termination = dt

    if effective and termination and termination < effective:
        logger.warning(
            f"Logical date violation: termination ({termination.date()}) "
            f"precedes effective ({effective.date()})."
        )

    return entities


# ───────────────────────────────────────────────────────────────────────────
#  Amount normalisation
# ───────────────────────────────────────────────────────────────────────────

_CURRENCY_SYMBOLS = re.compile(r"[£€$¥₹₦]")
_THOUSANDS_SEP = re.compile(r"(?<=\d),(?=\d{3})")


def normalize_amount(raw: str) -> str:
    """Convert a currency string to a clean numeric value.

    Examples
    --------
    >>> normalize_amount("$50,000.00")
    '50000.00'
    >>> normalize_amount("USD 1,234,567.89")
    '1234567.89'
    >>> normalize_amount("€ 2.500,00")  # European format
    '2500.00'
    """
    text = raw.strip()
    # Strip currency symbols and currency codes
    text = _CURRENCY_SYMBOLS.sub("", text)
    text = re.sub(r"\b(?:USD|EUR|GBP|INR|CAD|AUD|JPY)\b", "", text, flags=re.IGNORECASE)
    text = text.strip()

    # Detect European format: "2.500,00" → period as thousands, comma as decimal
    if re.search(r"\d\.\d{3},\d{2}$", text):
        text = text.replace(".", "").replace(",", ".")
    else:
        # Standard format: strip commas
        text = _THOUSANDS_SEP.sub("", text)

    # Remove any remaining non-numeric characters except the decimal point
    text = re.sub(r"[^\d.]", "", text)

    # Validate that we have a sensible number
    try:
        value = float(text)
        # Return with 2 decimal places if it has a decimal, else as integer-like
        if "." in text:
            return f"{value:.2f}"
        return str(int(value))
    except ValueError:
        return raw.strip()


def normalize_amounts(
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalise every AMOUNT entity value."""
    for ent in entities:
        if ent.get("entity") == "AMOUNT":
            ent["value"] = normalize_amount(ent["value"])
    return entities


# ───────────────────────────────────────────────────────────────────────────
#  Entity sanitisation
# ───────────────────────────────────────────────────────────────────────────

def _is_noise(text: str) -> bool:
    """Return *True* if the span is punctuation-only or trivially short."""
    stripped = text.strip()
    if not stripped:
        return True
    # All punctuation
    if all(ch in string.punctuation for ch in stripped):
        return True
    # Single character that isn't alphanumeric
    if len(stripped) == 1 and not stripped.isalnum():
        return True
    return False


def sanitize_entities(
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove noisy or invalid entity predictions.

    Removes:
    * Empty strings.
    * Punctuation-only spans (e.g. ``")"``).
    * Whitespace-only spans.
    """
    clean = []
    for ent in entities:
        if _is_noise(ent.get("value", "")):
            logger.debug(f"Sanitised out noisy entity: {ent}")
            continue
        # Strip leading/trailing whitespace from all values
        ent["value"] = ent["value"].strip()
        clean.append(ent)
    return clean


# ───────────────────────────────────────────────────────────────────────────
#  Master pipeline
# ───────────────────────────────────────────────────────────────────────────

def apply_all_rules(
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run the full post-processing pipeline on a list of entities.

    Pipeline order:
    1. Sanitise (remove noise).
    2. Normalise dates → ISO-8601.
    3. Normalise amounts → numeric.
    4. Validate date logic.
    """
    entities = sanitize_entities(entities)
    entities = validate_dates(entities)
    entities = normalize_amounts(entities)
    return entities


def group_entities(
    entities: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """Group validated entities by label into the final response format.

    Returns
    -------
    dict
        ``{"DATE": [...], "PARTY": [...], "AMOUNT": [...], "JURISDICTION": [...]}``
    """
    grouped: Dict[str, List[str]] = {
        "DATE": [],
        "PARTY": [],
        "AMOUNT": [],
        "JURISDICTION": [],
    }
    for ent in entities:
        label = ent.get("entity", "")
        value = ent.get("value", "")
        if label in grouped and value and value not in grouped[label]:
            grouped[label].append(value)
    return grouped


# ───────────────────────────────────────────────────────────────────────────
#  CLI quick-test
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_entities = [
        {"entity": "DATE", "value": "Oct 12 2023", "start_char": 0, "end_char": 11},
        {"entity": "DATE", "value": "October 12, 2023", "start_char": 20, "end_char": 36},
        {"entity": "AMOUNT", "value": "$50,000.00", "start_char": 50, "end_char": 60},
        {"entity": "AMOUNT", "value": "USD 1,234.56", "start_char": 65, "end_char": 77},
        {"entity": "PARTY", "value": "Acme Corp", "start_char": 80, "end_char": 89},
        {"entity": "PARTY", "value": ")", "start_char": 90, "end_char": 91},
        {"entity": "JURISDICTION", "value": "New York", "start_char": 100, "end_char": 108},
        {"entity": "PARTY", "value": "", "start_char": 110, "end_char": 110},
    ]

    validated = apply_all_rules(sample_entities)
    grouped = group_entities(validated)

    import json
    print(json.dumps(grouped, indent=2))
