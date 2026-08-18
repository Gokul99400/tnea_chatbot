"""
chatbot.py  —  TNEA Engineering College Counselling Chatbot

ARCHITECTURE
  Python layer  : state, parsing, filtering, ranking, table generation, guided flow
  OpenRouter    : general TNEA questions, college explanations, conversational response

RULES
  - NEVER send full data.txt to OpenRouter
  - NEVER use LLM to generate recommendations
  - NEVER invent college names, codes, cutoffs, fees, placements, or seats
  - Category (community) is REQUIRED before showing recommendations
  - is_any() MUST be used for every filter check — never `if value:`
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI


# ================================================================
# CONFIGURATION
# ================================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
DATA_FILE_ENV      = os.getenv("TNEA_DATA_FILE", "")
MAX_HISTORY        = 20  # max conversation turns sent to LLM


# ================================================================
# OPENROUTER CLIENT
# ================================================================

client = (
    OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    if OPENROUTER_API_KEY
    else None
)


# ================================================================
# DATA FILE DISCOVERY
# ================================================================

def _find_data_file() -> Optional[Path]:
    base = Path(__file__).resolve().parent
    if DATA_FILE_ENV:
        p = Path(DATA_FILE_ENV)
        p = p if p.is_absolute() else base / p
        if p.exists():
            return p
    for name in ["data.txt", "colleges.txt", "tnea_data.txt", "college_data.txt"]:
        p = base / name
        if p.exists():
            return p
    others = [f for f in base.glob("*.txt") if f.name.lower() != "requirements.txt"]
    return others[0] if others else None


DATA_FILE = _find_data_file()


# ================================================================
# RECORD PARSING
# ================================================================

def _norm_community(val: str) -> str:
    """Normalise community label to canonical uppercase form."""
    v = val.strip().upper()
    return {"MBC/DNC": "MBC", "MBCDNC": "MBC", "BCGM": "BCM", "BCG": "BC"}.get(v, v)


def parse_records(raw: str) -> List[Dict]:
    """
    Split raw text on '---' separators and parse each chunk into a dict.
    Returns only records that have at least: code, name, branch.
    """
    records: List[Dict] = []
    chunks = re.split(r"\n-{3,}\n", raw)

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        rec: Dict = {}
        for line in chunk.splitlines():
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            if   key == "College Code":              rec["code"]      = val
            elif key == "College Name and Location": rec["name"]      = val
            elif key == "District":                  rec["district"]  = val
            elif key == "Branch":                    rec["branch"]    = val
            elif key == "Admission Year":            rec["year"]      = val
            elif key == "Community":                 rec["community"] = _norm_community(val)
            elif key == "Closing Cutoff Mark":
                try:    rec["cutoff"] = float(val)
                except: rec["cutoff"] = None  # noqa: E722
        if all(k in rec for k in ("code", "name", "branch")):
            records.append(rec)
    return records


def _load_all() -> List[Dict]:
    if DATA_FILE is None:
        return []
    try:
        return parse_records(DATA_FILE.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []


# Loaded ONCE at startup. Never sent wholesale to OpenRouter.
ALL_RECORDS: List[Dict] = _load_all()


def get_data_status() -> str:
    if DATA_FILE is None:
        return "No data file found. Place data.txt beside app.py."
    if not ALL_RECORDS:
        return f"File `{DATA_FILE.name}` found but 0 records were parsed."
    return f"Loaded **{len(ALL_RECORDS):,}** records from `{DATA_FILE.name}`."


# ================================================================
# ANY-VALUE HELPERS  — critical bug prevention
# ================================================================

def is_any(value) -> bool:
    """
    Returns True when value means 'no preference / no filter'.
    Handles: None, '', 'any', 'all', 'no preference'.

    ALWAYS use this instead of bare `if value:` for filter checks.
    """
    if value is None:
        return True
    return str(value).strip().lower() in {"", "any", "all", "no preference"}


def _is_pure_any_msg(text: str) -> bool:
    """True if the ENTIRE message expresses 'no preference'."""
    return text.strip().lower() in {
        "any", "all", "no preference", "any district", "all districts",
        "any branch", "all branches", "any type", "all types",
        "no district preference", "no branch preference",
        "doesn't matter", "dont matter", "no matter",
        "i don't mind", "i dont mind", "whatever", "anything",
    }


# ================================================================
# LOOKUP TABLES
# ================================================================

DISTRICT_ALIASES: Dict[str, str] = {
    "chennai": "Chennai",
    "madras": "Chennai",
    "kanchipuram": "Kancheepuram",
    "kancheepuram": "Kancheepuram",
    "kanchi": "Kancheepuram",
    "chengalpattu": "Chengalpattu",
    "chengalpet": "Chengalpattu",
    "coimbatore": "Coimbatore",
    "kovai": "Coimbatore",
    "salem": "Salem",
    "madurai": "Madurai",
    "trichy": "Tiruchirappalli",
    "tiruchi": "Tiruchirappalli",
    "tiruchirappalli": "Tiruchirappalli",
    "tirunelveli": "Tirunelveli",
    "nellai": "Tirunelveli",
    "erode": "Erode",
    "vellore": "Vellore",
    "thanjavur": "Thanjavur",
    "tanjore": "Thanjavur",
    "tiruppur": "Tiruppur",
    "tirupur": "Tiruppur",
    "dindigul": "Dindigul",
    "virudhunagar": "Virudhunagar",
    "thoothukudi": "Thoothukudi",
    "tuticorin": "Thoothukudi",
    "sivaganga": "Sivagangai",
    "sivagangai": "Sivagangai",
    "namakkal": "Namakkal",
    "karur": "Karur",
    "cuddalore": "Cuddalore",
    "villupuram": "Viluppuram",
    "viluppuram": "Viluppuram",
    "kallakurichi": "Kallakurichi",
    "dharmapuri": "Dharmapuri",
    "krishnagiri": "Krishnagiri",
    "thiruvallur": "Tiruvallur",
    "tiruvallur": "Tiruvallur",
    "tiruvannamalai": "Tiruvannamalai",
    "thiruvannamalai": "Tiruvannamalai",
    "nagapattinam": "Nagapattinam",
    "mayiladuthurai": "Mayiladuthurai",
    "perambalur": "Perambalur",
    "ariyalur": "Ariyalur",
    "pudukkottai": "Pudukkottai",
    "ramanathapuram": "Ramanathapuram",
    "ramnad": "Ramanathapuram",
    "the nilgiris": "The Nilgiris",
    "nilgiris": "The Nilgiris",
    "ooty": "The Nilgiris",
    "tenkasi": "Tenkasi",
    "theni": "Theni",
    "ranipet": "Ranipet",
    "tirupathur": "Tirupathur",
    "tiruvarur": "Tiruvarur",
    "kanniyakumari": "Kanniyakumari",
    "kanyakumari": "Kanniyakumari",
}

DISTRICTS: List[str] = sorted(list(set(DISTRICT_ALIASES.values())))


def normalize_district(value) -> Optional[str]:
    if not value:
        return None
    val_l = str(value).strip().lower()
    if val_l in {"any", "all", "all districts", "none", ""}:
        return None
    return DISTRICT_ALIASES.get(val_l, str(value).strip().title())


def district_matches(record_district: Optional[str], requested_district: Optional[str]) -> bool:
    if not requested_district or is_any(requested_district):
        return True
    actual = normalize_district(record_district)
    requested = normalize_district(requested_district)
    if not actual or not requested:
        return True
    return actual.lower() == requested.lower()


def branch_matches(record_branch: Optional[str], requested_branch: Optional[str]) -> bool:
    if not requested_branch or is_any(requested_branch):
        return True
    if not record_branch:
        return False
    rec_b_l = record_branch.lower()
    data_kws = BRANCH_MAP.get(requested_branch, {}).get("data", [requested_branch.lower()])
    return any(kw in rec_b_l for kw in data_kws)


def college_type_matches(record_name: Optional[str], requested_type: Optional[str]) -> bool:
    if not requested_type or is_any(requested_type):
        return True
    if not record_name:
        return False
    if requested_type == "Autonomous":
        return "autonomous" in record_name.lower()
    return _type_from_name(record_name).lower() == requested_type.lower()

# branch_display → {"user": [input aliases], "data": [data.txt keywords]}
BRANCH_MAP: Dict[str, Dict[str, List[str]]] = {
    "CSE": {
        "user": ["computer science and engineering", "computer science engineering",
                 "computer science", "cse"],
        "data": ["computer science"],
    },
    "ECE": {
        "user": ["electronics and communication engineering",
                 "electronics and communication", "electronics & communication", "ece"],
        "data": ["electronics and communication"],
    },
    "EEE": {
        "user": ["electrical and electronics engineering",
                 "electrical and electronics", "electrical & electronics", "eee"],
        "data": ["electrical and electronics"],
    },
    "AI & DS": {
        "user": ["artificial intelligence and data science",
                 "artificial intelligence & data science",
                 "ai and ds", "ai & ds", "ai&ds", "ai ds", "aids", "data science"],
        "data": ["artificial intelligence and data science",
                 "artificial intelligence & data science"],
    },
    "AI & ML": {
        "user": ["artificial intelligence and machine learning",
                 "artificial intelligence & machine learning",
                 "ai and ml", "ai & ml", "ai&ml", "ai ml", "aiml", "machine learning"],
        "data": ["artificial intelligence and machine learning",
                 "artificial intelligence & machine learning", "aiml"],
    },
    "IT": {
        "user": ["information technology"],  # standalone "IT" / "it" handled separately
        "data": ["information technology"],
    },
    "MECH": {
        "user": ["mechanical engineering", "mechanical", "mech"],
        "data": ["mechanical engineering"],
    },
    "CIVIL": {
        "user": ["civil engineering", "civil"],
        "data": ["civil engineering"],
    },
    "AGRI": {
        "user": ["agriculture engineering", "agricultural engineering", "agriculture", "agri"],
        "data": ["agriculture"],
    },
    "CHEM": {
        "user": ["chemical engineering", "chemical"],
        "data": ["chemical engineering"],
    },
    "MARINE": {
        "user": ["marine engineering", "marine"],
        "data": ["marine engineering"],
    },
    "AERO": {
        "user": ["aeronautical engineering", "aeronautical", "aero"],
        "data": ["aeronautical"],
    },
    "AUTO": {
        "user": ["automobile engineering", "automobile"],
        "data": ["automobile"],
    },
    "BIO": {
        "user": ["biomedical engineering", "biotechnology", "biomedical", "biotech"],
        "data": ["biomedical", "biotechnology"],
    },
}

VALID_CATEGORIES: Dict[str, str] = {
    "oc": "OC",
    "bc": "BC",
    "bcm": "BCM",
    "mbc": "MBC/DNC",
    "mbc/dnc": "MBC/DNC",
    "mbc & dnc": "MBC/DNC",
    "dnc": "MBC/DNC",
    "sc": "SC",
    "sca": "SCA",
    "st": "ST",
}


def extract_category(text: str) -> Optional[str]:
    text_l = text.lower().strip()
    if text_l in VALID_CATEGORIES:
        return VALID_CATEGORIES[text_l]
    tokens = re.split(r"[\s,;]+", text_l)
    for tok in tokens:
        if tok in VALID_CATEGORIES:
            return VALID_CATEGORIES[tok]
    if "open category" in text_l or "general" in text_l:
        return "OC"
    if "backward class muslim" in text_l:
        return "BCM"
    if "backward class" in text_l:
        return "BC"
    if "most backward" in text_l or "denotified" in text_l:
        return "MBC/DNC"
    if "scheduled caste arunth" in text_l or "arunthathiyar" in text_l:
        return "SCA"
    if "scheduled caste" in text_l:
        return "SC"
    if "scheduled tribe" in text_l:
        return "ST"
    return None


# ================================================================
# EXTRACTION FUNCTIONS
# ================================================================

def extract_cutoff(text: str) -> Optional[float]:
    text_l = text.lower().strip()
    patterns = [
        r"cut[\s\-]?off\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)",
        r"(?:my\s+)?cutoff\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)",
        r"(?:i\s+(?:got|scored|have)|score(?:\s+is)?|mark(?:\s+is)?|got|scored|enaku)\s+(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s+(?:cutoff|marks?|score|ku)",
        r"(?:for|with)\s+(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s+(?:bc|oc|bcm|mbc|sc|sca|st)\b",
    ]
    for p in patterns:
        m = re.search(p, text_l)
        if m:
            try:
                v = float(m.group(1))
                if 50 <= v <= 200:
                    return v
            except ValueError:
                pass
    # Standalone number 50-200 in text (ignoring 4-6 digit college codes)
    for m in re.finditer(r"\b(\d{2,3}(?:\.\d+)?)\b", text_l):
        try:
            v = float(m.group(1))
            if 50 <= v <= 200:
                return v
        except ValueError:
            pass
    return None




def extract_district(text: str) -> Optional[str]:
    text_l = text.lower().strip()
    if re.search(r"\bany\s+districts?\b|\bno\s+district\b", text_l):
        return "Any"
    for alias, canonical in sorted(DISTRICT_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(r"\b" + re.escape(alias) + r"\b", text_l):
            return canonical
    return None


def extract_branch(text: str) -> Optional[str]:
    text_l = text.lower().strip()
    # IT: case-sensitive uppercase check to avoid matching English pronoun "it"
    if re.search(r"\bIT\b", text) or "information technology" in text_l:
        return "IT"
    if re.fullmatch(r"it", text_l):
        return "IT"
    # All other branches: longest alias wins
    candidates: List[Tuple[int, str]] = []
    for branch, info in BRANCH_MAP.items():
        if branch == "IT":
            continue
        for alias in sorted(info["user"], key=len, reverse=True):
            if alias in text_l:
                candidates.append((len(alias), branch))
                break
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return None


def extract_college_type(text: str) -> Optional[str]:
    text_l = text.lower()
    if "self financing" in text_l or "self-financing" in text_l or "private" in text_l:
        return "Self Financing"
    if "government aided" in text_l or "govt aided" in text_l or \
       ("aided" in text_l and "government" not in text_l and "govt" not in text_l):
        return "Government Aided"
    if "autonomous" in text_l:
        return "Autonomous"
    if "government" in text_l or "govt" in text_l:
        return "Government"
    return None


def extract_college_code(text: str) -> Optional[str]:
    """Extract college code from explicit patterns only (avoids matching cutoff values 0-200)."""
    text_l = text.lower()
    patterns = [
        r"college\s+code\s*[:#]?\s*(\d{4,6})",
        r"college\s+#?\s*(\d{4,6})\b",
        r"\bcode\s*[:#]?\s*(\d{4,6})\b",
        r"(?:about|details?|tell me about)\s+college\s+(\d{4,6})",
    ]
    for p in patterns:
        m = re.search(p, text_l)
        if m:
            c = m.group(1)
            if float(c) > 200:
                return c
    # Standalone 4-6 digit number only when context is clearly a college lookup
    if any(kw in text_l for kw in ["college", "code", "about"]):
        m = re.search(r"\b(\d{4,6})\b", text_l)
        if m and float(m.group(1)) > 200:
            return m.group(1)
    return None


def extract_result_limit(text: str) -> Optional[int]:
    m = re.search(r"\btop\s+(\d+)\b", text.lower())
    if m:
        return max(1, min(int(m.group(1)), 20))
    m = re.search(r"\b(?:show|give|list|get)\s+(?:me\s+)?(\d+)\b", text.lower())
    if m:
        return max(1, min(int(m.group(1)), 20))
    return None


# ================================================================
# STATE
# ================================================================

def new_state() -> Dict:
    """Return a fresh, fully initialised state dict."""
    return {
        "cutoff":                None,
        "category":              None,
        "district":              None,   # None = no filter (same as 'any')
        "branch":                None,   # None = no filter
        "college_type":          None,   # None = no filter
        "limit":                 10,     # default results count
        "recommendation_offset": 0,      # pagination offset for 'more colleges'
        "selected_college":      None,
        "_flow_step":            None,   # 'cutoff', 'category', or None
    }


def clear_filters(state: Dict) -> None:
    """Clear district, branch, college_type, selected_college while keeping cutoff & category."""
    state["district"] = None
    state["branch"] = None
    state["college_type"] = None
    state["selected_college"] = None
    state["recommendation_offset"] = 0


def update_state(user_message: str, state: Dict) -> List[str]:
    """Extract fields from user_message, update state, return list of changed fields."""
    newly_set: List[str] = []

    def _set(field: str, value):
        if state.get(field) != value:
            state[field] = value
            newly_set.append(field)

    v = extract_cutoff(user_message)
    if v is not None:
        _set("cutoff", v)

    v = extract_category(user_message)
    if v:
        _set("category", v)

    v = extract_district(user_message)
    if v is not None:
        _set("district", v)

    v = extract_branch(user_message)
    if v is not None:
        _set("branch", v)

    v = extract_college_type(user_message)
    if v is not None:
        _set("college_type", v)

    v = extract_result_limit(user_message)
    if v is not None:
        _set("limit", v)

    return newly_set


def clear_filters(state: Dict) -> None:
    """Clear district/branch/college_type/selected_college. Retain cutoff + category."""
    state["district"]         = None
    state["branch"]           = None
    state["college_type"]     = None
    state["selected_college"] = None
    # Return guided flow to district step
    if state.get("cutoff") is not None and state.get("category") is not None:
        state["_flow_step"] = 2
    elif state.get("cutoff") is not None:
        state["_flow_step"] = 1
    else:
        state["_flow_step"] = 0


def reset_state(state: Dict) -> None:
    """Clear ALL state including internal flow step."""
    state.clear()
    state.update(new_state())


# ================================================================
# COLLEGE TYPE DETECTION FROM NAME
# ================================================================

_GOV_KEYWORDS = [
    "government college of engineering", "government engineering college",
    "government polytechnic", "university college of engineering",
    "college of engineering, anna", "college of engineering, guindy",
    "alagappa college of technology", "madras institute of technology",
    "government college", "govt college", " gce ",
]


def _type_from_name(name: str) -> str:
    n = name.lower()
    if any(kw in n for kw in _GOV_KEYWORDS):
        return "Government"
    if "aided" in n:
        return "Government Aided"
    return "Self Financing"


# ================================================================
# FILTER AND RANK  —  pure Python, zero LLM
# ================================================================

def filter_and_rank(state: Dict, limit: int = 10) -> List[Dict]:
    """
    Filter ALL_RECORDS by student preferences, rank by closeness to cutoff.
    Hard validation: filter FIRST, rank AFTER.
    """
    if not ALL_RECORDS:
        return []

    student_cutoff = state.get("cutoff")
    category       = state.get("category")
    district       = state.get("district")
    branch         = state.get("branch")
    college_type   = state.get("college_type")

    candidates = ALL_RECORDS.copy()

    # 1. DISTRICT HARD FILTER
    if district and not is_any(district):
        candidates = [r for r in candidates if district_matches(r.get("district"), district)]

    # 2. BRANCH HARD FILTER
    if branch and not is_any(branch):
        candidates = [r for r in candidates if branch_matches(r.get("branch"), branch)]

    # 3. COLLEGE TYPE HARD FILTER
    if college_type and not is_any(college_type):
        candidates = [r for r in candidates if college_type_matches(r.get("name"), college_type)]

    # 4. COMMUNITY (exact match if category provided, else default to OC baseline)
    target_cat = category.upper() if (category and not is_any(category)) else "OC"
    candidates = [r for r in candidates if r.get("community", "") == target_cat]

    # 5. CUTOFF (college cutoff <= student cutoff)
    if student_cutoff is not None:
        candidates = [r for r in candidates if r.get("cutoff") is not None and r.get("cutoff") <= student_cutoff]

    # 6. RANK AFTER FILTERS
    if student_cutoff is not None:
        candidates.sort(key=lambda r: (
            r.get("cutoff") is None,
            -(r.get("cutoff") or 0),
        ))

    # 7. DEDUPLICATE BY UNIQUE COLLEGE CODE
    seen_codes = set()
    deduped: List[Dict] = []
    for rec in candidates:
        code = rec.get("code")
        if code not in seen_codes:
            seen_codes.add(code)
            deduped.append(rec)

    # 8. HARD VALIDATION ASSERTION
    if district and not is_any(district):
        deduped = [r for r in deduped if district_matches(r.get("district"), district)]

    return deduped[:limit]


def find_by_code(code: str) -> List[Dict]:
    """Return ALL records for a given college code (all branches, all communities)."""
    return [r for r in ALL_RECORDS if r.get("code") == code]


def get_general_colleges(
    district: Optional[str] = None,
    branch: Optional[str] = None,
    college_type: Optional[str] = None,
    limit: int = 10,
) -> List[Dict]:
    """
    Get top colleges for general recommendations (when cutoff is None).
    Uses OC community as baseline closing cutoff.
    Filters FIRST, ranks AFTER.
    """
    candidates = ALL_RECORDS.copy()

    # 1. DISTRICT HARD FILTER
    if district and not is_any(district):
        candidates = [r for r in candidates if district_matches(r.get("district"), district)]

    # 2. BRANCH HARD FILTER
    if branch and not is_any(branch):
        candidates = [r for r in candidates if branch_matches(r.get("branch"), branch)]

    # 3. COLLEGE TYPE HARD FILTER
    if college_type and not is_any(college_type):
        candidates = [r for r in candidates if college_type_matches(r.get("name"), college_type)]

    # 4. COMMUNITY (OC baseline)
    candidates = [r for r in candidates if r.get("community") == "OC"]

    # 5. RANK AFTER FILTERS
    candidates.sort(key=lambda r: -(r.get("cutoff") or 0))

    # 6. DEDUPLICATE BY UNIQUE COLLEGE CODE
    seen_codes = set()
    deduped = []
    for rec in candidates:
        code = rec.get("code")
        if code not in seen_codes:
            seen_codes.add(code)
            deduped.append(rec)

    # 7. HARD VALIDATION ASSERTION
    if district and not is_any(district):
        deduped = [r for r in deduped if district_matches(r.get("district"), district)]

    return deduped[:limit]


def build_general_recommendation_response(records: List[Dict], state: Dict, start_offset: int = 0) -> str:
    """Build markdown table for general (non-personalized) college recommendation."""
    dist = state.get("district")
    norm_dist = normalize_district(dist) if dist else None

    if not records:
        if norm_dist:
            return (
                f"No colleges were found in the TNEA data for **{norm_dist}**.\n\n"
                "**Suggestions:**\n"
                "- Try searching another district (e.g. `Chennai`, `Coimbatore`, `Salem`)\n"
                "- Search across all districts by typing `any district`\n"
                "- Provide your TNEA cutoff for personalized recommendations"
            )
        return _no_results_msg(state)

    dist_display = norm_dist or "Tamil Nadu"
    branch_display = f" • Branch: {state['branch']}" if state.get("branch") else ""

    lines = [
        f"## \U0001f3af Top Colleges in {dist_display}",
        "",
        f"Here are the top colleges available in the TNEA data for **{dist_display}**{branch_display}.",
        "",
        "| # | College | District | Branch | Cutoff |",
        "|---|---|---|---|---|",
    ]
    for i, rec in enumerate(records, start_offset + 1):
        co = rec.get("cutoff")
        co_str = f"{co:g}" if co is not None else "N/A"
        lines.append(
            f"| {i} | {_short_name(rec.get('name', '?'))} | "
            f"{rec.get('district', '?')} | {rec.get('branch', '?')} | "
            f"{co_str} |"
        )
    lines += [
        "",
        "For a personalized recommendation based on your cutoff and community, tell me your TNEA cutoff.",
        "",
        "Example:",
        "",
        "`160`",
        "",
        "or",
        "",
        "`my cutoff is 160`",
    ]
    return "\n".join(lines)


def filter_records(state: Dict, limit: int = 10) -> List[Dict]:
    """Filter records for personalized recommendation."""
    return filter_and_rank(state, limit)


def build_recommendation_response(records: List[Dict], state: Dict) -> str:
    """Build markdown table for personalized recommendation."""
    return generate_recommendation_table(records, state)


# ================================================================
# CHANCE LABEL
# ================================================================

def chance_label(student_cutoff: Optional[float], rec_cutoff: Optional[float]) -> str:
    if student_cutoff is None or rec_cutoff is None:
        return "\u2014"
    diff = student_cutoff - rec_cutoff
    if diff >= 10: return "Safe"
    if diff >= 5:  return "Good"
    if diff >= 0:  return "Moderate"
    return "Difficult"


# ================================================================
# FORMAT: RECOMMENDATION TABLE  (Python Markdown — no LLM)
# ================================================================

def _short_name(full_name: str) -> str:
    return full_name.split(",")[0].strip()


def generate_recommendation_table(records: List[Dict], state: Dict, start_offset: int = 0) -> str:
    """Build a complete Markdown recommendation table entirely in Python."""
    if not records:
        return _no_results_msg(state)

    sc       = state.get("cutoff")
    cat      = state.get("category")
    dist_d   = state.get("district") or "Any"
    branch_d = state.get("branch") or "Any"
    type_d   = state.get("college_type") or "Any"

    cutoff_str = f"{sc:g}" if sc is not None else "Not specified"
    cat_str    = cat if cat else "OC (General)"

    lines = [
        f"## \U0001f3af Top {len(records)} Colleges",
        "",
        f"**Cutoff:** {cutoff_str} \u2022 **Category:** {cat_str} \u2022 "
        f"**District:** {dist_d} \u2022 **Branch:** {branch_d} \u2022 **Type:** {type_d}",
        "",
        f"| # | College | District | Branch | {cat_str} Cutoff | Chance |",
        "|---:|:---|:---|:---|---:|:---|",
    ]
    for i, rec in enumerate(records, start_offset + 1):
        co  = rec.get("cutoff")
        lines.append(
            f"| {i} | {_short_name(rec.get('name', '?'))} | "
            f"{rec.get('district', '?')} | {rec.get('branch', '?')} | "
            f"{f'{co:g}' if co is not None else 'N/A'} | "
            f"{chance_label(sc, co)} |"
        )
    lines += [
        "",
        "_Real TNEA 2025 data. Admission depends on counselling round, "
        "seat availability, competition, and TNEA rules._",
    ]

    missing = []
    if sc is None:
        missing.append("TNEA cutoff mark")
    if cat is None:
        missing.append("community/category")
    if missing:
        lines += [
            "",
            f"\U0001f4a1 *To check your specific admission chances, tell me your {' and '.join(missing)} (e.g. `160 BC`).*",
        ]

    return "\n".join(lines)


def _no_results_msg(state: Dict) -> str:
    lines = ["**No matching colleges found.** Active filters:"]
    if state.get("cutoff"):
        lines.append(f"- Cutoff: {state['cutoff']:g}")
    if state.get("category"):
        lines.append(f"- Category: {state['category']}")
    if not is_any(state.get("district")):
        lines.append(f"- District: {state['district']}")
    if not is_any(state.get("branch")):
        lines.append(f"- Branch: {state['branch']}")
    if not is_any(state.get("college_type")):
        lines.append(f"- College Type: {state['college_type']}")
    lines += [
        "",
        "**Try relaxing one filter:**",
        "- Type `any` to remove the district filter",
        "- Type `any` to remove the branch filter",
        "- Type `clear filters` to reset district, branch, and type",
    ]
    return "\n".join(lines)


# ================================================================
# FORMAT: COLLEGE DETAIL
# ================================================================

def format_college_detail(records: List[Dict]) -> str:
    """Format all records for one college into a compact context string for OpenRouter."""
    if not records:
        return "No records found for this college code."
    name     = records[0].get("name", "Unknown")
    code     = records[0].get("code", "?")
    district = records[0].get("district", "?")
    lines = [
        f"College Code: {code}",
        f"Name: {name}",
        f"District: {district}",
        "",
        "Branches and community-wise closing cutoff marks:",
    ]
    branch_map: Dict[str, List] = {}
    for r in records:
        branch_map.setdefault(r.get("branch", "Unknown"), []).append(r)
    for branch, rows in sorted(branch_map.items()):
        lines.append(f"  {branch}:")
        for row in sorted(rows, key=lambda x: x.get("community", "")):
            co = row.get("cutoff")
            lines.append(
                f"    {row.get('community', '?')}: "
                f"{f'{co:g}' if co is not None else 'N/A'}"
            )
    return "\n".join(lines)


# ================================================================
# INTENT DETECTION
# ================================================================

_GREETINGS = {
    "hi", "hello", "hey", "hai", "hii", "helo", "hiya",
    "good morning", "good afternoon", "good evening", "vanakkam", "sup", "yo",
}

_IDENTITY_PHRASES = [
    "who are you", "who r you", "who r u", "who are ypu", "what are you",
    "who is this", "tell me about yourself", "what can you do", "what do you do",
    "what is this chatbot", "what is this bot", "how can you help",
]

_KNOWLEDGE_PHRASES = [
    "what is tnea", "what does tnea", "how does tnea", "tnea full form",
    "how is cutoff calculated", "how to calculate cutoff", "cutoff formula",
    "what is counselling", "what is choice filling", "what is seat allotment",
    "what is sliding", "what is upward movement", "what is supplementary",
    "what is government college", "what is autonomous college",
    "what is self financing", "what is college code", "what is branch",
    "what is community", "what is category", "tnea process", "tnea procedure",
    "how does admission work", "what documents", "tnea registration",
    "what is oc", "what is bc", "what is bcm", "what is mbc", "what is sc", "what is st",
    "what is cse", "what is ece", "what is eee", "what is mechanical", "what is civil",
    "what is ai and data science", "what is ai and machine learning",
    "cse or ece", "which engineering branch", "how do i choose a college",
    "what is engineering", "is cse difficult", "jobs after cse", "jobs after ece",
]

_REC_PATTERNS = [
    r"\btop\b", r"\brecommend", r"\bsuggest",
    r"\bbest\b", r"\bshow\s+(?:me\s+)?college",
    r"\bgive\s+(?:me\s+)?college", r"\bwhich\s+college",
    r"\bwhat\s+college", r"\blist\s+college",
    r"\bfind\s+college", r"\bcollege\s+list",
    r"\bcolleges?\s+in\b", r"\bcolleges?\s+for\b",
]


def is_general_question(text: str) -> bool:
    """True if text is a general knowledge / educational / career / off-topic question."""
    text_l = text.lower().strip()

    # Direct cutoff mark or community entry is NOT a general question
    if extract_cutoff(text) is not None or extract_category(text) is not None:
        return False

    # Question indicator words/phrases
    question_starters = [
        "what is", "what does", "what do", "what are", "what stand", "what means",
        "how is", "how does", "how to", "how do", "how can", "how many",
        "which is", "which branch", "is it", "is cse", "is ece", "is autonomous", "is government",
        "jobs after", "career after", "salary after", "future of",
        "difference between", "vs", "versus", "explain", "tell me about",
        "what happens", "what should", "can i get after", "why", "where", "when",
    ]

    general_topics = [
        "tnea", "counselling", "choice filling", "seat allotment", "upward movement",
        "sliding", "community", "category", "open category", "backward class", "scheduled caste",
        "autonomous", "government college", "self financing", "anna university",
        "computer science", "mechanical engineering", "civil engineering",
        "electrical and electronics", "electronics and communication", "artificial intelligence",
        "data science", "machine learning", "engineering", "placements", "salary", "jobs",
    ]

    has_rec = any(re.search(p, text_l) for p in _REC_PATTERNS)
    has_starter = any(qs in text_l for qs in question_starters) or text_l.endswith("?")
    has_topic = any(gt in text_l for gt in general_topics)

    # General question if it has a question starter or topic (unless purely a recommendation request)
    if (has_starter or (has_topic and not has_rec)):
        return True

    return False


def detect_intent(text: str) -> str:
    """
    Deterministic intent router.
    Greetings and identity queries NEVER return 'recommendation'.
    """
    text_l = text.lower().strip()

    if text_l in {"reset", "restart", "reset chat", "start again", "start over", "new chat"}:
        return "reset"

    if any(p in text_l for p in [
        "clear filter", "remove filter", "clear all filter",
        "remove all filter", "reset filter", "reset filters",
    ]):
        return "clear_filters"

    if any(p in text_l for p in [
        "more college", "more colleges", "show more", "show next",
        "next colleges", "next college", "next", "more"
    ]):
        return "more_results"

    if text_l in _GREETINGS:
        return "greeting"

    if text_l == "help" or any(p in text_l for p in _IDENTITY_PHRASES):
        return "identity"

    if re.search(r"college\s+(?:code\s+)?\d{3,6}", text_l):
        return "college_lookup"
    if re.search(r"\bcode\s*[:#]?\s*\d{3,6}\b", text_l):
        return "college_lookup"

    if any(re.search(p, text_l) for p in _REC_PATTERNS):
        return "recommendation"

    if any(p in text_l for p in _KNOWLEDGE_PHRASES):
        return "knowledge"

    return "general"


# ================================================================
# GUIDED FLOW PROMPTS AND LOGIC
# ================================================================

_CAT_PROMPT = """\
What is your **community/category**?

| Code | Community |
|:---|:---|
| OC | Open Category |
| BC | Backward Class |
| BCM | Backward Class Muslim |
| MBC | Most Backward Class / MBC/DNC |
| SC | Scheduled Caste |
| SCA | SC Arunthathiyar |
| ST | Scheduled Tribe |

Just type the code \u2014 e.g. `BC` or `OC`"""

_DIST_PROMPT = """\
Which **district** do you prefer?

Examples: `Chennai`, `Coimbatore`, `Salem`, `Madurai`, `Trichy`, `Vellore`

Type `any` if you have no district preference."""

_BRANCH_PROMPT = """\
Which **branch** are you interested in?

| Code | Branch |
|:---|:---|
| CSE | Computer Science Engineering |
| ECE | Electronics & Communication |
| EEE | Electrical & Electronics |
| IT | Information Technology |
| MECH | Mechanical Engineering |
| CIVIL | Civil Engineering |
| AI & DS | Artificial Intelligence & Data Science |
| AI & ML | Artificial Intelligence & Machine Learning |

Type `any` for all branches."""

_TYPE_PROMPT = """\
What **type of college** are you looking for?

- Government
- Government Aided
- Self Financing (Private)
- Autonomous
- Any"""


def _handle_any_in_flow(state: Dict, flow_step: int) -> Optional[str]:
    """
    Apply 'any' to the correct field based on the current guided flow step.
    Returns the next prompt, or None to fall through to recommendations.
    """
    if flow_step == 2:
        state["district"] = None
        state["_flow_step"] = 3
        return "Got it. District: **Any**.\n\n" + _BRANCH_PROMPT
    if flow_step == 3:
        state["branch"] = None
        state["_flow_step"] = 4
        return "Got it. Branch: **Any**.\n\n" + _TYPE_PROMPT
    if flow_step == 4:
        state["college_type"] = None
        state["_flow_step"] = 5
        return None  # fall through to recommendation
    return None


def _advance_flow(state: Dict, newly_set: List[str]) -> Optional[str]:
    """
    Advance the guided flow after update_state().
    Returns a guidance prompt, or None to fall through.
    Multiple fields in one message bypasses the guided flow entirely.
    """
    step   = state.get("_flow_step", 0)
    cutoff = state.get("cutoff")
    cat    = state.get("category")

    # Multiple fields at once → skip guided, go straight to recommendations
    meaningful = [f for f in newly_set if f != "limit"]
    if len(meaningful) > 1:
        if cutoff is not None and cat is not None:
            state["_flow_step"] = 5
        return None

    if "cutoff" in newly_set and step <= 0 and not cat:
        state["_flow_step"] = 1
        return f"Got it \U0001f44d Your TNEA cutoff is **{cutoff:g}**.\n\n" + _CAT_PROMPT

    if "category" in newly_set and step <= 1 and cutoff is not None:
        state["_flow_step"] = 2
        return f"Got it. Category: **{cat}**.\n\n" + _DIST_PROMPT

    if "district" in newly_set and step <= 2 and cutoff is not None and cat:
        state["_flow_step"] = 3
        return f"Got it. District: **{state.get('district') or 'Any'}**.\n\n" + _BRANCH_PROMPT

    if "branch" in newly_set and step <= 3 and cutoff is not None and cat:
        state["_flow_step"] = 4
        return f"Got it. Branch: **{state.get('branch') or 'Any'}**.\n\n" + _TYPE_PROMPT

    if "college_type" in newly_set and step <= 4 and cutoff is not None and cat:
        state["_flow_step"] = 5
        return None  # fall through to recommendation

    return None


# ================================================================
# OPENROUTER SYSTEM PROMPT
# ================================================================

_SYSTEM_PROMPT = """\
You are a TNEA Engineering College Counselling Assistant for Tamil Nadu, India.

RESPONSIBILITIES:
- Answer general questions naturally (what is TNEA, cutoff calculation, counselling process, choice filling, seat allotment, community categories, engineering branches like CSE/ECE/EEE/Mech/Civil/AI&DS, career guidance, college choice advice, engineering concepts, etc.)
- For general engineering and TNEA questions, provide clear, educational, student-friendly explanations.
- Maintain conversation history naturally so follow-up questions (e.g. "is it difficult?", "what jobs can I get after it?") make sense in context.
- If an off-topic question is asked (e.g. "What is the capital of France?"), answer briefly and gently add: "I'm mainly designed for TNEA and engineering college guidance."
- If the user asks about current official TNEA dates, fee structures, seat matrix, or official rules that may change annually and are not in data.txt, remind the user to verify from official TNEA/DoTE websites (tneaonline.org).
- Explain college information using ONLY the structured data supplied by the Python application.

STRICT DATA RULES:
1. NEVER invent college names, codes, cutoffs, fees, placements, seats, rankings, or hostel details
2. NEVER guarantee admission
3. Use ONLY the supplied college data for college-specific answers
4. If data is missing, say: "I don't have that information in my current TNEA data."
5. Chance labels: Safe | Good | Moderate | Difficult
6. Admission depends on counselling round, seat availability, competition, and TNEA rules
7. Do not say "As an AI language model..."
8. Do not fabricate data to fill gaps

LANGUAGE: Simple English by default. Tamil-English mix acceptable if student uses it.
TONE: Helpful, direct, honest, realistic, student-friendly."""


# ================================================================
# OPENROUTER CALL
# ================================================================

def ask_openrouter(
    user_message: str,
    state: Dict,
    context: str,
    chat_history: List[Dict],
) -> str:
    if client is None:
        return (
            "OpenRouter API key is missing. "
            "Add `OPENROUTER_API_KEY=...` to your `.env` file."
        )

    state_block = (
        "\n\nCURRENT STUDENT STATE\n"
        "=====================\n"
        f"Cutoff:        {state.get('cutoff') or 'Not set'}\n"
        f"Category:      {state.get('category') or 'Not set'}\n"
        f"District:      {state.get('district') or 'Any'}\n"
        f"Branch:        {state.get('branch') or 'Any'}\n"
        f"College Type:  {state.get('college_type') or 'Any'}\n"
        f"Selected Code: {state.get('selected_college') or 'None'}\n"
        "\nSUPPLIED DATA (Python-filtered; NOT the full dataset)\n"
        "======================================================\n"
        f"{context}"
    )

    messages = [{"role": "system", "content": _SYSTEM_PROMPT + state_block}]
    messages.extend(chat_history[-MAX_HISTORY:])
    messages.append({"role": "user", "content": user_message})

    try:
        resp = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1500,
        )
        content = resp.choices[0].message.content
        return content if content else "Sorry, I couldn't generate a response. Please try again."
    except Exception as e:
        return f"OpenRouter error: {e}"


# ================================================================
# PROCESS MESSAGE  —  MAIN ENTRY POINT
# ================================================================

def process_message(
    user_message: str,
    state: Dict,
    chat_history: List[Dict],
) -> Tuple[str, List[Dict]]:
    """
    Route user message. Return (response_text, updated_chat_history).

    Python  → state extraction, filtering, ranking, table generation, guided flow
    OpenRouter → TNEA knowledge, college explanations, conversation
    """

    def done(text: str) -> Tuple[str, List[Dict]]:
        return text, chat_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": text},
        ]

    text_l = user_message.lower().strip()
    intent = detect_intent(user_message)

    # ----------------------------------------------------------------
    # BUG 1 CHECK — Waiting for category after cutoff entry
    # ----------------------------------------------------------------
    if state.get("cutoff") is not None and state.get("category") is None and state.get("_flow_step") == "category":
        cat_extracted = extract_category(user_message)
        if cat_extracted:
            state["category"] = cat_extracted
            state["_flow_step"] = None
            state["recommendation_offset"] = 0
            records = filter_records(state=state, limit=state.get("limit", 10))
            return done(build_recommendation_response(records, state))
        else:
            # Invalid category input!
            upper_raw = user_message.strip().upper()
            return done(
                f"I didn't recognize `{upper_raw}` as a TNEA community/category.\n\n"
                "Please choose one:\n\n"
                "- OC\n"
                "- BC\n"
                "- BCM\n"
                "- MBC/DNC\n"
                "- SC\n"
                "- SCA\n"
                "- ST"
            )

    # 1. EXTRACT INFORMATION FROM THE CURRENT MESSAGE
    cutoff       = extract_cutoff(user_message)
    district     = extract_district(user_message)
    branch       = extract_branch(user_message)
    category     = extract_category(user_message)
    college_type = extract_college_type(user_message)
    limit        = extract_result_limit(user_message)

    # Handle explicit "any" clear keywords in user message
    if re.search(r"\bany\s+districts?\b|\ball\s+districts?\b|\bno\s+district\b", text_l):
        state["district"] = None
        district = None
    elif district == "Any":
        state["district"] = None
        district = None

    if re.search(r"\bany\s+branch(?:es)?\b|\ball\s+branch(?:es)?\b", text_l):
        state["branch"] = None
        branch = None

    if re.search(r"\bany\s+(?:college\s+)?type\b|\ball\s+(?:college\s+)?types\b", text_l):
        state["college_type"] = None
        college_type = None

    # BUG 2 RULE — Only update state fields that are EXPLICITLY present in current message
    if cutoff is not None:
        # Don't overwrite existing cutoff with a college code number like 1413 unless explicitly cutoff prompt
        if cutoff <= 200:
            state["cutoff"] = cutoff
    if district is not None and district != "Any":
        state["district"] = district
        state["recommendation_offset"] = 0
    if branch is not None and branch != "Any":
        state["branch"] = branch
        state["recommendation_offset"] = 0
    if category is not None:
        state["category"] = category
        state["recommendation_offset"] = 0
    if college_type is not None and college_type != "Any":
        state["college_type"] = college_type
        state["recommendation_offset"] = 0
    if limit is not None:
        state["limit"] = limit

    # ----------------------------------------------------------------
    # BUG 3 CHECK — Pure Numeric Input Handling
    # ----------------------------------------------------------------
    if re.fullmatch(r"\d{1,6}", text_l):
        code_candidate = text_l.strip()
        num_val = float(code_candidate)

        # Case A: Bot is specifically waiting for cutoff mark entry
        if state.get("_flow_step") == "cutoff" or (state.get("cutoff") is None and num_val <= 200):
            state["cutoff"] = num_val
            state["_flow_step"] = "category"
            return done(
                f"Got it 👍 Your cutoff is **{num_val:g}**.\n\n"
                "For a personalized recommendation, what is your **community/category**?\n\n"
                "- OC\n"
                "- BC\n"
                "- BCM\n"
                "- MBC/DNC\n"
                "- SC\n"
                "- SCA\n"
                "- ST"
            )

        # Case B: Cutoff already set or not in cutoff prompt -> Check college code
        recs = find_by_code(code_candidate)
        if recs:
            state["selected_college"] = code_candidate
            return done(format_college_detail(recs))
        else:
            return done(
                f"I couldn't find college code `{code_candidate}` in the current TNEA data.\n\n"
                "You can enter a valid college code or ask me another TNEA question."
            )

    # ----------------------------------------------------------------
    # BUG 4 CHECK — Pagination ("More Colleges")
    # ----------------------------------------------------------------
    if intent == "more_results":
        offset = state.get("recommendation_offset", 0)
        lim    = state.get("limit", 10)

        # Check if personalized
        if state.get("cutoff") is not None and state.get("category") is not None:
            all_records = filter_and_rank(state, limit=9999)
            if offset >= len(all_records):
                return done("There are no more colleges matching your current filters.")
            page = all_records[offset : offset + lim]
            state["recommendation_offset"] = offset + len(page)
            return done(generate_recommendation_table(page, state, start_offset=offset))
        else:
            all_records = get_general_colleges(
                district=state.get("district"),
                branch=state.get("branch"),
                college_type=state.get("college_type"),
                limit=9999,
            )
            if offset >= len(all_records):
                return done("There are no more colleges matching your current filters.")
            page = all_records[offset : offset + lim]
            state["recommendation_offset"] = offset + len(page)
            return done(build_general_recommendation_response(page, state, start_offset=offset))

    # CASE 5 & CASE 6: Clear filters / New search
    if any(p in text_l for p in [
        "clear filter", "remove filter", "clear preferences", "reset filter",
        "new search", "search something else", "another college", "start a new search"
    ]) or intent == "clear_filters":
        cutoff_val = state.get("cutoff")
        cat_val    = state.get("category")
        clear_filters(state)
        msg = "Filters cleared. ✅"
        notes = []
        if cutoff_val is not None:
            notes.append(f"Cutoff **{cutoff_val:g}** retained.")
        if cat_val:
            notes.append(f"Category **{cat_val}** retained.")
        if notes:
            msg += "\n\n" + "  ".join(notes)
        msg += "\n\nWhat would you like to search for?"
        return done(msg)

    # RESET
    if intent == "reset":
        reset_state(state)
        return (
            "Chat reset. 🔄\n\n"
            "What is your **TNEA cutoff mark**?\n\n"
            "Example: `160` or `my cutoff is 160`",
            [],
        )

    # CASE 1: GREETING
    if intent == "greeting":
        return done(
            "Hello! I'm your **TNEA Engineering College Counselling Assistant**. 🎓\n\n"
            "I use real TNEA 2025 data to help you find colleges that match "
            "your cutoff, community, district, and branch.\n\n"
            "To get started, tell me your **TNEA cutoff mark**.\n\n"
            "Example: `160` or `my cutoff is 160`"
        )

    # CASE 1: IDENTITY / HELP
    if intent == "identity":
        return done(
            "I'm a **TNEA Engineering College Counselling Assistant**.\n\n"
            "**What I can do:**\n"
            "- Find colleges by cutoff, community, district, branch, and college type\n"
            "- Look up any college by its code\n"
            "- Answer general TNEA questions\n\n"
            "**To start:** `my cutoff is 160`"
        )

    # CASE 9: COLLEGE LOOKUP
    if intent == "college_lookup":
        code = extract_college_code(user_message)
        if not code:
            return done("Please provide a college code.\n\nExample: `college code 2347`")
        recs = find_by_code(code)
        if not recs:
            return done(f"I couldn't find college code `{code}` in the current TNEA data.\n\nYou can enter a valid college code or ask me another TNEA question.")
        state["selected_college"] = code
        return done(format_college_detail(recs))
    # ----------------------------------------------------------------
    # GENERAL QUESTION & MIXED QUESTION ROUTING
    # ----------------------------------------------------------------
    has_rec = (intent == "recommendation") or any(re.search(p, text_l) for p in _REC_PATTERNS)
    is_recommendation = has_rec or (district is not None or branch is not None or college_type is not None or limit is not None)
    has_gen = is_general_question(user_message)

    # Case A: Pure General Question (e.g. "What is TNEA?", "what is CSE?", "is CSE difficult?")
    if has_gen and not has_rec and intent != "college_lookup":
        # DO NOT modify state filters. Answer naturally using OpenRouter.
        response = ask_openrouter(
            user_message, state,
            "The user is asking a general knowledge / educational / career question. "
            "Answer clearly and concisely using your general knowledge. "
            "Do not invent college data or official TNEA dates/fees.",
            chat_history,
        )
        return done(response)

    # Case B: Mixed Question (e.g. "what is CSE and what are the best CSE colleges in Chennai?")
    if has_gen and has_rec:
        # 1. Update state filters with explicit values in current message
        if district is not None and district != "Any": state["district"] = district
        if branch is not None and branch != "Any": state["branch"] = branch
        if college_type is not None and college_type != "Any": state["college_type"] = college_type

        # 2. Get real Python records
        lim = state.get("limit", 10)
        if state.get("cutoff") is not None and state.get("category") is not None:
            recs = filter_records(state=state, limit=lim)
            rec_table = generate_recommendation_table(recs, state)
        else:
            recs = get_general_colleges(
                district=state.get("district"),
                branch=state.get("branch"),
                college_type=state.get("college_type"),
                limit=lim,
            )
            rec_table = build_general_recommendation_response(recs, state)

        # 3. Ask OpenRouter to combine brief explanation + real recommendation table
        context_str = (
            "INSTRUCTIONS FOR MIXED QUESTION:\n"
            "1. Briefly answer the general question part first in 2-3 sentences.\n"
            "2. Then present the exact Python college recommendations table below.\n\n"
            f"PYTHON COLLEGE DATA TABLE:\n{rec_table}"
        )
        response = ask_openrouter(user_message, state, context_str, chat_history)
        return done(response)

    # DIRECT GENERAL RECOMMENDATION (or personalized if cutoff/category present)
    if is_recommendation or state.get("district") is not None or state.get("branch") is not None:
        state["recommendation_offset"] = 0
        lim = state.get("limit", 10)

        # If cutoff and category are both available -> Personalized
        if state.get("cutoff") is not None and state.get("category") is not None:
            records = filter_records(state=state, limit=lim)
            state["recommendation_offset"] = len(records)
            return done(build_recommendation_response(records, state))

        # If cutoff is available but category is missing -> Ask category
        if state.get("cutoff") is not None and state.get("category") is None:
            cutoff_val = state["cutoff"]
            state["_flow_step"] = "category"
            return done(
                "Got it 👍 Your cutoff is **{:.1f}**.\n\n"
                "For a personalized recommendation, what is your **community/category**?\n\n"
                "- OC\n"
                "- BC\n"
                "- BCM\n"
                "- MBC/DNC\n"
                "- SC\n"
                "- SCA\n"
                "- ST".format(cutoff_val)
            )

        # No cutoff -> General recommendation
        records = get_general_colleges(
            district=state.get("district"),
            branch=state.get("branch"),
            college_type=state.get("college_type"),
            limit=lim,
        )
        state["recommendation_offset"] = len(records)
        return done(build_general_recommendation_response(records, state))

    # Fallback to general openrouter conversation
    response = ask_openrouter(
        user_message, state,
        "No specific college data for this query. "
        "Answer from general TNEA knowledge. Do not invent college names or data.",
        chat_history,
    )
    return done(response)