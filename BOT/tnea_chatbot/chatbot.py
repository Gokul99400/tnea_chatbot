"""
chatbot.py  —  TNEA Engineering College Counselling Chatbot  (Phase 1 + Phase 2)

ARCHITECTURE
  Python layer  : state management, parsing, filtering, ranking, tiers,
                  comparison, explanation, table generation, guided flow
  OpenRouter    : general TNEA questions, college explanations,
                  conversational responses

STRICT RULES
  - NEVER send full data.txt to OpenRouter
  - NEVER let LLM generate college recommendations
  - NEVER invent college names, codes, cutoffs, fees, placements, or seats
  - Category (community) is REQUIRED before personalized recommendations
  - is_any() MUST be used for every filter check — never bare `if value:`
  - filter_and_rank() is the ONLY source of college recommendations
  - Two filter modes: STRICT (cutoff <= student) and WITH_REACH (+5 band)
"""

import difflib
import os
import re
import sys
import traceback
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
MAX_HISTORY        = 20   # max conversation turns kept in LLM context
REACH_BAND         = 5    # marks above student cutoff included as "Reach"


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
# COMMUNITY NORMALIZATION
# ================================================================

# Canonical tags that MUST match exactly what _norm_community() produces.
# data.txt stores: OC, BC, BCM, MBC, SC, SCA, ST
# "MBC/DNC" in data.txt → normalized to "MBC" by _norm_community().
# VALID_CATEGORIES must also map user input → "MBC" (not "MBC/DNC").
CANONICAL_COMMUNITIES = {"OC", "BC", "BCM", "MBC", "SC", "SCA", "ST"}

COMMUNITY_DISPLAY: Dict[str, str] = {
    "OC":  "Open Category (OC)",
    "BC":  "Backward Class (BC)",
    "BCM": "Backward Class Muslim (BCM)",
    "MBC": "Most Backward Class / MBC/DNC",
    "SC":  "Scheduled Caste (SC)",
    "SCA": "SC Arunthathiyar (SCA)",
    "ST":  "Scheduled Tribe (ST)",
}


def _norm_community(val: str) -> str:
    """Normalise a community value from data.txt to canonical form."""
    v = val.strip().upper()
    return {
        "MBC/DNC": "MBC",
        "MBCDNC":  "MBC",
        "MBC DNC": "MBC",
        "BCGM":    "BCM",
        "BCG":     "BC",
    }.get(v, v)


# ================================================================
# RECORD PARSING
# ================================================================

def parse_records(raw: str) -> List[Dict]:
    """
    Split raw text on '---' separators; parse each chunk into a dict.
    Returns only records that have at least: code, name, branch.
    """
    records: List[Dict] = []
    for chunk in re.split(r"\n-{3,}\n", raw):
        chunk = chunk.strip()
        if not chunk:
            continue
        # Skip the file header block
        if "KNOWLEDGE BASE" in chunk or "Purpose:" in chunk:
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
    except Exception as exc:
        print(f"[chatbot] ERROR loading data: {exc}", file=sys.stderr)
        return []


# Loaded ONCE at module import — effectively cached for the lifetime of the process.
# NEVER sent wholesale to OpenRouter.
ALL_RECORDS: List[Dict] = _load_all()


# ================================================================
# DATA VALIDATION  (run once at startup)
# ================================================================

def validate_data() -> str:
    """Return a Markdown-formatted field-level validation report."""
    if DATA_FILE is None:
        return "⚠️ **No data file found.** Place `data.txt` beside `app.py`."
    if not ALL_RECORDS:
        return f"⚠️ `{DATA_FILE.name}` found but **0 records** were parsed."

    total          = len(ALL_RECORDS)
    missing_code   = sum(1 for r in ALL_RECORDS if not r.get("code"))
    missing_name   = sum(1 for r in ALL_RECORDS if not r.get("name"))
    missing_dist   = sum(1 for r in ALL_RECORDS if not r.get("district"))
    missing_branch = sum(1 for r in ALL_RECORDS if not r.get("branch"))
    bad_cutoff     = sum(1 for r in ALL_RECORDS if r.get("cutoff") is None)
    bad_comm       = sum(1 for r in ALL_RECORDS if r.get("community") not in CANONICAL_COMMUNITIES)

    return (
        f"**Records loaded:** {total:,}\n"
        f"- Missing college codes: {missing_code}\n"
        f"- Missing college names: {missing_name}\n"
        f"- Missing districts: {missing_dist}\n"
        f"- Missing branches: {missing_branch}\n"
        f"- Invalid/missing cutoffs: {bad_cutoff}\n"
        f"- Unknown communities: {bad_comm}"
    )


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
    True when value means 'no preference / no filter'.
    Handles: None, '', 'any', 'all', 'no preference'.
    ALWAYS use this instead of bare `if value:` for filter checks.
    """
    if value is None:
        return True
    return str(value).strip().lower() in {"", "any", "all", "no preference"}


# ================================================================
# LOOKUP TABLES
# ================================================================

DISTRICT_ALIASES: Dict[str, str] = {
    "chennai": "Chennai", "chennai district": "Chennai", "madras": "Chennai",
    "kanchipuram": "Kancheepuram", "kancheepuram": "Kancheepuram", "kanchi": "Kancheepuram", "kancheepuram district": "Kancheepuram", "kanchipuram district": "Kancheepuram",
    "chengalpattu": "Chengalpattu", "chengalpet": "Chengalpattu", "chengalpattu district": "Chengalpattu", "chengalpat": "Chengalpattu",
    "coimbatore": "Coimbatore", "coimbatore district": "Coimbatore", "cbe": "Coimbatore", "kovai": "Coimbatore",
    "salem": "Salem", "salem district": "Salem",
    "madurai": "Madurai", "madurai district": "Madurai",
    "trichy": "Tiruchirappalli", "trichy district": "Tiruchirappalli", "tiruchi": "Tiruchirappalli", "tiruchi district": "Tiruchirappalli",
    "tiruchirappalli": "Tiruchirappalli", "tiruchirappalli district": "Tiruchirappalli", "tiruchirapalli": "Tiruchirappalli", "thiruchirappalli": "Tiruchirappalli",
    "tirunelveli": "Tirunelveli", "tirunelveli district": "Tirunelveli", "nellai": "Tirunelveli", "nellai district": "Tirunelveli", "thirunelveli": "Tirunelveli",
    "erode": "Erode", "erode district": "Erode",
    "vellore": "Vellore", "vellore district": "Vellore",
    "thanjavur": "Thanjavur", "thanjavur district": "Thanjavur", "tanjore": "Thanjavur",
    "tiruppur": "Tiruppur", "tiruppur district": "Tiruppur", "tirupur": "Tiruppur", "thiruppur": "Tiruppur", "thirupur": "Tiruppur",
    "dindigul": "Dindigul", "dindigul district": "Dindigul",
    "virudhunagar": "Virudhunagar", "virudhunagar district": "Virudhunagar",
    "thoothukudi": "Thoothukudi", "thoothukudi district": "Thoothukudi", "tuticorin": "Thoothukudi", "tuticorin district": "Thoothukudi",
    "sivaganga": "Sivagangai", "sivagangai": "Sivagangai", "sivaganga district": "Sivagangai", "sivagangai district": "Sivagangai",
    "namakkal": "Namakkal", "namakkal district": "Namakkal",
    "karur": "Karur", "karur district": "Karur",
    "cuddalore": "Cuddalore", "cuddalore district": "Cuddalore",
    "villupuram": "Viluppuram", "viluppuram": "Viluppuram", "villupuram district": "Viluppuram", "viluppuram district": "Viluppuram",
    "villpuram": "Viluppuram", "vilupuram": "Viluppuram", "villapuram": "Viluppuram", "vilapuram": "Viluppuram", "villupuram disrtict": "Viluppuram", "villpuram disrtict": "Viluppuram",
    "kallakurichi": "Kallakurichi", "kallakurichi district": "Kallakurichi",
    "dharmapuri": "Dharmapuri", "dharmapuri district": "Dharmapuri",
    "krishnagiri": "Krishnagiri", "krishnagiri district": "Krishnagiri",
    "thiruvallur": "Tiruvallur", "tiruvallur": "Tiruvallur", "tiruvallur district": "Tiruvallur", "thiruvallur district": "Tiruvallur",
    "tiruvannamalai": "Tiruvannamalai", "thiruvannamalai": "Tiruvannamalai", "tiruvannamalai district": "Tiruvannamalai",
    "nagapattinam": "Nagapattinam", "nagapattinam district": "Nagapattinam",
    "mayiladuthurai": "Mayiladuthurai", "mayiladuthurai district": "Mayiladuthurai", "mayavaram": "Mayiladuthurai",
    "perambalur": "Perambalur", "perambalur district": "Perambalur",
    "ariyalur": "Ariyalur", "ariyalur district": "Ariyalur",
    "pudukkottai": "Pudukkottai", "pudukkottai district": "Pudukkottai",
    "ramanathapuram": "Ramanathapuram", "ramanathapuram district": "Ramanathapuram", "ramnad": "Ramanathapuram",
    "the nilgiris": "The Nilgiris", "nilgiris": "The Nilgiris", "ooty": "The Nilgiris", "nilgiris district": "The Nilgiris",
    "tenkasi": "Tenkasi", "tenkasi district": "Tenkasi",
    "theni": "Theni", "theni district": "Theni",
    "ranipet": "Ranipet", "ranipet district": "Ranipet",
    "tirupathur": "Tirupathur", "tirupattur": "Tirupathur", "tirupathur district": "Tirupathur", "tirupattur district": "Tirupathur",
    "tiruvarur": "Tiruvarur", "thiruvarur": "Tiruvarur", "tiruvarur district": "Tiruvarur",
    "kanniyakumari": "Kanniyakumari", "kanyakumari": "Kanniyakumari", "kanniyakumari district": "Kanniyakumari",
}

DISTRICTS: List[str] = sorted(set(DISTRICT_ALIASES.values()))

# branch_display → {"user": [input aliases], "data": [data.txt keywords]}
BRANCH_MAP: Dict[str, Dict[str, List[str]]] = {
    "CSE": {
        "user": [
            "computer science and engineering", "computer science engineering",
            "computer science", "cse", "cs", "computer science and design",
            "computer science and technology", "computer and communication engineering",
            "software engineering"
        ],
        "data": ["computer science", "computer and communication"]
    },
    "ECE": {
        "user": [
            "electronics and communication engineering",
            "electronics and communication", "electronics & communication",
            "electronics and computer engineering", "electronics engineering", "ece"
        ],
        "data": ["electronics and communication", "electronics and computer", "electronics engineering"]
    },
    "EEE": {
        "user": [
            "electrical and electronics engineering",
            "electrical and electronics", "electrical & electronics", "eee",
            "electrical and computer engineering"
        ],
        "data": ["electrical and electronics", "electrical and computer"]
    },
    "AI & DS": {
        "user": [
            "artificial intelligence and data science",
            "artificial intelligence & data science",
            "ai and ds", "ai & ds", "ai&ds", "ai ds", "aids", "data science"
        ],
        "data": ["artificial intelligence and data science", "data science"]
    },
    "AI & ML": {
        "user": [
            "artificial intelligence and machine learning",
            "artificial intelligence & machine learning",
            "ai and ml", "ai & ml", "ai&ml", "ai ml", "aiml",
            "artificial intelligence", "machine learning"
        ],
        "data": [
            "artificial intelligence and machine learning",
            "ai and machine learning", "aiml"
        ]
    },
    "IT": {
        "user": ["information technology", "cloud computing"],
        "data": ["information technology"]
    },
    "MECH": {
        "user": ["mechanical engineering", "mechanical", "mech"],
        "data": ["mechanical engineering", "mechanical and smart", "mechanical and automation"]
    },
    "CIVIL": {
        "user": ["civil engineering", "civil and structutural", "civil"],
        "data": ["civil engineering", "civil and structutural"]
    },
    "AGRI": {
        "user": ["agriculture engineering", "agricultural engineering", "agriculture", "agri"],
        "data": ["agriculture", "agricultural"]
    },
    "CHEM": {
        "user": ["chemical engineering", "chemical and electro chemical", "chemical", "chem"],
        "data": ["chemical engineering", "chemical and electro"]
    },
    "PETROCHEM": {
        "user": [
            "petrochemical engineering", "petrochemical", "petrochem",
            "petro chemical technology", "petroleum engineering",
            "petroleum engineering and technology"
        ],
        "data": ["petro chemical", "petrochemical", "petroleum"]
    },
    "ROBOTICS": {
        "user": [
            "robotics and automation", "robotics & automation",
            "robotics and artificial intelligence", "robotics", "automation"
        ],
        "data": ["robotics and automation", "robotics and artificial", "robotics"]
    },
    "MECHAT": {
        "user": [
            "mechatronics engineering", "mechatronics",
            "mechanical and mechatronics engineering"
        ],
        "data": ["mechatronics"]
    },
    "FOOD": {
        "user": ["food technology", "food tech", "food processing"],
        "data": ["food technology"]
    },
    "INDENG": {
        "user": ["industrial engineering", "industrial engg", "industrial engineering and management"],
        "data": ["industrial engineering"]
    },
    "BIO": {
        "user": [
            "biomedical engineering", "biomedical", "bio medical engineering", "bio medical",
            "biotechnology", "biotech", "bio technology", "bio technology and bio chemical",
            "medical electronics engineering", "bio"
        ],
        "data": ["biomedical", "bio medical", "biotechnology", "bio technology", "medical electronics"]
    },
    "CSBS": {
        "user": [
            "computer science and business systems", "computer science and business system",
            "computer science and bussiness system", "csbs"
        ],
        "data": ["computer science and business", "computer science and bussiness"]
    },
    "CYBER": {
        "user": [
            "cyber security engineering", "cyber security", "cybersecurity"
        ],
        "data": ["cyber security"]
    },
    "EIE": {
        "user": [
            "electronics and instrumentation engineering",
            "electronics and instrumentation", "instrumentation and control engineering",
            "eie", "ice"
        ],
        "data": ["electronics and instrumentation", "instrumentation and control"]
    },
    "MARINE": {
        "user": ["marine engineering", "marine"],
        "data": ["marine engineering"]
    },
    "AERO": {
        "user": ["aeronautical engineering", "aerospace engineering", "aeronautical", "aerospace", "aero"],
        "data": ["aeronautical", "aerospace"]
    },
    "AUTO": {
        "user": ["automobile engineering", "automobile", "auto"],
        "data": ["automobile"]
    },
    "TEXTILE": {
        "user": ["textile technology", "textile engineering", "fashion technology", "apparel technology", "textile"],
        "data": ["textile", "fashion", "apparel"]
    },
    "PRINTING": {
        "user": ["printing technology", "printing"],
        "data": ["printing"]
    },
}

# User input → canonical community
# IMPORTANT: must map to "MBC" not "MBC/DNC" — that is what _norm_community() produces.
VALID_CATEGORIES: Dict[str, str] = {
    "oc":          "OC",
    "bc":          "BC",
    "bcm":         "BCM",
    "mbc":         "MBC",
    "mbc/dnc":     "MBC",
    "mbc & dnc":   "MBC",
    "mbc and dnc": "MBC",
    "dnc":         "MBC",
    "sc":          "SC",
    "sca":         "SCA",
    "st":          "ST",
}

# Branch abbreviations used only to guard "vs" intent detection
_BRANCH_ABBREVS = frozenset(
    [b.lower() for b in BRANCH_MAP]
    + [a for info in BRANCH_MAP.values() for a in info["user"]]
)


# ================================================================
# NORMALIZATION HELPERS
# ================================================================

def normalize_district(value) -> Optional[str]:
    if not value:
        return None
    val_l = str(value).strip().lower()
    if val_l in {"any", "all", "all districts", "none", ""}:
        return None
    return DISTRICT_ALIASES.get(val_l, str(value).strip().title())


def district_matches(record_district: Optional[str], requested: Optional[str]) -> bool:
    if not requested or is_any(requested):
        return True
    actual = normalize_district(record_district)
    want   = normalize_district(requested)
    if not actual or not want:
        return True
    return actual.lower() == want.lower()


def branch_matches(record_branch: Optional[str], requested: Optional[str]) -> bool:
    if not requested or is_any(requested):
        return True
    if not record_branch:
        return False
    rec_l    = record_branch.lower()
    data_kws = BRANCH_MAP.get(requested, {}).get("data", [requested.lower()])
    return any(kw in rec_l for kw in data_kws)


def college_type_matches(record_name: Optional[str], requested: Optional[str]) -> bool:
    if not requested or is_any(requested):
        return True
    if not record_name:
        return False
    if requested == "Autonomous":
        return "autonomous" in record_name.lower()
    return _type_from_name(record_name).lower() == requested.lower()


# ================================================================
# EXTRACTION FUNCTIONS
# ================================================================

def extract_pcm_marks(text: str) -> Dict[str, Optional[float]]:
    """
    Extract individual marks for Mathematics, Physics, Chemistry.
    Returns dict: {'maths': float|None, 'physics': float|None, 'chemistry': float|None}
    """
    text_l = text.lower()
    cleaned = re.sub(r"[,;]", " ", text_l)

    math_patterns = [
        r"\b(?:maths?|mathematics|mat)\b\s*(?:mark\s+is|mark|marks\s+is|marks|is|score\s+is|score|[:=])?\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)",
        r"\bm\s*[:=]\s*(\d{1,3}(?:\.\d+)?)",
    ]
    phy_patterns = [
        r"\b(?:physics?|phy)\b\s*(?:mark\s+is|mark|marks\s+is|marks|is|score\s+is|score|[:=])?\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)",
        r"\bp\s*[:=]\s*(\d{1,3}(?:\.\d+)?)",
    ]
    chem_patterns = [
        r"\b(?:chemistry?|chem|che)\b\s*(?:mark\s+is|mark|marks\s+is|marks|is|score\s+is|score|[:=])?\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)",
        r"\bc\s*[:=]\s*(\d{1,3}(?:\.\d+)?)",
    ]

    def find_mark(patterns: List[str]) -> Optional[float]:
        for p in patterns:
            m = re.search(p, cleaned)
            if m:
                try:
                    val = float(m.group(1))
                    if 0 <= val <= 100:
                        return val
                except ValueError:
                    pass
        return None

    return {
        "maths": find_mark(math_patterns),
        "physics": find_mark(phy_patterns),
        "chemistry": find_mark(chem_patterns),
    }


def extract_cutoff(text: str) -> Optional[float]:
    pcm = extract_pcm_marks(text)
    if pcm["maths"] is not None and pcm["physics"] is not None and pcm["chemistry"] is not None:
        return pcm["maths"] + (pcm["physics"] / 2.0) + (pcm["chemistry"] / 2.0)
    if any(v is not None for v in pcm.values()):
        return None

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
    # Standalone 2–3 digit number in range 50–200
    for m in re.finditer(r"\b(\d{2,3}(?:\.\d+)?)\b", text_l):
        try:
            v = float(m.group(1))
            if 50 <= v <= 200:
                return v
        except ValueError:
            pass
    return None


def extract_category(text: str) -> Optional[str]:
    """
    Return canonical community if recognized, None if not present.
    Handles abbreviations and natural-language phrases.
    Does NOT guess — unknown input returns None (use suggest_category separately).
    """
    text_l = text.lower().strip()

    # Whole-message direct lookup
    if text_l in VALID_CATEGORIES:
        return VALID_CATEGORIES[text_l]

    # Token-level lookup with punctuation stripping
    for tok in re.split(r"[\s,;/?!\.\(\)\[\]]+", text_l):
        if tok in VALID_CATEGORIES:
            return VALID_CATEGORIES[tok]

    # Natural-language phrases (longest/most specific first)
    if "backward class muslim" in text_l or "bc muslim" in text_l:
        return "BCM"
    if "sc arunthathiyar" in text_l or "arunthathiyar" in text_l:
        return "SCA"
    if "most backward class" in text_l or "most backward" in text_l or "denotified" in text_l:
        return "MBC"
    if "backward class" in text_l:
        return "BC"
    if "scheduled caste" in text_l:
        return "SC"
    if "scheduled tribe" in text_l:
        return "ST"
    if "open category" in text_l or "open class" in text_l or "general category" in text_l:
        return "OC"

    return None


def suggest_category(raw: str) -> str:
    """Friendly error for an unrecognized category input."""
    raw_u = raw.strip().upper()
    return (
        f"I couldn't identify **`{raw_u}`** as a TNEA community/category.\n\n"
        "Please choose one of:\n\n"
        "| Code | Community |\n"
        "|:-----|:----------|\n"
        "| OC   | Open Category |\n"
        "| BC   | Backward Class |\n"
        "| BCM  | Backward Class Muslim |\n"
        "| MBC  | Most Backward Class / MBC/DNC |\n"
        "| SC   | Scheduled Caste |\n"
        "| SCA  | SC Arunthathiyar |\n"
        "| ST   | Scheduled Tribe |\n\n"
        "Just type the code — e.g. `BC` or `MBC`"
    )


def extract_district(text: str) -> Optional[str]:
    text_l = text.lower().strip()
    if re.search(r"\bany\s+districts?\b|\bno\s+district\b", text_l):
        return "Any"

    matches: List[Tuple[int, int, str]] = []  # (start_pos, -length, canonical)
    for alias, canonical in DISTRICT_ALIASES.items():
        m = re.search(r"\b" + re.escape(alias) + r"\b", text_l)
        if m:
            matches.append((m.start(), -len(alias), canonical))

    if matches:
        matches.sort()
        return matches[0][2]

    # Fuzzy match individual words against district alias keys
    clean_words = [re.sub(r"[^\w]", "", w) for w in text_l.split() if len(w) >= 4]
    stop_dist = {"college", "district", "disrtict", "engineering", "branch", "cutoff", "marks", "government", "private", "aided"}
    for w in clean_words:
        if w in stop_dist:
            continue
        close = difflib.get_close_matches(w, list(DISTRICT_ALIASES.keys()), n=1, cutoff=0.75)
        if close:
            return DISTRICT_ALIASES[close[0]]

    return None


def extract_branch(text: str) -> Optional[str]:
    text_l = text.lower().strip()
    if re.search(r"\bany\s+branch(?:es)?\b|\ball\s+branch(?:es)?\b|\bno\s+branch\b", text_l):
        return "Any"

    # Avoid matching "it companies", "it sector", "it jobs", "it field", "it industry" as IT branch
    cleaned_for_it = re.sub(r"\b(?:in\s+)?it\s+(?:companies|company|industry|jobs?|sector|field|placement|placements)\b", " ", text, flags=re.IGNORECASE)
    cleaned_for_it_l = cleaned_for_it.lower()

    cleaned = text_l.replace("-", " ")
    cleaned = re.sub(r"[/&]", " & ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)

    candidates: List[Tuple[int, str]] = []

    # IT check
    if re.search(r"\bIT\b", cleaned_for_it) or "information technology" in cleaned_for_it_l or "cloud computing" in cleaned_for_it_l or re.fullmatch(r"it", text_l):
        candidates.append((2, "IT"))

    for branch, info in BRANCH_MAP.items():
        if branch == "IT":
            continue
        for alias in sorted(info["user"], key=len, reverse=True):
            if alias in {"me", "in"}:
                continue
            alias_clean = alias.replace("-", " ").replace("&", " & ")
            alias_clean = re.sub(r"\s+", " ", alias_clean)
            if re.search(r"\b" + re.escape(alias_clean) + r"\b", cleaned) or re.search(r"\b" + re.escape(alias) + r"\b", text_l):
                candidates.append((len(alias), branch))
                break

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    # Standalone "AI" → AI & Data Science (the common single "AI" TNEA branch).
    # Only reached when no longer alias matched, so phrases like "AI and ML",
    # "AI & DS", "aiml", "aids", "artificial intelligence" are handled earlier.
    if re.search(r"\bai\b", text_l) and not re.search(
        r"\b(?:ai\s*[&+]\s*(?:ml|ds)|ai\s+and\s+(?:ml|ds)|ai\s+(?:ml|ds)\b|aiml|aids)\b",
        text_l,
    ):
        return "AI & DS"
    return None


def extract_ownership(text: str) -> Optional[str]:
    """Extract college ownership: Government, Government Aided, or Self Financing (Private)."""
    text_l = text.lower().replace("-", " ")
    if "self financing" in text_l or "private" in text_l or "self finance" in text_l:
        return "Self Financing"
    if "government aided" in text_l or "govt aided" in text_l or \
       ("aided" in text_l and "government" not in text_l and "govt" not in text_l):
        return "Government Aided"
    if "government" in text_l or "govt" in text_l:
        return "Government"
    return None


def extract_autonomous(text: str) -> Optional[bool]:
    """Extract whether autonomous status is requested (True/False/None)."""
    text_l = text.lower()
    if "non autonomous" in text_l or "non-autonomous" in text_l or "nonautonomous" in text_l:
        return False
    if "autonomous" in text_l:
        return True
    return None


def extract_college_type(text: str) -> Optional[str]:
    """
    Backwards-compatible college type extractor.
    Returns ownership if present, otherwise 'Autonomous' if autonomous requested.
    """
    ow = extract_ownership(text)
    if ow:
        return ow
    auto = extract_autonomous(text)
    if auto is True:
        return "Autonomous"
    return None


def extract_college_code(text: str) -> Optional[str]:
    """Extract a 4–6 digit college code from explicit mention patterns."""
    text_l = text.lower()
    for p in [
        r"college\s+code\s*[:#]?\s*(\d{4,6})",
        r"college\s+#?\s*(\d{4,6})\b",
        r"\bcode\s*[:#]?\s*(\d{4,6})\b",
        r"(?:about|details?|tell me about|info(?:rmation)?(?:\s+(?:about|on|for))?)\s+college\s+(\d{4,6})",
        r"(?:about|details?|tell me about)\s+(\d{4,6})\b",
    ]:
        m = re.search(p, text_l)
        if m and float(m.group(1)) > 200:
            return m.group(1)
    # Bare 4–6 digit number when "college" or "code" is in the message
    if any(kw in text_l for kw in ["college", "code", "about"]):
        m = re.search(r"\b(\d{4,6})\b", text_l)
        if m and float(m.group(1)) > 200:
            return m.group(1)
    return None


def extract_result_limit(text: str) -> Optional[int]:
    m = re.search(r"\btop\s+(\d+)\b", text.lower())
    if m:
        return max(1, min(int(m.group(1)), 50))
    m = re.search(r"\b(?:show|give|list|get)\s+(?:me\s+)?(\d+)\b", text.lower())
    if m:
        return max(1, min(int(m.group(1)), 50))
    return None


# ================================================================
# COLLEGE TYPE DETECTION FROM NAME
# ================================================================

_AIDED_KEYWORDS = [
    "psg college of technology", "coimbatore institute of technology",
    "thiagarajar college of engineering", "aided",
]

_GOV_KEYWORDS = [
    "government college of engineering", "government college of technology",
    "government engineering college", "university college of engineering",
    "college of engineering, anna", "college of engineering, guindy",
    "college of engineering guindy", "alagappa college of technology",
    "alagappa chettiar government college", "madras institute of technology",
    "government college", "govt college", "anna university regional campus",
    "anna university", "annamalai university", "university voc", "university v.o.c",
    " gce ", " gct ",
]


def _type_from_name(name: str) -> str:
    n = name.lower()
    if any(kw in n for kw in _AIDED_KEYWORDS):
        return "Government Aided"
    if any(kw in n for kw in _GOV_KEYWORDS):
        return "Government"
    return "Self Financing"


def _short_name(full_name: str) -> str:
    """College name without address."""
    return full_name.split(",")[0].strip()


# ================================================================
# STATE
# ================================================================

def new_state() -> Dict:
    """Return a fresh, fully initialised state dict."""
    return {
        "cutoff":                None,
        "category":              None,
        "district":              None,   # None = any
        "branch":                None,   # None = any
        "college_type":          None,   # kept for backward compatibility
        "ownership":             None,   # Government | Government Aided | Self Financing
        "autonomous":            None,   # True | False | None
        "limit":                 10,
        "recommendation_offset": 0,
        "selected_college":      None,
        "last_recommendations":  [],     # for "why this college?"
        "_flow_step":            0,      # 0=init 1=cat 2=dist 3=branch 4=type 5=done
    }


def clear_filters(state: Dict) -> None:
    """Clear district/branch/ownership/autonomous/selected_college. Retain cutoff + category."""
    state["district"]              = None
    state["branch"]                = None
    state["college_type"]          = None
    state["ownership"]             = None
    state["autonomous"]            = None
    state["selected_college"]      = None
    state["recommendation_offset"] = 0
    state["last_recommendations"]  = []
    # Reset guided flow to appropriate step
    if state.get("cutoff") is not None and state.get("category") is not None:
        state["_flow_step"] = 2
    elif state.get("cutoff") is not None:
        state["_flow_step"] = 1
    else:
        state["_flow_step"] = 0


def reset_state(state: Dict) -> None:
    """Clear ALL state."""
    state.clear()
    state.update(new_state())


# ================================================================
# PROFILE + PER-MESSAGE QUERY STATE
# ================================================================

def new_user_profile() -> Dict:
    """Persistent facts a student explicitly shares with Saarthi."""
    return {"cutoff": None, "category": None, "district": None,
            "branch": None, "college_type": None}


def new_current_query() -> Dict:
    """Filters for one message only.  Never reuse this for the next message."""
    return {"intent": None, "college": None, "college_code": None,
            "district": None, "branches": [], "cutoff": None,
            "category": None, "college_type": None}


def reset_current_query(current_query: Dict) -> None:
    current_query.clear()
    current_query.update(new_current_query())


def extract_branches(text: str) -> List[str]:
    """Return every requested canonical branch, preserving the user's order."""
    text_l = text.lower().replace("-", " ")
    found: List[Tuple[int, str]] = []
    for branch, info in BRANCH_MAP.items():
        aliases = [branch.lower(), *info["user"]]
        for alias in sorted(set(aliases), key=len, reverse=True):
            if alias in {"it", "me", "in"}:
                pattern = r"\\b" + re.escape(alias) + r"\\b"
            else:
                pattern = r"\\b" + re.escape(alias.replace("&", " & ")) + r"\\b"
            match = re.search(pattern, text_l.replace("&", " & "))
            if match:
                found.append((match.start(), branch))
                break
    found.sort()
    return list(dict.fromkeys(branch for _, branch in found))


def classify_query_intent(text: str) -> str:
    """Intent classifier used before any CSV filtering in the modern router."""
    t = text.lower().strip()
    if extract_pcm_marks(text) and all(extract_pcm_marks(text).values()):
        return "cutoff_calculation"
    if re.search(r"\\b(compare|versus|vs)\\b", t):
        return "college_comparison"
    if re.search(r"\\b(bus|travel|route|transport)\\b", t):
        return "travel_or_bus"
    if re.match(r"^(what is|what are|how is|how does|why is|explain)\\b", t):
        return "tnea_general" if "tnea" in t else "engineering_general"
    if t in {"aiml", "ai ml", "ai & ml", "cse", "ece", "it", "di"}:
        return "clarification"
    if re.search(r"\\b(can i get|eligible|eligibility|chance|can i join|which colleges can i get)\\b", t):
        return "college_eligibility"
    if re.search(r"\\b(show|list|any|what branches|which branches)\\b", t):
        target = extract_target_college_query("can i get " + text) or extract_college_code(text)
        if target and ("branch" in t or "branches" in t):
            return "college_branch_search"
        return "branch_search" if extract_branches(text) else "list_colleges"
    if is_college_eligibility_query(text):
        return "college_eligibility"
    return "unknown"


def _active_filters(query: Dict) -> str:
    rows = []
    for label, key in (("College", "college"), ("District", "district"),
                       ("Branch", "branches"), ("Cutoff", "cutoff"),
                       ("Category", "category"), ("College type", "college_type")):
        value = query.get(key)
        if value:
            rows.append(f"{label}: {', '.join(value) if isinstance(value, list) else value}")
    return "\n".join(rows) if rows else "None"


def _query_records(query: Dict, eligibility: bool = False) -> List[Dict]:
    """The single data access path for current-query listing/searches."""
    records = ALL_RECORDS[:]
    if query.get("college_code"):
        records = [r for r in records if r.get("code") == query["college_code"]]
    if query.get("district"):
        records = [r for r in records if district_matches(r.get("district"), query["district"])]
    branches = query.get("branches") or []
    if branches:
        records = [r for r in records if any(branch_matches(r.get("branch"), branch) for branch in branches)]
    if query.get("college_type"):
        records = [r for r in records if college_type_matches(r.get("name"), query["college_type"])]
    if eligibility:
        category = query.get("category") or "OC"
        records = [r for r in records if r.get("community") == category]
        cutoff = query.get("cutoff")
        if cutoff is not None:
            records = [r for r in records if r.get("cutoff") is not None and r["cutoff"] <= cutoff]
    return records


def _format_current_results(records: List[Dict], query: Dict, eligibility: bool = False) -> str:
    seen, rows = set(), []
    for record in sorted(records, key=lambda r: (r.get("name", ""), r.get("branch", ""))):
        key = (record.get("code"), record.get("branch"))
        if key not in seen:
            seen.add(key)
            rows.append(record)
    if not rows:
        return "No results found after checking the current query filters:\n\n" + _active_filters(query)
    lines = ["**Search filters (this question only):**", _active_filters(query), "",
             "| College | Code | District | Branch | Closing cutoff |",
             "|:---|:---:|:---|:---|---:|"]
    for r in rows[:50]:
        lines.append(f"| {_short_name(r.get('name', ''))} | {r.get('code', '')} | {r.get('district', '')} | {r.get('branch', '')} | {r.get('cutoff', 0):g} |")
    if len(rows) > 50:
        lines.append(f"\nShowing 50 of {len(rows)} matching branch records.")
    if eligibility:
        lines.append("\n> Historical cutoff estimate only; admission is not guaranteed.")
    return "\n".join(lines)


def process_current_query(user_message: str, profile: Dict, query: Dict, chat_history: List[Dict]) -> Tuple[str, List[Dict]]:
    """Modern router: profile facts and transient filters intentionally never share a dict."""
    raw_lower = user_message.lower().strip()
    if raw_lower in {"reset", "restart", "reset chat", "new chat"}:
        profile.clear()
        profile.update(new_user_profile())
        reset_current_query(query)
        return "Chat and profile reset. What is your TNEA cutoff?", []
    if "clear filter" in raw_lower or "clear search" in raw_lower:
        reset_current_query(query)
        return "Current search filters cleared. Your saved cutoff and category are unchanged.", chat_history

    reset_current_query(query)
    intent = classify_query_intent(user_message)
    query["intent"] = intent
    pcm = extract_pcm_marks(user_message)
    explicit_cutoff = extract_cutoff(user_message)
    explicit_category = extract_category(user_message)
    if explicit_cutoff is not None:
        profile["cutoff"] = explicit_cutoff
        query["cutoff"] = explicit_cutoff
    if explicit_category:
        profile["category"] = explicit_category
        query["category"] = explicit_category
    query["district"] = extract_district(user_message)
    query["branches"] = extract_branches(user_message)
    query["college_type"] = extract_ownership(user_message)

    def done(answer: str) -> Tuple[str, List[Dict]]:
        return answer, chat_history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": answer}]

    if intent == "cutoff_calculation":
        cutoff = pcm["maths"] + pcm["physics"] / 2 + pcm["chemistry"] / 2
        profile["cutoff"] = cutoff
        query["cutoff"] = cutoff
        return done(f"Your TNEA cutoff is **{cutoff:g}**.\n\n`{pcm['maths']:g} + {pcm['physics']:g}/2 + {pcm['chemistry']:g}/2 = {cutoff:g}`")
    if intent == "clarification":
        return done("What would you like to know about that branch or term? For example: `List AIML colleges in Chennai`.")
    if intent == "travel_or_bus":
        return done("I don't have verified live bus timing data for this route.")
    if intent in {"tnea_general", "engineering_general"}:
        if user_message.lower().strip() == "what is di?":
            return done("`DI` can mean different things. Do you mean a TNEA branch, a subject, or something else?")
        return done(ask_openrouter(user_message, new_state(), "Answer this general question concisely. Do not search or invent TNEA college data.", chat_history))
    if intent == "college_comparison":
        return done(compare_colleges(user_message))

    target = extract_target_college_query(user_message) or extract_college_code(user_message)
    # List/detail wording does not use eligibility verbs. Prefer an explicit alias
    # found anywhere in the question when a parsed phrase cannot be resolved.
    target_codes = resolve_colleges(target)[0] if target else []
    if not target_codes:
        for alias in sorted(EXPLICIT_COLLEGE_ALIASES, key=len, reverse=True):
            if re.search(r"\b" + re.escape(alias) + r"\b", user_message, re.IGNORECASE):
                target = alias
                target_codes = resolve_colleges(alias)[0]
                break
    if target:
        codes = target_codes
        if codes:
            query["college_code"] = codes[0]
            query["college"] = _short_name(COLLEGE_CODE_NAME_MAP.get(codes[0], ""))

    if intent == "college_branch_search" and query.get("college_code"):
        return done(_format_current_results(_query_records(query), query))

    if intent == "college_eligibility" and query.get("college_code"):
        query["cutoff"] = query.get("cutoff") if query.get("cutoff") is not None else profile.get("cutoff")
        query["category"] = query.get("category") or profile.get("category") or "OC"
        if query["cutoff"] is None:
            return done(f"Please share your cutoff to check **{query['college']}**.")
        all_college = _query_records(query)
        category_records = [r for r in all_college if r.get("community") == query["category"]]
        if not category_records:
            category_records = [r for r in all_college if r.get("community") == "OC"]
        if not category_records:
            return done("No verified cutoff record is available for that college and category.")
        lowest = min(r["cutoff"] for r in category_records if r.get("cutoff") is not None)
        gap = lowest - query["cutoff"]
        verdict = "Reach" if gap <= 5 and gap > 0 else ("Possible" if gap <= 0 else "Unlikely")
        icon = "✅" if verdict == "Possible" else ("🎯" if verdict == "Reach" else "❌")
        return done(f"### {query['college']} (Code {query['college_code']})\n\nWith a **{query['cutoff']:g}** cutoff in **{query['category']}**, getting this college is **{verdict.lower()}** based on historical TNEA data.\n\nLowest listed closing cutoff: **{lowest:g}**. Gap: **{gap:.1f}** marks.\n\n**Verdict: {icon} {verdict}**\n\n> Historical estimate only; admission depends on the counselling year, competition and seat availability. Ask `Show all branches in {query['college']}` for the full branch table.")

    if intent in {"list_colleges", "branch_search", "college_eligibility"}:
        eligibility = intent == "college_eligibility"
        if eligibility:
            query["cutoff"] = query.get("cutoff") if query.get("cutoff") is not None else profile.get("cutoff")
            query["category"] = query.get("category") or profile.get("category")
            if query["cutoff"] is None:
                return done("Please share your TNEA cutoff (and category, if known) to check eligibility.")
        return done(_format_current_results(_query_records(query, eligibility), query, eligibility))
    return done(ask_openrouter(user_message, new_state(), "Answer only the user's question. Never invent TNEA data.", chat_history))


# ================================================================
# FILTER AND RANK  —  pure Python, zero LLM
# ================================================================

def filter_and_rank(
    state: Dict,
    limit: int = 10,
    with_reach: bool = False,
) -> List[Dict]:
    """
    Filter ALL_RECORDS by student preferences, rank by closeness to cutoff.

    STRICT mode  (with_reach=False): college_cutoff <= student_cutoff
    REACH mode   (with_reach=True):  college_cutoff <= student_cutoff + REACH_BAND

    Tiers are assigned in build_tiered_response(); this function only filters + ranks.
    """
    if not ALL_RECORDS:
        return []

    student_cutoff = state.get("cutoff")
    category       = state.get("category")
    district       = state.get("district")
    branch         = state.get("branch")
    ownership      = state.get("ownership") or state.get("college_type")
    autonomous     = state.get("autonomous")

    candidates = ALL_RECORDS.copy()

    # 1. DISTRICT — hard filter
    if district and not is_any(district):
        candidates = [r for r in candidates if district_matches(r.get("district"), district)]

    # 2. BRANCH — hard filter
    if branch and not is_any(branch):
        candidates = [r for r in candidates if branch_matches(r.get("branch"), branch)]

    # 3. OWNERSHIP — hard filter (ignore 'Autonomous' as ownership)
    if ownership and not is_any(ownership) and ownership != "Autonomous":
        candidates = [r for r in candidates if college_type_matches(r.get("name"), ownership)]

    # 4. AUTONOMOUS — hard filter
    if autonomous is True or ownership == "Autonomous":
        candidates = [r for r in candidates if "autonomous" in r.get("name", "").lower()]
    elif autonomous is False:
        candidates = [r for r in candidates if "autonomous" not in r.get("name", "").lower()]

    # 5. COMMUNITY — exact match; default to OC if category not set
    target_cat = category.upper() if (category and not is_any(category)) else "OC"
    candidates = [r for r in candidates if r.get("community", "") == target_cat]

    # 6. CUTOFF — strict or with-reach
    if student_cutoff is not None:
        upper = student_cutoff + (REACH_BAND if with_reach else 0)
        candidates = [
            r for r in candidates
            if r.get("cutoff") is not None and r.get("cutoff") <= upper
        ]

    # 7. RANK — descending by college cutoff (closest to student cutoff first)
    candidates.sort(key=lambda r: (r.get("cutoff") is None, -(r.get("cutoff") or 0)))

    # 8. DEDUPLICATE by college code
    seen_codes: set = set()
    deduped: List[Dict] = []
    for rec in candidates:
        code = rec.get("code")
        if code not in seen_codes:
            seen_codes.add(code)
            deduped.append(rec)

    # 9. HARD VALIDATION re-assertion — catches any edge-case filter leakage
    if district and not is_any(district):
        deduped = [r for r in deduped if district_matches(r.get("district"), district)]

    return deduped if limit >= 9999 else deduped[:limit]


# ================================================================
# COLLEGE ALIASES & ACRONYM RESOLUTION
# ================================================================

# Code -> Full College Name map
COLLEGE_CODE_NAME_MAP: Dict[str, str] = {}
for _r in ALL_RECORDS:
    _c = _r.get("code")
    if _c and _c not in COLLEGE_CODE_NAME_MAP:
        COLLEGE_CODE_NAME_MAP[_c] = _r.get("name", "")

# Explicit popular aliases & acronyms
EXPLICIT_COLLEGE_ALIASES: Dict[str, List[str]] = {
    "rit": ["1432"],
    "ramco": ["4678"],
    "ramco rit": ["4678"],
    "srit": ["2725"],
    "svce": ["1219"],
    "ssn": ["1315"],
    "rec": ["1211"],
    "cit": ["1399", "2007"],
    "psg": ["2006", "2377"],
    "psg tech": ["2006"],
    "psg itech": ["2377"],
    "ceg": ["1"],
    "mit": ["4"],
    "act": ["2"],
    "actech": ["2"],
    "kct": ["2712"],
    "skcet": ["2718"],
    "skct": ["2722"],
    "tce": ["5008"],
    "gct": ["2005"],
    "bit": ["2702"],
    "kec": ["2711"],
    "licet": ["1450"],
    "rmk": ["1113"],
    "rmd": ["1112"],
    "srm": ["1422", "3795", "5842"],
    "srm valliammai": ["1422"],
    "valliammai": ["1422"],
    "msec": ["1309"],
    "saveetha": ["1216"],
    "sec": ["1216"],
    "sjce": ["1317"],
    "sjit": ["1149"],
    "st joseph": ["1317", "1149"],
    "st. joseph": ["1317", "1149"],
    "st. joseph's": ["1317", "1149"],
    "loyola": ["1450", "1225", "4993"],
}

_COLLEGE_STOP_WORDS = {
    "of", "and", "the", "in", "for", "at", "campus", "autonomous",
    "post", "district", "taluk", "village", "road", "college",
    "engineering", "technology", "institute", "chennai", "coimbatore"
}


def _build_acronym_map() -> Dict[str, List[str]]:
    """Build dynamic acronym and alias lookup from ALL_RECORDS."""
    acr_map: Dict[str, List[str]] = {}
    for alias, codes in EXPLICIT_COLLEGE_ALIASES.items():
        for code in codes:
            if code in COLLEGE_CODE_NAME_MAP:
                acr_map.setdefault(alias.lower(), []).append(code)

    for code, full_name in COLLEGE_CODE_NAME_MAP.items():
        short = _short_name(full_name)
        cleaned = re.sub(r"[\(\)\,\.\-\#\d]", " ", short)
        words = [w for w in cleaned.split() if w.lower() not in {"of", "and", "the", "in", "for", "at", "autonomous"} and len(w) > 0]
        if len(words) >= 2:
            acr1 = "".join(w[0] for w in words).lower()
            if len(acr1) >= 2:
                if code not in acr_map.setdefault(acr1, []):
                    acr_map[acr1].append(code)
            if words[0].lower() in {"sri", "st", "sree", "dr", "er", "prof"}:
                acr2 = "".join(w[0] for w in words[1:]).lower()
                if len(acr2) >= 2:
                    if code not in acr_map.setdefault(acr2, []):
                        acr_map[acr2].append(code)
    return acr_map


DYNAMIC_ACRONYM_MAP = _build_acronym_map()


def resolve_colleges(target_str: Optional[str]) -> Tuple[List[str], str]:
    """
    Resolve a target query string to a list of matching college codes.
    Returns (list_of_codes, match_type).
    """
    if not target_str:
        return [], "none"
    q = target_str.strip().lower()
    if not q:
        return [], "none"

    # 1. Exact numeric code
    m_code = re.search(r"\b(\d{1,6})\b", q)
    if m_code and m_code.group(1) in COLLEGE_CODE_NAME_MAP:
        return [m_code.group(1)], "code"

    # 2. Acronym / explicit alias
    q_clean = re.sub(r"^(?:college\s+|the\s+)", "", q).strip()
    q_clean = re.sub(r"\s+college$", "", q_clean).strip()
    if q_clean in DYNAMIC_ACRONYM_MAP:
        return DYNAMIC_ACRONYM_MAP[q_clean], "alias"
    if q in DYNAMIC_ACRONYM_MAP:
        return DYNAMIC_ACRONYM_MAP[q], "alias"

    # 3. Direct substring match
    sub_matches = []
    for code, name in COLLEGE_CODE_NAME_MAP.items():
        if q_clean in name.lower() or q in name.lower():
            sub_matches.append(code)
    if sub_matches:
        return sub_matches[:3], "substring"

    # 4. Keyword token match
    toks = [w for w in q_clean.split() if len(w) > 2 and w not in _COLLEGE_STOP_WORDS]
    if toks:
        kw_matches = []
        for code, name in COLLEGE_CODE_NAME_MAP.items():
            name_l = name.lower()
            if all(t in name_l for t in toks):
                kw_matches.append(code)
        if kw_matches:
            return kw_matches[:3], "keywords"

    # 5. Fuzzy close matches
    q_no_generic = re.sub(r"\b(?:college|engineering|technology|institute|autonomous|campus)\b", "", q_clean, flags=re.IGNORECASE).strip()
    if q_no_generic and len(q_no_generic) >= 3:
        clean_names = {}
        for short_n, c in COLLEGE_CODE_NAME_MAP.items():
            sn_clean = re.sub(r"\b(?:college|engineering|technology|institute|autonomous|campus)\b", "", _short_name(short_n), flags=re.IGNORECASE).strip()
            if sn_clean:
                clean_names[sn_clean.lower()] = c
        close = difflib.get_close_matches(q_no_generic, clean_names.keys(), n=3, cutoff=0.6)
        if close:
            return [clean_names[k] for k in close], "fuzzy"

    return [], "none"


def find_by_code(code: str) -> List[Dict]:
    """Return ALL records for a given college code (all branches, communities)."""
    return [r for r in ALL_RECORDS if r.get("code") == code]


def find_by_name_fuzzy(name_query: str) -> Optional[str]:
    """
    Find a college code by fuzzy name match, acronym, or code.
    Returns the first matching college code, or None.
    """
    q = name_query.strip().lower()
    if not q:
        return None

    # First try resolve_colleges
    codes, mtype = resolve_colleges(q)
    if codes and mtype != "fuzzy":
        return codes[0]

    # Direct substring in ALL_RECORDS
    for rec in ALL_RECORDS:
        full = rec.get("name", "").lower()
        short = _short_name(rec.get("name", "")).lower()
        if q in full or q in short:
            return rec.get("code")

    # All words match
    words = [w for w in q.split() if len(w) > 2]
    if words:
        for rec in ALL_RECORDS:
            name = rec.get("name", "").lower()
            if all(w in name for w in words):
                return rec.get("code")

    if codes:
        return codes[0]

    return None


def extract_target_college_query(text: str) -> Optional[str]:
    """
    Extract the candidate college name/code from an eligibility question.
    e.g. 'Can I get RIT with 120 BC?' -> 'RIT'
         'can get rit college for 120 cutoff' -> 'rit college'
         'Can I get CSE in SVCE with 175 OC?' -> 'SVCE'
    """
    t = text.strip()
    patterns = [
        r"^(?:can\s+(?:i\s+|we\s+)?(?:get\s+admission\s+in|get\s+into|get\s+in|get|join|enter|admit\s+to|have))\s+(.+)$",
        r"^(?:will\s+(?:i\s+|we\s+)?(?:get\s+admission\s+in|get\s+into|get\s+in|get|join|enter))\s+(.+)$",
        r"^(?:is\s+(?:getting\s+into\s+|admission\s+in\s+)?)\s*(.+)$",
        r"^(?:what\s+are\s+(?:my|our)\s+chances\s+(?:of|for)\s+(?:getting\s+into\s+|getting\s+in\s+|getting\s+)?)\s*(.+)$",
        r"^(?:chances?\s+(?:of|for)\s+(?:getting\s+into\s+|getting\s+)?)\s*(.+)$",
        r"^(?:am\s+i\s+eligible\s+for|are\s+we\s+eligible\s+for|eligibility\s+(?:for|in))\s+(.+)$",
        r"^(?:is\s+it\s+possible\s+to\s+get(?:\s+into)?)\s+(.+)$",
        r"^(?:chance\s+of\s+getting)\s+(.+)$",
    ]

    target_part = None
    for p in patterns:
        m = re.search(p, t, re.IGNORECASE)
        if m:
            target_part = m.group(1).strip()
            break

    if not target_part:
        m_poss = re.search(r"^(.*?)\s+is\s+possible(?:\s+for\s+me|\s+with|\s+for|\s*\?|$)", t, re.IGNORECASE)
        if m_poss:
            target_part = m_poss.group(1).strip()

    if not target_part:
        m_gen = re.search(r"(?:can\s+(?:i\s+|we\s+)?get|will\s+(?:i\s+|we\s+)?get|is)\s+(.+?)(?:\s+(?:with|for|at)\s+\d+|\s*\?|$)", t, re.IGNORECASE)
        if m_gen:
            target_part = m_gen.group(1).strip()

    if not target_part:
        return None

    cleaned = target_part
    cleaned = re.sub(r"\s+possible(?:\s+for\s+me)?\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:with|for|at)\s+(?:my\s+cutoff|\d+(?:\.\d+)?\s*(?:bc|oc|bcm|mbc|sc|sca|st|cutoff|marks?)|my\s+cutoff|\d+(?:\.\d+)?).*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:in|for)\s+(?:chennai|coimbatore|salem|madurai|trichy|vellore).*$", "", cleaned, flags=re.IGNORECASE)

    # Handle branch prefixes like "CSE in RIT" or "ECE at SVCE"
    m_branch_in = re.search(r"^(?:cse|ece|eee|mech|civil|it|ai\s*&\s*ds|ai\s*&\s*ml)\s+(?:in|at|into)\s+(.+)$", cleaned.strip(), re.IGNORECASE)
    if m_branch_in:
        cleaned = m_branch_in.group(1)

    cleaned = re.sub(r"[\?\.\!]", "", cleaned).strip()
    cleaned = re.sub(r"^(?:admission\s+(?:in|into|to)\s+|getting\s+into\s+|it\s+to\s+get\s+into\s+)", "", cleaned, flags=re.IGNORECASE).strip()

    return cleaned if cleaned else None


def is_college_eligibility_query(text: str) -> bool:
    """Detect if message is inquiring about getting admission into a specific college."""
    t = text.lower().strip()
    if t in {"reset", "restart", "clear filters", "more", "help", "hi", "hello"}:
        return False
    if t.startswith("what is") or t.startswith("how is") or t.startswith("compare ") or " vs " in t:
        return False
    if re.search(r"\btop\s+\d+\b|\btop\s+colleges\b|\bbest\s+colleges\b", t):
        if not re.search(r"\b(?:can\s+(?:i\s+|we\s+)?|will\s+(?:i\s+|we\s+)?|is\s+it\s+possible)\b", t):
            return False

    patterns = [
        r"\b(?:can\s+(?:i\s+|we\s+)?(?:get|join|enter|have|admit))\b",
        r"\b(?:will\s+(?:i\s+|we\s+)?(?:get|join|enter|admit))\b",
        r"\b(?:is\s+.+?\s+possible(?:\s+for\s+me|\s+with|\s+for|\s*\?|$))\b",
        r"\b(?:possible\s+to\s+get(?:\s+into)?)\b",
        r"\b(?:chances?\s+(?:of|for)\s+(?:getting\s+)?)\b",
        r"\b(?:am\s+i\s+eligible\s+for|are\s+we\s+eligible\s+for|eligibility\s+(?:for|in))\b",
        r"\b(?:chance\s+of\s+getting)\b",
        r"\b(?:can\s+(?:i\s+|we\s+)?get\s+into|get\s+admission\s+in)\b",
    ]
    return any(re.search(p, t) for p in patterns)


def check_college_eligibility(user_message: str, state: Dict) -> Tuple[str, Optional[str]]:
    """
    Evaluate student's admission eligibility for a specific college.
    Returns (response_markdown, matched_college_code).
    """
    student_cutoff = state.get("cutoff")
    student_category = state.get("category") or "OC"
    student_branch = state.get("branch")

    target_str = extract_target_college_query(user_message)
    if not target_str:
        code_m = extract_college_code(user_message)
        if code_m:
            target_str = code_m
        else:
            return "Please specify which college you'd like to check eligibility for.", None

    codes, match_type = resolve_colleges(target_str)

    if not codes or match_type == "none":
        all_names = list({_short_name(n) for n in COLLEGE_CODE_NAME_MAP.values()})
        close = difflib.get_close_matches(target_str, all_names, n=3, cutoff=0.3)
        if close:
            sugg_text = "\n".join([f"- **{c}**" for c in close])
            return f"❌ I couldn't find **{target_str}** in the current TNEA dataset.\n\nDid you mean:\n{sugg_text}", None
        return f"❌ I couldn't find **{target_str}** in the current TNEA dataset. Please check the college name or use a college code (e.g. `1413`).", None

    primary_code = codes[0]

    # If student cutoff is not yet set:
    if student_cutoff is None:
        c_name = _short_name(COLLEGE_CODE_NAME_MAP.get(primary_code, f"College {primary_code}"))
        records = [r for r in ALL_RECORDS if r.get("code") == primary_code and r.get("community") == student_category]
        if not records:
            records = [r for r in ALL_RECORDS if r.get("code") == primary_code and r.get("community") == "OC"]
            active_cat = "OC"
        else:
            active_cat = student_category

        lines = [
            f"### 🎓 {c_name} (Code: `{primary_code}`) — Eligibility Check",
            "",
            f"To calculate your exact eligibility and chance for **{c_name}**, please tell me your **TNEA Cutoff Mark**.",
            "",
            f"*Closing cutoffs for {active_cat} category (TNEA 2025):*",
            "",
            "| Branch | Closing Cutoff |",
            "|:---|---:|",
        ]
        for r in sorted(records, key=lambda x: -(x.get("cutoff") or 0)):
            co = r.get("cutoff")
            lines.append(f"| {r.get('branch', 'Unknown')} | {f'{co:g}' if co is not None else 'N/A'} |")
        lines += [
            "",
            f"👉 *Reply with your cutoff (e.g. `160`) or `160 {active_cat}` to check your chances!*",
        ]
        return "\n".join(lines), primary_code

    # If student cutoff is available, evaluate the single targeted college
    primary_code = codes[0]
    records = [r for r in ALL_RECORDS if r.get("code") == primary_code and r.get("community") == student_category]
    if not records:
        records = [r for r in ALL_RECORDS if r.get("code") == primary_code and r.get("community") == "OC"]
        active_cat = "OC"
    else:
        active_cat = student_category

    if student_branch and student_branch != "Any":
        records = [r for r in records if branch_matches(r.get("branch"), student_branch)]

    c_name = _short_name(COLLEGE_CODE_NAME_MAP.get(primary_code, f"College {primary_code}"))
    if not records:
        return f"No branch records found for **{c_name}** matching branch `{student_branch}`.", primary_code

    cat_disp = COMMUNITY_DISPLAY.get(active_cat, active_cat)
    branch_disp = student_branch or "Any"

    branch_rows = []
    for r in records:
        co = r.get("cutoff")
        if co is None:
            continue
        gap = student_cutoff - co
        if gap >= 10:
            chance = "Safe"
            tier_order = 1
        elif gap >= 4:
            chance = "Good Chance"
            tier_order = 2
        elif gap >= 0:
            chance = "Moderate"
            tier_order = 3
        elif gap >= -5:
            chance = "Reach"
            tier_order = 4
        else:
            chance = "Unlikely"
            tier_order = 5

        gap_str = f"+{gap:.1f}" if gap > 0 else (f"{gap:.1f}" if gap < 0 else "0.0")
        branch_rows.append({
            "branch": r.get("branch", "Unknown"),
            "cutoff": co,
            "gap": gap,
            "gap_str": gap_str,
            "chance": chance,
            "tier_order": tier_order,
        })

    branch_rows.sort(key=lambda x: (x["tier_order"], -x["gap"]))

    lines = [
        f"### 🎓 {c_name} (Code: `{primary_code}`)",
        "",
        f"**Your Cutoff:** {student_cutoff:g} | **Category:** {cat_disp} | **Branch:** {branch_disp}",
        "",
        f"| Branch | {active_cat} Closing Cutoff | Gap | Chance |",
        "|:---|---:|---:|:---|",
    ]
    for row in branch_rows:
        lines.append(f"| {row['branch']} | {row['cutoff']:g} | {row['gap_str']} | {row['chance']} |")

    lines.append("")

    if branch_rows:
        best_chance = branch_rows[0]["chance"]
        realistic = [r["branch"] for r in branch_rows if r["chance"] in {"Safe", "Good Chance", "Moderate", "Reach"}]
        min_co_row = min(branch_rows, key=lambda x: x["cutoff"])
        min_co = min_co_row["cutoff"]
        min_gap = min_co - student_cutoff

        if best_chance in {"Safe", "Good Chance"}:
            lines.append(f"**Verdict: ✅ {best_chance}**")
            lines.append(f"\nBased on TNEA 2025 cutoff data, you have a **strong chance** of admission to **{c_name}**.")
            if realistic:
                lines.append(f"- **Eligible branches ({len(realistic)}):** {', '.join(realistic[:4])}")
        elif best_chance == "Moderate":
            lines.append(f"**Verdict: 🟡 Moderate**")
            lines.append(f"\nYou have a **moderate chance** of getting a seat at **{c_name}**.")
            if realistic:
                lines.append(f"- **Possible branches:** {', '.join(realistic[:4])}")
        elif best_chance == "Reach":
            lines.append(f"**Verdict: 🎯 Reach / Ambitious**")
            lines.append(f"\n**{c_name}** is a **reach option** for your cutoff.")
            if realistic:
                lines.append(f"- **Closest branches:** {', '.join(realistic[:3])}")
        else:
            lines.append(f"**Verdict: ❌ Unlikely**")
            lines.append(f"\nWith a **{student_cutoff:g} cutoff in {cat_disp}**, getting a seat at **{c_name}** is **unlikely based on TNEA 2025 cutoff data**.")
            lines.append(f"\nThe lowest closing cutoff among the available {c_name} branches was **{min_co:g}** ({min_co_row['branch']}), which is about **{min_gap:.1f} marks higher** than your cutoff.")
            lines.append(f"\nSo, based on the previous year's cutoff, **{c_name} is not a realistic option at {student_cutoff:g} cutoff**.")

    lines += [
        "",
        "> ⚠️ *Historical TNEA 2025 cutoff-based estimate. Cutoffs can change each year depending on competition, seat availability, and counselling rounds.*",
    ]

    return "\n".join(lines), primary_code


def get_general_colleges(
    district: Optional[str] = None,
    branch: Optional[str] = None,
    college_type: Optional[str] = None,
    ownership: Optional[str] = None,
    autonomous: Optional[bool] = None,
    limit: int = 10,
) -> List[Dict]:
    """
    Top colleges for general (non-personalised) queries.
    Uses OC community as ranking proxy (highest OC closing cutoff first).
    """
    candidates = ALL_RECORDS.copy()

    if district and not is_any(district):
        candidates = [r for r in candidates if district_matches(r.get("district"), district)]
    if branch and not is_any(branch):
        candidates = [r for r in candidates if branch_matches(r.get("branch"), branch)]

    target_ownership = ownership or college_type
    if target_ownership and not is_any(target_ownership) and target_ownership != "Autonomous":
        candidates = [r for r in candidates if college_type_matches(r.get("name"), target_ownership)]

    if autonomous is True or target_ownership == "Autonomous":
        candidates = [r for r in candidates if "autonomous" in r.get("name", "").lower()]
    elif autonomous is False:
        candidates = [r for r in candidates if "autonomous" not in r.get("name", "").lower()]

    candidates = [r for r in candidates if r.get("community") == "OC"]
    candidates.sort(key=lambda r: -(r.get("cutoff") or 0))

    seen_codes: set = set()
    deduped: List[Dict] = []
    for rec in candidates:
        code = rec.get("code")
        if code not in seen_codes:
            seen_codes.add(code)
            deduped.append(rec)

    if district and not is_any(district):
        deduped = [r for r in deduped if district_matches(r.get("district"), district)]

    return deduped if limit >= 9999 else deduped[:limit]


# ================================================================
# CHANCE LABEL
# ================================================================

def chance_label(student_cutoff: Optional[float], rec_cutoff: Optional[float]) -> str:
    """
    Classify admission chance based on mark difference.
    Reach = college cutoff ABOVE student cutoff (only appears in with_reach mode).
    """
    if student_cutoff is None or rec_cutoff is None:
        return "—"
    diff = student_cutoff - rec_cutoff   # positive → student above college
    if diff < 0:    return "🎯 Reach"
    if diff < 4:    return "🟡 Moderate"
    if diff < 10:   return "🟢 Good"
    return "🔵 Safe"


# ================================================================
# FORMAT: TIERED RECOMMENDATION  (the main personalized response)
# ================================================================

def build_tiered_response(state: Dict) -> Tuple[str, List[Dict]]:
    """
    Build the tiered recommendation view and return (markdown_text, top_records).

    Tier separation is STRICT — Reach colleges are NEVER mixed with eligible ones.

    🎯 Reach     — college cutoff is ABOVE student cutoff (up to +REACH_BAND)
    🟢 Good Chance — eligible, diff < 10 marks
    🔵 Safe Options — eligible, diff >= 10 marks

    Returns (response_str, top_eligible_records) so process_message can store
    last_recommendations for the "why this college?" feature.
    """
    sc  = state.get("cutoff")
    cat = state.get("category")

    if sc is None or cat is None:
        return _no_results_msg(state), []

    # Fetch all with reach, already ranked descending by college cutoff
    all_recs = filter_and_rank(state, limit=9999, with_reach=True)

    # ── STRICT split ─────────────────────────────────────────────
    reach_recs    = [r for r in all_recs if (r.get("cutoff") or 0) > sc]
    eligible_recs = [r for r in all_recs if (r.get("cutoff") or 0) <= sc]

    # Within eligible: Good vs Safe
    good_recs = [r for r in eligible_recs if sc - (r.get("cutoff") or 0) < 10]
    safe_recs = [r for r in eligible_recs if sc - (r.get("cutoff") or 0) >= 10]

    # Per-tier display limits
    reach_limit = 3
    good_limit  = min(7, max(3, state.get("limit", 10)))
    safe_limit  = min(12, max(3, state.get("limit", 10)))

    cat_disp   = COMMUNITY_DISPLAY.get(cat, cat)
    dist_d     = state.get("district") or "Any"
    branch_d   = state.get("branch") or "Any"
    type_d     = state.get("college_type") or "Any"

    lines = [
        "## 🎯 College Recommendations",
        "",
        f"**Cutoff:** {sc:g} | **Category:** {cat_disp} | "
        f"**District:** {dist_d} | **Branch:** {branch_d} | **Type:** {type_d}",
        "",
    ]

    # ── REACH tier ───────────────────────────────────────────────
    if reach_recs:
        lines += [
            "### 🎯 Reach",
            "*Closing cutoff is slightly above your score. Possible but not guaranteed — "
            "competition and seat availability determine actual chances.*",
            "",
            "| # | College | District | Branch | Cutoff | Gap |",
            "|--:|:--------|:---------|:-------|-------:|:----|",
        ]
        for i, rec in enumerate(reach_recs[:reach_limit], 1):
            co  = rec.get("cutoff")
            gap = f"+{co - sc:.1f}" if co is not None else "N/A"
            co_s = f"{co:g}" if co is not None else "N/A"
            lines.append(
                f"| {i} | {_short_name(rec.get('name', '?'))} | "
                f"{rec.get('district', '?')} | {rec.get('branch', '?')} | "
                f"{co_s} | {gap} |"
            )
        lines.append("")

    # ── GOOD CHANCE tier ─────────────────────────────────────────
    if good_recs:
        lines += [
            "### 🟢 Good Chance",
            "*Closing cutoff is at or just below your score.*",
            "",
            "| # | College | District | Branch | Cutoff | Chance |",
            "|--:|:--------|:---------|:-------|-------:|:-------|",
        ]
        for i, rec in enumerate(good_recs[:good_limit], 1):
            co = rec.get("cutoff")
            lines.append(
                f"| {i} | {_short_name(rec.get('name', '?'))} | "
                f"{rec.get('district', '?')} | {rec.get('branch', '?')} | "
                f"{f'{co:g}' if co is not None else 'N/A'} | {chance_label(sc, co)} |"
            )
        lines.append("")

    # ── SAFE OPTIONS tier ────────────────────────────────────────
    if safe_recs:
        lines += [
            "### 🔵 Safe Options",
            "*Closing cutoff is significantly below your score.*",
            "",
            "| # | College | District | Branch | Cutoff | Chance |",
            "|--:|:--------|:---------|:-------|-------:|:-------|",
        ]
        for i, rec in enumerate(safe_recs[:safe_limit], 1):
            co = rec.get("cutoff")
            lines.append(
                f"| {i} | {_short_name(rec.get('name', '?'))} | "
                f"{rec.get('district', '?')} | {rec.get('branch', '?')} | "
                f"{f'{co:g}' if co is not None else 'N/A'} | {chance_label(sc, co)} |"
            )
        lines.append("")

    if not reach_recs and not good_recs and not safe_recs:
        return _no_results_msg(state), []

    lines += [
        "> ⚠️ *Based on historical TNEA 2025 closing cutoffs. "
        "Actual admission depends on counselling round, seat availability, "
        "and competition. This is **not** an admission guarantee.*",
        "",
        "Type **`more`** for additional colleges  |  "
        "Type **`why this college?`** to understand the top result.",
    ]

    top_eligible = eligible_recs[:state.get("limit", 10)]
    return "\n".join(lines), top_eligible


# ================================================================
# FORMAT: FLAT TABLE (used for "more" pagination)
# ================================================================

def generate_recommendation_table(
    records: List[Dict],
    state: Dict,
    start_offset: int = 0,
) -> str:
    """Flat Markdown table — used for pagination after the initial tiered view."""
    if not records:
        return _no_results_msg(state)

    sc       = state.get("cutoff")
    cat      = state.get("category")
    dist_d   = state.get("district") or "Any"
    branch_d = state.get("branch") or "Any"
    cat_str  = COMMUNITY_DISPLAY.get(cat, cat) if cat else "OC (General)"

    lines = [
        f"**Cutoff:** {f'{sc:g}' if sc else 'N/A'} | **Category:** {cat_str} | "
        f"**District:** {dist_d} | **Branch:** {branch_d}",
        "",
        "| # | College | District | Branch | Cutoff | Chance |",
        "|--:|:--------|:---------|:-------|-------:|:-------|",
    ]
    for i, rec in enumerate(records, start_offset + 1):
        co = rec.get("cutoff")
        lines.append(
            f"| {i} | {_short_name(rec.get('name', '?'))} | "
            f"{rec.get('district', '?')} | {rec.get('branch', '?')} | "
            f"{f'{co:g}' if co is not None else 'N/A'} | {chance_label(sc, co)} |"
        )
    lines += [
        "",
        "> ⚠️ *Historical TNEA 2025 data. Not an admission guarantee.*",
        "",
        "Type **`more`** for the next batch.",
    ]
    return "\n".join(lines)


# ================================================================
# FORMAT: GENERAL (no cutoff) RECOMMENDATION TABLE
# ================================================================

def build_general_recommendation_response(
    records: List[Dict],
    state: Dict,
    start_offset: int = 0,
) -> str:
    dist   = state.get("district")
    norm_d = normalize_district(dist) if dist else None

    if not records:
        if norm_d:
            return (
                f"No colleges found in the TNEA data for **{norm_d}**.\n\n"
                "**Try:**\n"
                "- Another district — e.g. `Chennai`, `Coimbatore`, `Salem`\n"
                "- All districts — type `any district`\n"
                "- Provide your cutoff for personalised results"
            )
        return _no_results_msg(state)

    dist_display   = norm_d or "Tamil Nadu"
    branch_display = f" · Branch: {state['branch']}" if state.get("branch") else ""
    type_display   = f" · Type: {state['college_type']}" if state.get("college_type") else ""

    lines = [
        f"## 🏛️ Top Colleges — {dist_display}{branch_display}{type_display}",
        "",
        "*Ranked by OC closing cutoff mark (TNEA 2025).*",
        "",
        "| # | College | District | Branch | OC Cutoff |",
        "|--:|:--------|:---------|:-------|----------:|",
    ]
    for i, rec in enumerate(records, start_offset + 1):
        co = rec.get("cutoff")
        lines.append(
            f"| {i} | {_short_name(rec.get('name', '?'))} | "
            f"{rec.get('district', '?')} | {rec.get('branch', '?')} | "
            f"{f'{co:g}' if co is not None else 'N/A'} |"
        )
    lines += [
        "",
        "For personalised chances, tell me your cutoff and category.",
        "Example: `160 BC CSE Chennai`",
        "",
        "Type **`more`** for the next batch.",
    ]
    return "\n".join(lines)


def _no_results_msg(state: Dict) -> str:
    lines = ["## 🔍 No Exact Matches Found", "", "**Active filters:**"]
    if state.get("cutoff"):
        lines.append(f"- **Cutoff:** {state['cutoff']:g}")
    if state.get("category"):
        lines.append(f"- **Category:** {state['category']}")
    if not is_any(state.get("district")):
        lines.append(f"- **District:** {state['district']}")
    if not is_any(state.get("branch")):
        lines.append(f"- **Branch:** {state['branch']}")
    if not is_any(state.get("ownership")):
        lines.append(f"- **Ownership:** {state['ownership']}")
    elif not is_any(state.get("college_type")):
        lines.append(f"- **College Type:** {state['college_type']}")
    if state.get("autonomous") is not None:
        lines.append(f"- **Autonomous:** {'Yes' if state['autonomous'] else 'No'}")

    lines.append("")

    dist = state.get("district") if not is_any(state.get("district")) else None
    branch = state.get("branch") if not is_any(state.get("branch")) else None
    ow = state.get("ownership") if not is_any(state.get("ownership")) else None

    # Check if branch exists in district at all
    has_branch_in_dist = True
    if dist and branch:
        has_branch_in_dist = any(
            r.get("district") == dist and branch_matches(r.get("branch", ""), branch)
            for r in ALL_RECORDS
        )
        if not has_branch_in_dist:
            lines.append(f"> ⚠️ **Note:** **{branch}** is not offered by any engineering college in **{dist}** district.\n")

    # Alternative 1: Same branch across all Tamil Nadu
    alt_recs_tn = []
    if branch:
        alt_state_tn = dict(state)
        alt_state_tn["district"] = None
        alt_state_tn["ownership"] = None
        alt_state_tn["limit"] = 5
        alt_recs_tn = filter_and_rank(alt_state_tn)
        if alt_recs_tn:
            lines.append(f"### 💡 Colleges offering **{branch}** across Tamil Nadu:")
            lines.append("| College | District | Branch | Cutoff | Chance |")
            lines.append("|:--------|:---------|:-------|-------:|:-------|")
            for r in alt_recs_tn[:4]:
                co = r.get("cutoff")
                co_str = f"{co:g}" if co is not None else "N/A"
                lines.append(f"| {_short_name(r.get('name', ''))} | {r.get('district', '')} | {r.get('branch', '')} | {co_str} | {r.get('tier', 'Reach')} |")
            lines.append("")

    # Alternative 2: Available branches in this district for user's cutoff
    if dist and (not branch or not alt_recs_tn or not has_branch_in_dist):
        alt_state_dist = dict(state)
        alt_state_dist["branch"] = None
        alt_state_dist["ownership"] = None
        alt_state_dist["limit"] = 5
        alt_recs_dist = filter_and_rank(alt_state_dist)
        if alt_recs_dist:
            lines.append(f"### 💡 Available branches in **{dist}** for your cutoff:")
            lines.append("| College | Branch | Cutoff | Chance |")
            lines.append("|:--------|:-------|-------:|:-------|")
            for r in alt_recs_dist[:4]:
                co = r.get("cutoff")
                co_str = f"{co:g}" if co is not None else "N/A"
                lines.append(f"| {_short_name(r.get('name', ''))} | {r.get('branch', '')} | {co_str} | {r.get('tier', 'Reach')} |")
            lines.append("")

    lines += [
        "**Quick ways to explore options:**",
        f"- Reply with `any district` to see colleges offering **{branch}** across Tamil Nadu" if dist and branch else "",
        f"- Reply with `any branch` to see all available engineering branches in **{dist}**" if branch and dist else "",
        "- Reply with `any type` to include Self-Financing colleges" if ow else "",
        "- Reply with `clear filters` to start fresh with a new search",
    ]
    lines = [l for l in lines if l != ""]
    return "\n".join(lines)


# ================================================================
# FORMAT: COLLEGE DETAIL CARD
# ================================================================

def format_college_detail(records: List[Dict]) -> str:
    """Rich Markdown card for a single college (all branches + communities)."""
    if not records:
        return "No records found for this college code."

    name     = records[0].get("name", "Unknown")
    code     = records[0].get("code", "?")
    district = records[0].get("district", "?")
    year     = records[0].get("year", "2025")
    ctype    = _type_from_name(name)

    lines = [
        "## 🏫 College Details",
        "",
        f"**Code:** `{code}`",
        f"**Name:** {_short_name(name)}",
        f"**District:** {district}",
        f"**Type:** {ctype}",
        f"**Data Year:** {year}",
        "",
        "---",
        "",
        "### 📚 Branch-wise Closing Cutoff Marks",
        "",
    ]

    branch_map: Dict[str, List] = {}
    for r in records:
        branch_map.setdefault(r.get("branch", "Unknown"), []).append(r)

    for branch in sorted(branch_map.keys()):
        rows = branch_map[branch]
        lines.append(f"**{branch}**")
        lines.append("| Community | Closing Cutoff |")
        lines.append("|:----------|---------------:|")
        for row in sorted(rows, key=lambda x: x.get("community", "")):
            co      = row.get("cutoff")
            comm    = row.get("community", "?")
            co_str  = f"{co:g}" if co is not None else "N/A"
            comm_d  = COMMUNITY_DISPLAY.get(comm, comm)
            lines.append(f"| {comm_d} | {co_str} |")
        lines.append("")

    lines.append("> *All cutoffs are TNEA 2025 closing marks.*")
    return "\n".join(lines)


# ================================================================
# COLLEGE COMPARISON
# ================================================================

def _parse_comparison_targets(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract two identifiers from a compare query."""
    # Try explicit "X and Y" pattern first
    m = re.search(
        r"(?:compare|vs|versus|between)\s+(.+?)\s+(?:and|vs|versus)\s+(.+)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Fallback: split on "and"
    parts = re.split(r"\s+and\s+", text, flags=re.IGNORECASE)
    if len(parts) >= 2:
        # Strip leading trigger words from first part
        p1 = re.sub(r"(?i)^\s*(?:compare|comparison|between)\s+", "", parts[0]).strip()
        p2 = parts[1].strip()
        return (p1 or None), (p2 or None)
    return None, None


def _resolve_college(identifier: str) -> Optional[str]:
    """Resolve an identifier to a college code. Code-first, then fuzzy name."""
    identifier = identifier.strip()
    if not identifier:
        return None
    # Exact code
    if re.fullmatch(r"\d{4,6}", identifier):
        if find_by_code(identifier):
            return identifier
    # Fuzzy name
    return find_by_name_fuzzy(identifier)


def compare_colleges(text: str, branch_filter: Optional[str] = None) -> str:
    """
    Compare two colleges side-by-side.
    Supports: 'compare 1413 and 2347'  or  'compare SVCE and SRM Valliammai'.
    Prefers exact code matching; falls back to fuzzy name matching.
    """
    id1, id2 = _parse_comparison_targets(text)

    if not id1 or not id2:
        return (
            "To compare colleges, use:\n\n"
            "`compare SVCE and SRM Valliammai`\n\n"
            "or\n\n"
            "`compare 1413 and 2347`"
        )

    code1 = _resolve_college(id1)
    code2 = _resolve_college(id2)

    if not code1 and not code2:
        return (
            f"I couldn't find **`{id1}`** or **`{id2}`** in the TNEA data.\n\n"
            "Try using a college code — e.g. `compare 1413 and 2347`"
        )
    if not code1:
        return f"I couldn't find **`{id1}`** in the TNEA data. Please check the name or use a code."
    if not code2:
        return f"I couldn't find **`{id2}`** in the TNEA data. Please check the name or use a code."

    recs1 = find_by_code(code1)
    recs2 = find_by_code(code2)

    if not recs1 or not recs2:
        return "Could not load data for one or both colleges."

    name1  = _short_name(recs1[0].get("name", code1))
    name2  = _short_name(recs2[0].get("name", code2))
    dist1  = recs1[0].get("district", "?")
    dist2  = recs2[0].get("district", "?")
    type1  = _type_from_name(recs1[0].get("name", ""))
    type2  = _type_from_name(recs2[0].get("name", ""))

    branches1 = sorted({r.get("branch") for r in recs1})
    branches2 = sorted({r.get("branch") for r in recs2})
    all_branches = sorted(set(branches1) | set(branches2))

    # Narrow to relevant branch if one is active in state
    if branch_filter:
        kws = BRANCH_MAP.get(branch_filter, {}).get("data", [branch_filter.lower()])
        all_branches = [b for b in all_branches if any(kw in b.lower() for kw in kws)]

    def cutoff_map(recs):
        m: Dict[str, Dict[str, Optional[float]]] = {}
        for r in recs:
            m.setdefault(r.get("branch", ""), {})[r.get("community", "")] = r.get("cutoff")
        return m

    cm1 = cutoff_map(recs1)
    cm2 = cutoff_map(recs2)

    lines = [
        "## 🔍 College Comparison",
        "",
        f"| Feature | {name1} | {name2} |",
        "|:--------|:--------|:--------|",
        f"| **College Code** | `{code1}` | `{code2}` |",
        f"| **District** | {dist1} | {dist2} |",
        f"| **Type** | {type1} | {type2} |",
        f"| **Branches offered** | {len(branches1)} | {len(branches2)} |",
        "",
    ]

    if all_branches:
        lines += ["### Cutoff Comparison (TNEA 2025)", "", "| Branch (Community) | " + name1 + " | " + name2 + " |",
                  "|:-------------------|----------:|----------:|"]
        for branch in all_branches:
            for comm in ["OC", "BC", "MBC", "SC", "BCM", "SCA", "ST"]:
                c1 = cm1.get(branch, {}).get(comm)
                c2 = cm2.get(branch, {}).get(comm)
                if c1 is not None or c2 is not None:
                    c1s = f"{c1:g}" if c1 is not None else "—"
                    c2s = f"{c2:g}" if c2 is not None else "—"
                    comm_d = COMMUNITY_DISPLAY.get(comm, comm)
                    lines.append(f"| {branch} ({comm_d}) | {c1s} | {c2s} |")

    lines += [
        "",
        "> *Only data available in TNEA 2025 dataset is shown. "
        "Fees, placements, and rankings are not in this dataset.*",
    ]
    return "\n".join(lines)


# ================================================================
# WHY THIS COLLEGE — explanation
# ================================================================

def explain_recommendation(state: Dict) -> str:
    """Explain the top college from last_recommendations."""
    recs = state.get("last_recommendations", [])
    if not recs:
        return (
            "I don't have an active recommendation to explain. "
            "Please ask for college suggestions first, then ask me to explain."
        )

    sc   = state.get("cutoff")
    cat  = state.get("category")
    top  = recs[0]

    name     = _short_name(top.get("name", "this college"))
    code     = top.get("code", "?")
    district = top.get("district", "?")
    branch   = top.get("branch", "?")
    co       = top.get("cutoff")
    cat_disp = COMMUNITY_DISPLAY.get(cat, cat) if cat else "your category"

    lines = [f"## 💡 Why was **{name}** recommended?", ""]

    if sc is not None and co is not None:
        diff = sc - co
        lines += [
            f"| Detail | Value |",
            f"|:-------|:------|",
            f"| Your cutoff | {sc:g} |",
            f"| {cat_disp} closing cutoff (2025) | {co:g} |",
            f"| Your margin | {diff:+.1f} marks |",
            f"| Chance label | {chance_label(sc, co)} |",
            "",
        ]
        if diff < 0:
            lines.append(
                f"This is a **Reach** college. Its {cat_disp} closing cutoff ({co:g}) is "
                f"{abs(diff):.1f} marks **above** your score ({sc:g}). "
                f"It appears because it is within the {REACH_BAND}-mark reach band. "
                "Competition and seat availability determine actual chances."
            )
        elif diff < 4:
            lines.append(
                f"This is a **Moderate** match. The closing cutoff ({co:g}) is very close "
                f"to your score ({sc:g}). Historical data suggests a reasonable chance, "
                "but competition in this band is high."
            )
        elif diff < 10:
            lines.append(
                f"This is a **Good** match. Your score ({sc:g}) is {diff:.1f} marks above "
                f"the {cat_disp} closing cutoff ({co:g}). Strong historical chance based on 2025 data."
            )
        else:
            lines.append(
                f"This is a **Safe** option. Your score ({sc:g}) is {diff:.1f} marks above "
                f"the {cat_disp} closing cutoff ({co:g}). Very strong historical chance."
            )
    else:
        lines.append(f"College Code: {code} | District: {district} | Branch: {branch}")

    lines += [
        "",
        f"**Other details:**",
        f"- College Code: `{code}`",
        f"- District: {district}",
        f"- Branch: {branch}",
        f"- Type: {_type_from_name(top.get('name', ''))}",
        "",
        "> ⚠️ *Based on historical TNEA 2025 closing cutoffs. "
        "Actual admission depends on counselling round, seat availability, "
        "and competition. Not an admission guarantee.*",
    ]
    return "\n".join(lines)


# ================================================================
# TNEA KNOWLEDGE GUIDES & PCM CALCULATOR
# ================================================================

_CUTOFF_FORMULA_PATTERNS = [
    r"how\s+is\s+(?:the\s+)?(?:tnea\s+)?cutoff\s+calculated",
    r"how\s+to\s+calculate\s+(?:the\s+)?(?:tnea\s+)?cutoff",
    r"cutoff\s+formula",
    r"calculate\s+(?:my\s+)?(?:tnea\s+)?cutoff",
    r"what\s+is\s+(?:the\s+)?(?:tnea\s+)?cutoff\s+formula",
    r"how\s+to\s+calculate\s+cutoff",
    r"how\s+cutoff\s+is\s+calculated",
]

_COUNSELLING_PATTERNS = [
    r"explain\s+(?:the\s+)?(?:tnea\s+)?counselling\s+process",
    r"(?:what\s+is\s+)?(?:the\s+)?(?:tnea\s+)?counselling\s+process",
    r"counselling\s+(?:process|procedure|steps|stages)",
    r"how\s+(?:does\s+)?(?:tnea\s+)?counselling\s+work",
    r"tnea\s+procedure",
]


def is_cutoff_formula_query(text: str) -> bool:
    t = text.lower().strip()
    return any(re.search(p, t) for p in _CUTOFF_FORMULA_PATTERNS)


def is_counselling_process_query(text: str) -> bool:
    t = text.lower().strip()
    return any(re.search(p, t) for p in _COUNSELLING_PATTERNS)


def format_cutoff_calculation_guide() -> str:
    return """\
## 🎯 How to Calculate TNEA Cutoff?

### Formula:
$$\\text{Cutoff} = \\text{Maths} + \\left(\\frac{\\text{Physics}}{2}\\right) + \\left(\\frac{\\text{Chemistry}}{2}\\right)$$

### Please provide your marks:
**Maths, Physics, Chemistry**

*Enter your marks in this format:*
`Maths = ?, Physics = ?, Chemistry = ?`"""


def format_counselling_process_guide() -> str:
    return """\
## 📘 TNEA 2025 Counselling Process

The Tamil Nadu Engineering Admissions (TNEA) single-window counselling is conducted online by DOTE in sequential stages:

### 1. 📝 Online Registration & Application
- Register online at [tneaonline.org](https://www.tneaonline.org).
- Enter academic marks and upload certificates (10th/12th marksheets, community certificate, special reservation forms).

### 2. 🔍 Certificate Verification (TFC)
- Certificates are verified online or at TNEA Facilitation Centres (TFCs).
- Eligible candidates are assigned a random number.

### 3. 📊 Rank List Publication
- Overall Rank and Community Rank are released based on normalized 200-mark cutoffs.
- Tie-breaking order: Mathematics > Physics > Optional Subject > Date of Birth (older preferred) > Random Number.

### 4. 🎯 General Online Counselling (4 Rounds)
Students participate round-wise based on their rank:
1. **Choice Filling (3 Days):** Add and prioritize as many college/branch choices as possible in preferred order.
2. **Tentative Allotment:** System allots the highest preferred choice available for your rank and community.
3. **Seat Confirmation (2 Days):**
   - **Accept & Join:** Confirm allotment and proceed to admission.
   - **Accept & Upward:** Hold current seat while competing for a higher preference choice in sliding.
   - **Decline & Upward:** Relinquish current seat and try for higher choices in next round.
   - **Decline & Quit:** Exit counselling.
4. **Provisional Allotment:** Download final allotment letter.

### 5. 🏫 College Reporting & Admission
- Report to the allotted college with original certificates and pay the tuition fees within the designated deadline.

---
👉 *Tell me your **cutoff and category** (e.g. `160 BC CSE Chennai`) to see colleges you can target in counselling!*"""

_GREETINGS = {
    "hi", "hello", "hey", "hai", "hii", "helo", "hiya",
    "good morning", "good afternoon", "good evening", "vanakkam", "sup", "yo",
}

_IDENTITY_PHRASES = [
    "who are you", "who r you", "who r u", "what are you",
    "who is this", "tell me about yourself", "what can you do", "what do you do",
    "what is this chatbot", "what is this bot", "how can you help",
]

_KNOWLEDGE_PHRASES = [
    "what is tnea", "what does tnea", "how does tnea", "tnea full form",
    "how is cutoff calculated", "how is tnea cutoff calculated", "how to calculate cutoff", "cutoff formula",
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
    r"\bbest\b",
    r"\bshow\s+(?:me\s+|my\s+)?(?:colleges?|options?)\b",
    r"\bgive\s+(?:me\s+|my\s+)?(?:colleges?|options?)\b",
    r"\bfind\s+(?:me\s+|my\s+)?(?:colleges?|options?)\b",
    r"\blist\s+(?:of\s+)?(?:colleges?|options?)\b",
    r"\bwhich\s+colleges?\b",
    r"\bwhat\s+colleges?\b",
    r"\bcollege\s+list\b",
    r"\bmy\s+options?\b",
    r"\bcollege\s+options?\b",
    r"\beligible\s+colleges?\b",
    r"\bavailable\s+colleges?\b",
    r"\bcolleges?\s+in\b", r"\bcolleges?\s+for\b", r"\bcolleges?\s+with\b",
    r"\bwhere\s+can\s+i\s+(?:get|join|apply)\b",
]

_COMPARE_PATTERNS = [
    r"\bcompare\b", r"\bcomparison\b", r"\bversus\b",
    r"\bdifference\s+between\b",
]

_EXPLAIN_PATTERNS = [
    r"\bwhy\s+(?:did\s+you|this)\s+(?:recommend|suggest|college)\b",
    r"\bexplain\s+(?:the\s+)?recommendation\b",
    r"\bwhy\s+(?:was|is)\s+(?:this|that)\s+college\b",
    r"\bwhy\s+this\s+college\b",
    r"\bwhy\s+these\s+colleges\b",
    r"\btell\s+me\s+(?:more\s+)?about\s+(?:this|the)\s+recommendation\b",
]

# Branch pairs in "X vs Y" comparisons — treated as general questions, not college compare
_BRANCH_VS_PATTERN = re.compile(
    r"\b(cse|ece|eee|mech|mechanical|civil|it|aiml|aids|aero|auto|bio)\s+vs\s+"
    r"(cse|ece|eee|mech|mechanical|civil|it|aiml|aids|aero|auto|bio)\b",
    re.IGNORECASE,
)


def is_general_question(text: str) -> bool:
    """True if text is a general TNEA knowledge / educational / career question."""
    text_l = text.lower().strip()

    # Cutoff mark or community entry is NOT a general question
    if extract_cutoff(text) is not None or extract_category(text) is not None:
        return False

    question_starters = [
        "what is", "what does", "what do", "what are", "what stand", "what means",
        "how is", "how does", "how to", "how do", "how can", "how many",
        "which is", "which branch", "is it", "is cse", "is ece",
        "is autonomous", "is government",
        "jobs after", "career after", "salary after", "future of",
        "difference between", "explain", "tell me about",
        "what happens", "what should", "why", "where", "when",
    ]
    general_topics = [
        "tnea", "counselling", "choice filling", "seat allotment", "upward movement",
        "sliding", "community", "category", "open category", "backward class",
        "scheduled caste", "autonomous", "government college", "self financing",
        "anna university", "computer science", "mechanical engineering",
        "civil engineering", "electrical and electronics",
        "electronics and communication", "artificial intelligence",
        "data science", "machine learning", "engineering", "placements",
        "salary", "jobs",
    ]

    has_rec     = any(re.search(p, text_l) for p in _REC_PATTERNS)
    has_starter = any(qs in text_l for qs in question_starters) or text_l.endswith("?")
    has_topic   = any(gt in text_l for gt in general_topics)

    return has_starter or (has_topic and not has_rec)


def detect_intent(text: str) -> str:
    """Deterministic intent router. Returns a single intent string."""
    text_l = text.lower().strip()

    if text_l in {"reset", "restart", "reset chat", "start again", "start over", "new chat"}:
        return "reset"

    if any(p in text_l for p in [
        "clear filter", "remove filter", "clear all filter",
        "remove all filter", "reset filter", "reset filters", "clear preferences",
    ]):
        return "clear_filters"

    if any(re.search(p, text_l) for p in _EXPLAIN_PATTERNS):
        return "explain"

    # "compare" intent — but NOT for "CSE vs ECE" style branch comparisons
    has_compare = any(re.search(p, text_l) for p in _COMPARE_PATTERNS)
    has_vs      = re.search(r"\bvs\b", text_l)
    if has_compare or has_vs:
        if not _BRANCH_VS_PATTERN.search(text_l):
            return "compare"

    if any(p in text_l for p in [
        "more college", "more colleges", "show more", "show next",
        "next colleges", "next college", "next", "more",
    ]):
        return "more_results"

    if text_l in _GREETINGS:
        return "greeting"

    if text_l == "help" or any(p in text_l for p in _IDENTITY_PHRASES):
        return "identity"

    if is_college_eligibility_query(text_l):
        return "college_eligibility"

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
# GUIDED FLOW
# ================================================================

_CAT_PROMPT = """\
What is your **community/category**?

| Code | Community |
|:-----|:----------|
| OC   | Open Category |
| BC   | Backward Class |
| BCM  | Backward Class Muslim |
| MBC  | Most Backward Class / MBC/DNC |
| SC   | Scheduled Caste |
| SCA  | SC Arunthathiyar |
| ST   | Scheduled Tribe |

Type the code — e.g. `BC` or `OC`"""

_DIST_PROMPT = """\
Which **district** do you prefer?

Examples: `Chennai`, `Coimbatore`, `Salem`, `Madurai`, `Trichy`, `Vellore`

Type `any` if you have no district preference."""

_BRANCH_PROMPT = """\
Which **branch** are you interested in?

| Code    | Branch |
|:--------|:-------|
| CSE     | Computer Science Engineering |
| ECE     | Electronics & Communication |
| EEE     | Electrical & Electronics |
| IT      | Information Technology |
| MECH    | Mechanical Engineering |
| CIVIL   | Civil Engineering |
| AI & DS | Artificial Intelligence & Data Science |
| AI & ML | Artificial Intelligence & Machine Learning |

Type `any` for all branches."""

_TYPE_PROMPT = """\
What **type of college** are you looking for?

- `Government`
- `Government Aided`
- `Self Financing` (Private)
- `Autonomous`
- `any` (no preference)"""


def _advance_flow(state: Dict, newly_set: List[str]) -> Optional[str]:
    """
    Advance the guided flow after update_state().
    Returns next guidance prompt, or None to fall through to recommendations.
    Multiple fields in one message bypasses the guided flow.
    """
    step   = state.get("_flow_step") or 0
    cutoff = state.get("cutoff")
    cat    = state.get("category")

    # Multiple meaningful fields at once → skip guided flow
    meaningful = [f for f in newly_set if f != "limit"]
    if len(meaningful) > 1:
        if cutoff is not None and cat is not None:
            state["_flow_step"] = 5
        return None

    if "cutoff" in newly_set and step <= 0 and not cat:
        state["_flow_step"] = 1
        return f"Got it 👍 Your cutoff is **{cutoff:g}**.\n\n" + _CAT_PROMPT

    if "category" in newly_set and step <= 1 and cutoff is not None:
        state["_flow_step"] = 2
        return (
            f"Got it. Category: **{COMMUNITY_DISPLAY.get(cat, cat)}**.\n\n"
            + _DIST_PROMPT
        )

    if "district" in newly_set and step <= 2 and cutoff is not None and cat:
        state["_flow_step"] = 3
        return (
            f"Got it. District: **{state.get('district') or 'Any'}**.\n\n"
            + _BRANCH_PROMPT
        )

    if "branch" in newly_set and step <= 3 and cutoff is not None and cat:
        state["_flow_step"] = 4
        return (
            f"Got it. Branch: **{state.get('branch') or 'Any'}**.\n\n"
            + _TYPE_PROMPT
        )

    if "college_type" in newly_set and step <= 4 and cutoff is not None and cat:
        state["_flow_step"] = 5
        return None  # fall through to recommendation

    return None


# ================================================================
# OPENROUTER SYSTEM PROMPT + CALL
# ================================================================

_SYSTEM_PROMPT = """\
You are a TNEA Engineering College Counselling Assistant for Tamil Nadu, India.

RESPONSIBILITIES:
- Answer general questions clearly (TNEA, cutoff calculation, counselling process,
  choice filling, seat allotment, community categories, engineering branches,
  career guidance, college choice advice, etc.)
- Maintain conversation history for natural follow-up questions.
- If off-topic, answer briefly and add: "I'm mainly designed for TNEA guidance."
- For official TNEA dates, fees, or seat matrix, remind users to check tneaonline.org.
- Explain college information ONLY from the structured data supplied by Python.

STRICT DATA RULES:
1. NEVER invent college names, codes, cutoffs, fees, placements, seats, or rankings.
2. NEVER guarantee admission.
3. Use ONLY the supplied college data for college-specific answers.
4. If data is missing, say: "I don't have that information in my TNEA data."
5. Do not say "As an AI language model..."
6. Do not fabricate data to fill gaps.
7. Chance labels: Safe | Good | Moderate | Reach (historical estimate only).

LANGUAGE: Simple English. Tamil-English mix acceptable if student uses it.
TONE: Helpful, direct, honest, realistic, student-friendly."""


def ask_openrouter(
    user_message: str,
    state: Dict,
    context: str,
    chat_history: List[Dict],
) -> str:
    if client is None:
        return (
            "⚠️ OpenRouter API key is not configured. "
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

    except Exception as exc:
        # Log full error to stderr; never expose raw traceback to user
        print(f"[chatbot] OpenRouter error: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        err = str(exc).lower()
        if any(kw in err for kw in ("api key", "authentication", "401", "403")):
            return "⚠️ API key error. Please check your `OPENROUTER_API_KEY` in `.env`."
        if any(kw in err for kw in ("rate limit", "429", "too many")):
            return "⚠️ Rate limit reached. Please wait a moment and try again."
        if any(kw in err for kw in ("timeout", "connection", "network")):
            return (
                "⚠️ Connection to AI service timed out. "
                "Your college data is still available — try a structured search:\n\n"
                "`160 BC CSE Chennai`"
            )
        return (
            "⚠️ I'm temporarily unable to contact the AI service. "
            "Your college data is still available for structured searches.\n\n"
            "Example: `160 BC CSE Chennai`"
        )


# ================================================================
# MAIN MESSAGE ROUTER
# ================================================================

def process_message(
    user_message: str,
    state: Dict,
    chat_history: List[Dict],
) -> Tuple[str, List[Dict]]:
    """
    Route user message → (response_text, updated_chat_history).

    Python  → state extraction, filtering, ranking, tiers, comparison, explanation
    OpenRouter → TNEA knowledge, conversational explanations
    """

    def done(text: str) -> Tuple[str, List[Dict]]:
        return text, chat_history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": text},
        ]

    text_l = user_message.lower().strip()
    intent = detect_intent(user_message)

    # ── RESET ────────────────────────────────────────────────────
    if intent == "reset":
        reset_state(state)
        return (
            "Chat reset. 🔄\n\n"
            "What is your **TNEA cutoff mark**?\n\n"
            "Example: `160` or `my cutoff is 160`",
            [],
        )

    # ── CLEAR FILTERS ────────────────────────────────────────────
    if intent == "clear_filters":
        cutoff_val = state.get("cutoff")
        cat_val    = state.get("category")
        clear_filters(state)
        msg = "Filters cleared ✅"
        notes = []
        if cutoff_val is not None:
            notes.append(f"Cutoff **{cutoff_val:g}** retained.")
        if cat_val:
            notes.append(f"Category **{cat_val}** retained.")
        if notes:
            msg += "\n\n" + "  ".join(notes)
        msg += "\n\nWhat would you like to search for?"
        return done(msg)

    # ── GREETING ─────────────────────────────────────────────────
    if intent == "greeting":
        return done(
            "👋 Hello! I'm your **TNEA Engineering College Counselling Assistant**. 🎓\n\n"
            "I use real TNEA 2025 data to help you find the right colleges.\n\n"
            "**To get started**, tell me your cutoff:\n\n"
            "`160`  or  `my cutoff is 160`\n\n"
            "Or give me everything at once:\n\n"
            "`160 BC CSE Chennai`"
        )

    # ── IDENTITY / HELP ──────────────────────────────────────────
    if intent == "identity":
        return done(
            "I'm a **TNEA Engineering College Counselling Assistant**. 🎓\n\n"
            "**What I can do:**\n"
            "- 🎯 Find colleges by cutoff, category, district, branch, and type\n"
            "- 🔍 Look up any college by code — `college code 2347`\n"
            "- 🔎 Compare two colleges — `compare SVCE and SRM Valliammai`\n"
            "- 💡 Explain my recommendations — `why this college?`\n"
            "- ❓ Answer general TNEA questions\n\n"
            "**To start:** `my cutoff is 160`  or  `160 BC CSE Chennai`"
        )

    # ── EXPLAIN RECOMMENDATION ───────────────────────────────────
    if intent == "explain":
        return done(explain_recommendation(state))

    # ── COLLEGE COMPARISON ───────────────────────────────────────
    if intent == "compare":
        return done(compare_colleges(user_message, branch_filter=state.get("branch")))

    # ── COLLEGE LOOKUP ───────────────────────────────────────────
    if intent == "college_lookup":
        code = extract_college_code(user_message)
        if not code:
            return done("Please provide a college code.\n\nExample: `college code 2347`")
        recs = find_by_code(code)
        if not recs:
            return done(
                f"I couldn't find college code `{code}` in the TNEA data.\n\n"
                "Check the code and try again, or ask me another question."
            )
        state["selected_college"] = code
        return done(format_college_detail(recs))

    # ── COLLEGE ELIGIBILITY CHECK ────────────────────────────────
    if intent == "college_eligibility":
        # Extract and update any parameters from current message
        cutoff       = extract_cutoff(user_message)
        district     = extract_district(user_message)
        branch       = extract_branch(user_message)
        category     = extract_category(user_message)
        ownership    = extract_ownership(user_message)
        autonomous   = extract_autonomous(user_message)

        if cutoff is not None and cutoff <= 200:
            state["cutoff"] = cutoff
        if district is not None and district != "Any":
            state["district"] = district
        if branch is not None and branch != "Any":
            state["branch"] = branch
        if category is not None:
            state["category"] = category
        if ownership is not None and ownership != "Any":
            state["ownership"] = ownership
            state["college_type"] = ownership
        if autonomous is not None:
            state["autonomous"] = autonomous

        resp, matched_code = check_college_eligibility(user_message, state)
        if matched_code:
            state["selected_college"] = matched_code
        if resp:
            return done(resp)

    # ── PCM MARKS INPUT ──────────────────────────────────────────
    pcm = extract_pcm_marks(user_message)
    if any(v is not None for v in pcm.values()):
        m, p, c = pcm["maths"], pcm["physics"], pcm["chemistry"]
        if m is not None and p is not None and c is not None:
            co = m + (p / 2.0) + (c / 2.0)
            state["cutoff"] = co
            state["_flow_step"] = 1

            curr_cat = extract_category(user_message)
            curr_dist = extract_district(user_message)
            curr_br = extract_branch(user_message)
            curr_ow = extract_ownership(user_message)
            curr_auto = extract_autonomous(user_message)

            if curr_cat:
                state["category"] = curr_cat
            if curr_dist and curr_dist != "Any":
                state["district"] = curr_dist
            if curr_br and curr_br != "Any":
                state["branch"] = curr_br
            if curr_ow and curr_ow != "Any":
                state["ownership"] = curr_ow
                state["college_type"] = curr_ow
            if curr_auto is not None:
                state["autonomous"] = curr_auto

            msg = (
                f"Your TNEA cutoff is **{co:g}**\n\n"
                f"Maths: {m:g}\n"
                f"Physics: {p:g}\n"
                f"Chemistry: {c:g}\n\n"
                f"Tell me your preferred branch or district, and I can find suitable colleges."
            )
            return done(msg)
        else:
            missing = []
            if m is None: missing.append("Maths")
            if p is None: missing.append("Physics")
            if c is None: missing.append("Chemistry")
            return done(
                f"Please provide your marks for all three subjects to calculate your cutoff.\n\n"
                f"Missing: **{', '.join(missing)}**\n\n"
                f"Enter your marks in this format:\n`Maths = ?, Physics = ?, Chemistry = ?`"
            )

    # ── CUTOFF FORMULA QUERY ─────────────────────────────────────
    if is_cutoff_formula_query(user_message):
        return done(format_cutoff_calculation_guide())

    # ── COUNSELLING PROCESS QUERY ────────────────────────────────
    if is_counselling_process_query(user_message):
        return done(format_counselling_process_guide())

    # ── PURE NUMERIC INPUT ───────────────────────────────────────
    if re.fullmatch(r"\d{1,6}", text_l):
        num_val = float(text_l)

        # Looks like a cutoff mark (50–200)
        if num_val <= 200:
            state["cutoff"]     = num_val
            state["_flow_step"] = 1
            if not state.get("category"):
                return done(f"Got it 👍 Your cutoff is **{num_val:g}**.\n\n" + _CAT_PROMPT)
            # Cutoff updated with category already set → re-run recommendation
            state["recommendation_offset"] = 0
            resp, top_recs = build_tiered_response(state)
            state["last_recommendations"]  = top_recs
            state["recommendation_offset"] = state.get("limit", 10)
            return done(resp)

        # Looks like a college code (> 200 digits)
        recs = find_by_code(text_l)
        if recs:
            state["selected_college"] = text_l
            return done(format_college_detail(recs))
        return done(
            f"I couldn't find college code `{text_l}` in the TNEA data.\n\n"
            "Check the code and try again, or ask me another question."
        )

    # ── APPLY "ANY" CLEARS ───────────────────────────────────────
    if re.search(r"\bany\s+districts?\b|\ball\s+districts?\b|\bno\s+district\b", text_l):
        state["district"] = None
        state["recommendation_offset"] = 0
    if re.search(r"\bany\s+branch(?:es)?\b|\ball\s+branch(?:es)?\b", text_l):
        state["branch"] = None
        state["recommendation_offset"] = 0
    if re.search(r"\bany\s+(?:college\s+)?type\b|\ball\s+(?:college\s+)?types\b", text_l):
        state["college_type"] = None
        state["ownership"] = None
        state["autonomous"] = None

    # ── EXTRACT FIELDS FROM CURRENT MESSAGE ─────────────────────
    cutoff       = extract_cutoff(user_message)
    district     = extract_district(user_message)
    branch       = extract_branch(user_message)
    category     = extract_category(user_message)
    ownership    = extract_ownership(user_message)
    autonomous   = extract_autonomous(user_message)
    college_type = extract_college_type(user_message)
    limit        = extract_result_limit(user_message)

    # Detect newly set fields for guided flow tracking
    newly_set: List[str] = []

    if cutoff is not None and cutoff <= 200:
        state["cutoff"] = cutoff
        newly_set.append("cutoff")
    if district is not None and district != "Any":
        state["district"] = district
        state["recommendation_offset"] = 0
        newly_set.append("district")
    if branch is not None and branch != "Any":
        state["branch"] = branch
        state["recommendation_offset"] = 0
        newly_set.append("branch")
    if category is not None:
        state["category"] = category
        state["recommendation_offset"] = 0
        newly_set.append("category")
    if ownership is not None and ownership != "Any":
        state["ownership"] = ownership
        state["college_type"] = ownership
        state["recommendation_offset"] = 0
        newly_set.append("ownership")
    if autonomous is not None:
        state["autonomous"] = autonomous
        state["recommendation_offset"] = 0
        newly_set.append("autonomous")
    if limit is not None:
        state["limit"] = limit
        newly_set.append("limit")

    # ── GUIDED FLOW ──────────────────────────────────────────────
    # Intercept: waiting for category (flow_step == 1, cutoff set, no category yet)
    if (
        state.get("cutoff") is not None
        and state.get("category") is None
        and (state.get("_flow_step") or 0) == 1
        and "category" not in newly_set
    ):
        # User replied but we couldn't extract a category
        cat_raw = user_message.strip()
        if not is_general_question(user_message) and len(cat_raw.split()) <= 3:
            return done(suggest_category(cat_raw))

    flow_prompt = _advance_flow(state, newly_set)
    if flow_prompt:
        return done(flow_prompt)

    # ── MORE RESULTS ─────────────────────────────────────────────
    if intent == "more_results":
        offset = state.get("recommendation_offset", 0)
        lim    = state.get("limit", 10)

        if state.get("cutoff") is not None and state.get("category") is not None:
            all_recs = filter_and_rank(state, limit=9999)   # STRICT — no reach for pagination
            if offset >= len(all_recs):
                return done("No more colleges matching your current filters.")
            page = all_recs[offset: offset + lim]
            state["recommendation_offset"] = offset + len(page)
            state["last_recommendations"]  = page
            return done(generate_recommendation_table(page, state, start_offset=offset))
        else:
            all_recs = get_general_colleges(
                district=state.get("district"),
                branch=state.get("branch"),
                college_type=state.get("college_type"),
                ownership=state.get("ownership"),
                autonomous=state.get("autonomous"),
                limit=9999,
            )
            if offset >= len(all_recs):
                return done("No more colleges matching your current filters.")
            page = all_recs[offset: offset + lim]
            state["recommendation_offset"] = offset + len(page)
            return done(build_general_recommendation_response(page, state, start_offset=offset))

    # ── GENERAL vs RECOMMENDATION ROUTING ───────────────────────
    has_rec = (intent == "recommendation") or any(re.search(p, text_l) for p in _REC_PATTERNS)
    is_rec  = has_rec or (
        district is not None or branch is not None
        or ownership is not None or autonomous is not None or limit is not None
    )
    has_gen = is_general_question(user_message)

    # Pure general question → OpenRouter
    if has_gen and not has_rec and intent not in ("college_lookup", "compare", "explain"):
        response = ask_openrouter(
            user_message, state,
            "The user is asking a general TNEA knowledge / educational / career question. "
            "Answer clearly and concisely. Do not invent college data or official TNEA dates/fees.",
            chat_history,
        )
        return done(response)

    # Mixed question (general + recommendation)
    if has_gen and has_rec:
        lim = state.get("limit", 10)
        if state.get("cutoff") is not None and state.get("category") is not None:
            recs = filter_and_rank(state, limit=lim)
            rec_table = generate_recommendation_table(recs, state)
            state["last_recommendations"]  = recs
        else:
            recs = get_general_colleges(
                district=state.get("district"),
                branch=state.get("branch"),
                college_type=state.get("college_type"),
                ownership=state.get("ownership"),
                autonomous=state.get("autonomous"),
                limit=lim,
            )
            rec_table = build_general_recommendation_response(recs, state)
        state["recommendation_offset"] = len(recs)
        response = ask_openrouter(
            user_message, state,
            "MIXED QUESTION INSTRUCTIONS:\n"
            "1. Briefly answer the general question part (2–3 sentences).\n"
            "2. Then present the Python college data table below exactly as-is.\n\n"
            f"PYTHON COLLEGE DATA TABLE:\n{rec_table}",
            chat_history,
        )
        return done(response)

    # ── DIRECT RECOMMENDATION ────────────────────────────────────
    if is_rec or state.get("district") is not None or state.get("branch") is not None or state.get("ownership") is not None:
        state["recommendation_offset"] = 0

        # Personalised — cutoff + category both available
        if state.get("cutoff") is not None and state.get("category") is not None:
            resp, top_recs = build_tiered_response(state)
            state["last_recommendations"]  = top_recs
            state["recommendation_offset"] = state.get("limit", 10)
            return done(resp)

        # Cutoff given but category missing → ask
        if state.get("cutoff") is not None and state.get("category") is None:
            state["_flow_step"] = 1
            return done(
                f"Got it 👍 Your cutoff is **{state['cutoff']:g}**.\n\n" + _CAT_PROMPT
            )

        # No cutoff → general top colleges
        lim  = state.get("limit", 10)
        recs = get_general_colleges(
            district=state.get("district"),
            branch=state.get("branch"),
            college_type=state.get("college_type"),
            ownership=state.get("ownership"),
            autonomous=state.get("autonomous"),
            limit=lim,
        )
        state["recommendation_offset"] = len(recs)
        return done(build_general_recommendation_response(recs, state))

    # ── FALLBACK → OPENROUTER ────────────────────────────────────
    response = ask_openrouter(
        user_message, state,
        "No specific college data for this query. "
        "Answer from general TNEA knowledge. Do not invent college names or data.",
        chat_history,
    )
    return done(response)
