"""
test_harness.py  —  TNEA Chatbot Filter Pipeline Test Suite
============================================================
Tests the REAL extract + filter pipeline from chatbot.py
against the 1,200-question dataset.

PIPELINE UNDER TEST:
  User Question
       ↓
  extract_cutoff / extract_category / extract_district /
  extract_branch / extract_college_type          (chatbot.py)
       ↓
  filter_and_rank / get_general_colleges         (chatbot.py)
       ↓
  Result count + PASS / FAIL

RUN:
  cd tnea_chatbot
  python test_harness.py

  # Flags:
  python test_harness.py --csv questions.csv    # custom dataset file
  python test_harness.py --limit 100            # only first N questions
  python test_harness.py --verbose              # print every row
  python test_harness.py --fail-only            # print only failures
  python test_harness.py --out results.csv      # output file (default: test_results.csv)
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Make chatbot.py importable from any working directory ─────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chatbot import (
    ALL_RECORDS,
    BRANCH_MAP,
    COMMUNITY_DISPLAY,
    DISTRICT_ALIASES,
    VALID_CATEGORIES,
    branch_matches,
    district_matches,
    extract_branch,
    extract_category,
    extract_college_type,
    extract_ownership,
    extract_autonomous,
    extract_cutoff,
    extract_district,
    filter_and_rank,
    get_general_colleges,
    is_any,
    new_state,
    normalize_district,
    college_type_matches,
)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / \
    ".." / ".." / ".." / ".." / \
    "Users/S GOKUL KUMAR/.gemini/antigravity/brain/f892446d-30d1-43fa-9770-83a9775bd1b6/tnea_questions_dataset.csv"

# Also try a sibling path relative to this file
SIBLING_DATASET = Path(__file__).resolve().parent / "tnea_questions_dataset.csv"

RESULTS_FILE = Path(__file__).resolve().parent / "test_results.csv"

REACH_BAND = 5  # matches chatbot.py


# ─────────────────────────────────────────────────────────────────
# CONDITION PARSER  (parses the "Conditions" column from the CSV)
# ─────────────────────────────────────────────────────────────────

def parse_conditions(cond_str: str) -> Dict[str, List[str]]:
    """
    Parse the Conditions column into a dict.
    e.g. "cutoff=180, branch=CSE, district=Coimbatore"
    -> {"cutoff": ["180"], "branch": ["CSE"], "district": ["Coimbatore"]}

    Multi-value keys (branch=X, branch=Y) are collected into lists.
    """
    result: Dict[str, List[str]] = {}
    for part in cond_str.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        key, _, val = part.partition("=")
        key = key.strip().lower()
        val = val.strip()
        result.setdefault(key, []).append(val)
    return result


# ─────────────────────────────────────────────────────────────────
# QUESTION RUNNER  — core test logic
# ─────────────────────────────────────────────────────────────────

class TestResult:
    __slots__ = (
        "q_id", "category", "question",
        "expected_conditions",
        "extracted_cutoff", "extracted_category",
        "extracted_district", "extracted_branch",
        "extracted_ownership", "extracted_autonomous",
        "extracted_college_type",
        "total_matches", "strict_matches", "reach_matches",
        "expected_zero",  # True when 0 results is CORRECT (data has no match)
        "pass_cutoff", "pass_category", "pass_district",
        "pass_branch", "pass_college_type",
        "overall_pass", "fail_reasons",
        "elapsed_ms",
    )

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)
        self.fail_reasons = []
        self.expected_zero = False


def data_supports(district, branch, ownership, autonomous, category, cutoff) -> bool:
    """
    Return True if the verified dataset contains ANY record that can satisfy the
    requested district + branch + college-type(+autonomous) + community + cutoff.

    This mirrors the pipeline's hard filters to decide whether a ZERO result is a
    genuine defect (data exists but not surfaced) or the CORRECT honest answer
    (the dataset simply has no such college). It intentionally returns False when
    the data genuinely lacks a matching college for the exact combination requested.
    """
    cands = ALL_RECORDS

    if district and not is_any(district):
        cands = [r for r in cands if district_matches(r.get("district"), district)]
    if branch and not is_any(branch):
        cands = [r for r in cands if branch_matches(r.get("branch"), branch)]

    if ownership and not is_any(ownership) and ownership != "Autonomous":
        cands = [r for r in cands if college_type_matches(r.get("name"), ownership)]

    if autonomous is True or ownership == "Autonomous":
        cands = [r for r in cands if "autonomous" in r.get("name", "").lower()]
    elif autonomous is False:
        cands = [r for r in cands if "autonomous" not in r.get("name", "").lower()]

    # Pipeline falls back to OC community when category is not supplied.
    target_cat = (category and category.upper()) or "OC"
    cands = [r for r in cands if r.get("community", "") == target_cat]

    if cutoff is not None:
        cands = [
            r for r in cands
            if r.get("cutoff") is not None and r.get("cutoff") <= cutoff + REACH_BAND
        ]

    return len(cands) > 0


def run_question(q_id: int, category: str, question: str,
                 conditions_str: str) -> TestResult:
    """
    Run a single question through the extract → filter pipeline.
    Returns a TestResult with extraction accuracy and match counts.
    """
    r = TestResult()
    r.q_id = q_id
    r.category = category
    r.question = question
    r.expected_conditions = parse_conditions(conditions_str)

    t0 = time.perf_counter()

    # ── 1. EXTRACTION ─────────────────────────────────────────────
    r.extracted_cutoff       = extract_cutoff(question)
    r.extracted_category     = extract_category(question)
    r.extracted_district     = extract_district(question)
    r.extracted_branch       = extract_branch(question)
    r.extracted_ownership    = extract_ownership(question)
    r.extracted_autonomous   = extract_autonomous(question)
    r.extracted_college_type = extract_college_type(question)

    # ── 2. BUILD STATE ────────────────────────────────────────────
    state = new_state()
    state["cutoff"]       = r.extracted_cutoff
    state["category"]     = r.extracted_category or "OC"  # default
    state["district"]     = r.extracted_district
    state["branch"]       = r.extracted_branch
    state["ownership"]    = r.extracted_ownership
    state["autonomous"]   = r.extracted_autonomous
    state["college_type"] = r.extracted_ownership

    # ── 3. FILTER ─────────────────────────────────────────────────
    if r.extracted_cutoff is not None and r.extracted_category is not None:
        # Personalized: strict + reach
        strict_recs = filter_and_rank(state, limit=9999, with_reach=False)
        reach_recs  = filter_and_rank(state, limit=9999, with_reach=True)
        r.strict_matches = len(strict_recs)
        r.reach_matches  = len(reach_recs) - len(strict_recs)
        r.total_matches  = len(reach_recs)
    else:
        # General: use OC proxy
        general_recs = get_general_colleges(
            district=r.extracted_district,
            branch=r.extracted_branch,
            ownership=r.extracted_ownership,
            autonomous=r.extracted_autonomous,
            limit=9999,
        )
        r.strict_matches = len(general_recs)
        r.reach_matches  = 0
        r.total_matches  = len(general_recs)

    r.elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    # ── 4. PASS/FAIL PER FIELD ────────────────────────────────────
    exp = r.expected_conditions

    # CUTOFF
    exp_cutoffs = exp.get("cutoff", [])
    if not exp_cutoffs or exp_cutoffs == ["general"]:
        r.pass_cutoff = True  # not required by this question
    else:
        try:
            exp_co = float(exp_cutoffs[0])
            r.pass_cutoff = (r.extracted_cutoff is not None and
                             abs(r.extracted_cutoff - exp_co) < 0.6)
        except ValueError:
            r.pass_cutoff = True

    # CATEGORY / COMMUNITY
    exp_cats = exp.get("community", [])
    if not exp_cats:
        r.pass_category = True
    else:
        # Map expected community label → canonical form using VALID_CATEGORIES
        exp_canon = VALID_CATEGORIES.get(exp_cats[0].lower())
        r.pass_category = (r.extracted_category == exp_canon) if exp_canon else True

    # DISTRICT (accept if extracted matches ANY expected district)
    exp_dists = exp.get("district", [])
    if not exp_dists or exp_dists == ["general"]:
        r.pass_district = True
    else:
        ext_d = normalize_district(r.extracted_district)
        matched_any_d = False
        for exp_d_raw in exp_dists:
            exp_d = normalize_district(exp_d_raw)
            if ext_d and exp_d and ext_d.lower() == exp_d.lower():
                matched_any_d = True
                break
        r.pass_district = matched_any_d

    # BRANCH  (accept if ANY expected branch was extracted)
    exp_branches = exp.get("branch", [])
    if not exp_branches or exp_branches == ["general"]:
        r.pass_branch = True
    else:
        # Map dataset branch labels to BRANCH_MAP keys
        _LABEL_TO_KEY = {
            "CSE": "CSE", "Information Technology": "IT", "ECE": "ECE",
            "EEE": "EEE", "Mechanical Engineering": "MECH",
            "Civil Engineering": "CIVIL", "Chemical Engineering": "CHEM",
            "Petrochemical Engineering": "PETROCHEM",
            "Biotechnology": "BIO", "Biomedical Engineering": "BIO",
            "AI & Data Science": "AI & DS", "AI & ML": "AI & ML",
            "Automobile Engineering": "AUTO",
            "Aeronautical Engineering": "AERO",
            "Mechatronics": "MECHAT",
            "Food Technology": "FOOD",
            "Industrial Engineering": "INDENG",
            "Robotics and Automation": "ROBOTICS",
            "Marine Engineering": "MARINE",
            "Agriculture Engineering": "AGRI",
            "CSBS": "CSBS",
            "Data Science": "AI & DS",
            "Cyber Security": "CYBER",
            "Cloud Computing": "IT",
            "Electronics and Instrumentation": "EIE",
            "Software Engineering": "CSE",
        }
        matched_any = False
        for eb in exp_branches:
            key = _LABEL_TO_KEY.get(eb)
            if key and r.extracted_branch == key:
                matched_any = True
                break
            # Loose: if extracted branch name appears in expected
            if r.extracted_branch and r.extracted_branch.lower() in eb.lower():
                matched_any = True
                break
        r.pass_branch = matched_any

    # COLLEGE TYPE (OWNERSHIP + AUTONOMOUS)
    exp_ownership = exp.get("ownership", [])
    exp_auto      = exp.get("autonomous", [])

    if not exp_ownership or exp_ownership == ["general"]:
        pass_ow = True
    else:
        _OW_MAP = {
            "Government": "Government",
            "Government Aided": "Government Aided",
            "Private": "Self Financing",
            "Self-Financing": "Self Financing",
        }
        expected_types = [_OW_MAP.get(v, v) for v in exp_ownership]
        pass_ow = (r.extracted_ownership in expected_types) if r.extracted_ownership else False

    if not exp_auto or exp_auto == ["comparison"] or exp_auto == ["general"]:
        pass_auto = True
    else:
        if "yes" in [v.lower() for v in exp_auto]:
            pass_auto = (r.extracted_autonomous is True)
        elif "no" in [v.lower() for v in exp_auto]:
            pass_auto = (r.extracted_autonomous is False)
        else:
            pass_auto = True

    r.pass_college_type = pass_ow and pass_auto

    # ── OVERALL ───────────────────────────────────────────────────
    field_passes = [r.pass_cutoff, r.pass_category, r.pass_district,
                    r.pass_branch, r.pass_college_type]
    r.fail_reasons = []

    # Presence checks: question requires results
    needs_results = (r.extracted_cutoff is not None or
                     r.extracted_district is not None or
                     r.extracted_branch is not None)
    if needs_results and r.total_matches == 0:
        # Distinguish a real filter defect from a CORRECT empty answer.
        # If the verified dataset has a college matching district+branch+type
        # +community+cutoff yet the pipeline found none → ZERO_RESULTS (bug).
        # If no such college exists in the data → EXPECTED_ZERO (honest no-data
        # answer, aligned with the app's "never fabricate" principle).
        if data_supports(
            r.extracted_district, r.extracted_branch,
            r.extracted_ownership, r.extracted_autonomous,
            r.extracted_category, r.extracted_cutoff,
        ):
            r.fail_reasons.append("ZERO_RESULTS")
        else:
            r.expected_zero = True

    if not r.pass_cutoff:    r.fail_reasons.append("CUTOFF_MISMATCH")
    if not r.pass_category:  r.fail_reasons.append("CATEGORY_MISMATCH")
    if not r.pass_district:  r.fail_reasons.append("DISTRICT_NOT_EXTRACTED")
    if not r.pass_branch:    r.fail_reasons.append("BRANCH_NOT_EXTRACTED")
    if not r.pass_college_type: r.fail_reasons.append("TYPE_NOT_EXTRACTED")

    r.overall_pass = len(r.fail_reasons) == 0
    return r


# ─────────────────────────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────────────────────────

def load_questions(path: Path, limit: Optional[int] = None) -> List[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────
# REPORT WRITER
# ─────────────────────────────────────────────────────────────────

RESULT_HEADERS = [
    "ID", "Category", "Question",
    "Expected_Conditions",
    "Extracted_Cutoff", "Extracted_Category", "Extracted_District",
    "Extracted_Branch", "Extracted_Type",
    "Total_Matches", "Strict_Matches", "Reach_Matches",
    "Expected_Zero",
    "Pass_Cutoff", "Pass_Category", "Pass_District",
    "Pass_Branch", "Pass_Type",
    "Overall_Pass", "Fail_Reasons", "Elapsed_ms",
]


def write_results(results: List[TestResult], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_HEADERS)
        w.writeheader()
        for r in results:
            w.writerow({
                "ID":                   r.q_id,
                "Category":             r.category,
                "Question":             r.question,
                "Expected_Conditions":  r.expected_conditions,
                "Extracted_Cutoff":     r.extracted_cutoff,
                "Extracted_Category":   r.extracted_category,
                "Extracted_District":   r.extracted_district,
                "Extracted_Branch":     r.extracted_branch,
                "Extracted_Type":       r.extracted_college_type,
                "Total_Matches":        r.total_matches,
                "Strict_Matches":       r.strict_matches,
                "Reach_Matches":        r.reach_matches,
                "Expected_Zero":        "✓" if r.expected_zero else "",
                "Pass_Cutoff":          "✓" if r.pass_cutoff else "✗",
                "Pass_Category":        "✓" if r.pass_category else "✗",
                "Pass_District":        "✓" if r.pass_district else "✗",
                "Pass_Branch":          "✓" if r.pass_branch else "✗",
                "Pass_Type":            "✓" if r.pass_college_type else "✗",
                "Overall_Pass":         "PASS" if r.overall_pass else "FAIL",
                "Fail_Reasons":         "; ".join(r.fail_reasons),
                "Elapsed_ms":           r.elapsed_ms,
            })


# ─────────────────────────────────────────────────────────────────
# SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────────

def print_summary(results: List[TestResult]) -> None:
    total   = len(results)
    passed  = sum(1 for r in results if r.overall_pass)
    failed  = total - passed
    expzero = sum(1 for r in results if r.expected_zero)

    # Per-category stats
    by_cat: Dict[str, List[bool]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r.overall_pass)

    # Failure breakdown
    fail_counts: Dict[str, int] = {}
    for r in results:
        for reason in r.fail_reasons:
            fail_counts[reason] = fail_counts.get(reason, 0) + 1

    avg_ms = sum(r.elapsed_ms for r in results) / total if total else 0

    print("\n" + "=" * 65)
    print("  TNEA CHATBOT FILTER PIPELINE — TEST RESULTS")
    print("=" * 65)
    print(f"  Total Questions : {total:>6}")
    print(f"  PASSED          : {passed:>6}  ({100*passed/total:.1f}%)")
    print(f"  FAILED          : {failed:>6}  ({100*failed/total:.1f}%)")
    print(f"  Expected zero   : {expzero:>6}  (correct empty answers, no data match)")
    print(f"  Avg time/q      : {avg_ms:>6.2f} ms")
    print(f"  Data records    : {len(ALL_RECORDS):>6,}")
    print("-" * 65)
    print("  PASS RATE BY CATEGORY")
    print("-" * 65)
    for cat, outcomes in sorted(by_cat.items()):
        p = sum(outcomes)
        t = len(outcomes)
        filled = int(p / t * 20)
        bar = "#" * filled + "." * (20 - filled)
        print(f"  {cat:<35} [{bar}]  {p}/{t} ({100*p/t:.0f}%)")
    print("-" * 65)
    print("  FAILURE BREAKDOWN")
    print("-" * 65)
    for reason, count in sorted(fail_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:<35} {count:>4} failures")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TNEA Chatbot Filter Pipeline Test Harness"
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Path to question dataset CSV (default: auto-detected)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max questions to test (default: all)",
    )
    parser.add_argument(
        "--out", type=Path, default=RESULTS_FILE,
        help="Output CSV path (default: test_results.csv)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print every question result",
    )
    parser.add_argument(
        "--fail-only", action="store_true",
        help="Print only failed questions",
    )
    args = parser.parse_args()

    # ── Locate dataset ────────────────────────────────────────────
    dataset_path = args.csv
    if dataset_path is None:
        # Try sibling first (copy dataset beside this script)
        if SIBLING_DATASET.exists():
            dataset_path = SIBLING_DATASET
        else:
            # Walk up to find the artifact
            home = Path.home()
            candidates = [
                home / ".gemini" / "antigravity" / "brain" /
                "f892446d-30d1-43fa-9770-83a9775bd1b6" /
                "tnea_questions_dataset.csv",
                Path(__file__).resolve().parent / "tnea_questions_dataset.csv",
            ]
            for c in candidates:
                if c.exists():
                    dataset_path = c
                    break

    if dataset_path is None or not dataset_path.exists():
        print(
            "❌  Cannot find tnea_questions_dataset.csv.\n"
            "    Either:\n"
            "      a) Copy it beside test_harness.py, or\n"
            "      b) Run:  python test_harness.py --csv /path/to/dataset.csv\n"
        )
        sys.exit(1)

    print(f"\n[Dataset]  : {dataset_path}")
    print(f"[Records]  : {len(ALL_RECORDS):,} loaded from data.txt")
    print(f"[Output]   : {args.out}")

    # ── Load questions ────────────────────────────────────────────
    questions = load_questions(dataset_path, limit=args.limit)
    print(f"[Running]  : {len(questions)} questions\n")

    # ── Run tests ─────────────────────────────────────────────────
    results: List[TestResult] = []
    for i, row in enumerate(questions, 1):
        q_id      = int(row.get("ID", i))
        category  = row.get("Category", "").strip()
        question  = row.get("Question", "").strip()
        conditions = row.get("Conditions", "").strip()

        result = run_question(q_id, category, question, conditions)
        results.append(result)

        # ── Console output ─────────────────────────────────────────
        if args.verbose or (args.fail_only and not result.overall_pass):
            status = "[PASS]" if result.overall_pass else "[FAIL]"
            print(f"  [{q_id:>4}] {status}  ({result.elapsed_ms:5.1f}ms)  {result.total_matches:>4} results")
            print(f"         Q: {question[:80]}")
            if not result.overall_pass:
                print(f"         Fail: {'; '.join(result.fail_reasons)}")
                print(f"         Extracted: cutoff={result.extracted_cutoff} "
                      f"cat={result.extracted_category} "
                      f"dist={result.extracted_district} "
                      f"branch={result.extracted_branch} "
                      f"ownership={result.extracted_ownership} "
                      f"auto={result.extracted_autonomous}")
            print()

        elif i % 100 == 0:
            passed_so_far = sum(1 for r in results if r.overall_pass)
            print(f"  Progress: {i:>4}/{len(questions)}  |  "
                  f"Pass rate so far: {100*passed_so_far/i:.1f}%")

    # ── Write CSV ─────────────────────────────────────────────────
    write_results(results, args.out)
    print(f"\n[DONE] Results written to: {args.out}")

    # ── Print summary ─────────────────────────────────────────────
    print_summary(results)

    # ── Exit code ─────────────────────────────────────────────────
    total  = len(results)
    passed = sum(1 for r in results if r.overall_pass)
    rate   = passed / total if total else 0
    sys.exit(0 if rate >= 0.70 else 1)  # fail CI if pass rate < 70%


if __name__ == "__main__":
    main()
