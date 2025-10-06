#!/usr/bin/env python3
"""
validate_query_jsonl.py

Validate JSONL datasets of the form:
{"query": "string", "labels": {"intent": "...", "ambiguity_score": 0-1, "is_overview": 0-1, "is_compositional": 0-1, "is_alias_or_underspecified": 0-1}}

Usage:
  python validate_query_jsonl.py /path/to/query_claud.jsonl
  python validate_query_jsonl.py /path/to/folder/with/jsonl
  python validate_query_jsonl.py /path/to/folder --strict --max-errors 50

Exit code is 0 if all files pass; 1 if any error is found.
"""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Tuple

ALLOWED_INTENTS = {"chitchat", "general_knowledge", "internal_knowledge", "unsafe"}
REQUIRED_LABEL_FLOAT_KEYS = [
    "ambiguity_score",
    "is_overview",
    "is_compositional",
    "is_alias_or_underspecified",
]

def is_float_0_1(x: Any) -> bool:
    try:
        # accept int or float, but within [0,1]
        xf = float(x)
    except (TypeError, ValueError):
        return False
    return 0.0 <= xf <= 1.0

def iter_jsonl_lines(p: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                # skip empty lines but still count
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p.name}:{i} JSON parse error: {e.msg} at pos {e.pos}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{p.name}:{i} Each line must be a JSON object, got {type(obj).__name__}")
            yield i, obj

def validate_record(obj: Dict[str, Any], strict: bool = False) -> Tuple[bool, str]:
    # query
    if "query" not in obj:
        return False, "missing field: 'query'"
    if not isinstance(obj["query"], str) or not obj["query"].strip():
        return False, "'query' must be a non-empty string"

    # labels
    if "labels" not in obj:
        return False, "missing field: 'labels'"
    labels = obj["labels"]
    if not isinstance(labels, dict):
        return False, "'labels' must be an object"

    # intent
    intent = labels.get("intent")
    if intent is None:
        return False, "labels.intent missing"
    if intent not in ALLOWED_INTENTS:
        return False, f"labels.intent '{intent}' not in {sorted(ALLOWED_INTENTS)}"

    # float-in-[0,1] keys
    for k in REQUIRED_LABEL_FLOAT_KEYS:
        v = labels.get(k)
        if v is None:
            return False, f"labels.{k} missing"
        if not is_float_0_1(v):
            return False, f"labels.{k} must be a number in [0,1], got {v!r}"

    # strict mode: disallow unknown top-level keys or unknown label keys
    if strict:
        allowed_top = {"query", "labels"}
        unknown_top = set(obj.keys()) - allowed_top
        if unknown_top:
            return False, f"unknown top-level keys in strict mode: {sorted(unknown_top)}"

        allowed_label_keys = {"intent", *REQUIRED_LABEL_FLOAT_KEYS}
        unknown_label = set(labels.keys()) - allowed_label_keys
        if unknown_label:
            return False, f"unknown 'labels' keys in strict mode: {sorted(unknown_label)}"

    return True, ""

def gather_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() == ".jsonl":
            yield path
        else:
            print(f"Skipping non-JSONL file: {path}", file=sys.stderr)
    elif path.is_dir():
        for p in sorted(path.rglob("*.jsonl")):
            if p.is_file():
                yield p
    else:
        print(f"Path not found: {path}", file=sys.stderr)

def validate_file(p: Path, strict: bool, max_errors: int) -> Tuple[bool, Dict[str, Any]]:
    errors = []
    seen_intents = Counter()
    count = 0
    for lineno, obj in iter_jsonl_lines(p):
        ok, msg = validate_record(obj, strict=strict)
        count += 1
        if ok:
            seen_intents[obj["labels"]["intent"]] += 1
        else:
            errors.append(f"{p.name}:{lineno}: {msg}")
            if len(errors) >= max_errors:
                errors.append(f"... stopping after {max_errors} errors for {p.name}")
                break
    return (len(errors) == 0, {
        "file": str(p),
        "lines_checked": count,
        "intent_distribution": dict(seen_intents),
        "errors": errors
    })

def main():
    ap = argparse.ArgumentParser(description="Validate query JSONL files.")
    ap.add_argument("path", type=Path, help="Path to a .jsonl file or a directory.")
    ap.add_argument("--strict", action="store_true", help="Disallow unknown keys.")
    ap.add_argument("--max-errors", type=int, default=100, help="Max errors to report per file.")
    args = ap.parse_args()

    files = list(gather_files(args.path))
    if not files:
        print("No .jsonl files found.", file=sys.stderr)
        sys.exit(1)

    any_fail = False
    global_counts = Counter()
    per_file_results = []

    for f in files:
        ok, res = validate_file(f, strict=args.strict, max_errors=args.max_errors)
        per_file_results.append((ok, res))
        if not ok:
            any_fail = True
        for k, v in res["intent_distribution"].items():
            global_counts[k] += v

    # Report
    print("="*80)
    print("Validation Summary")
    print("="*80)
    total_lines = sum(r["lines_checked"] for _, r in per_file_results)
    print(f"Files checked: {len(files)}")
    print(f"Records checked: {total_lines}")
    print("Intent distribution (all files):")
    for intent in sorted(ALLOWED_INTENTS):
        print(f"  - {intent}: {global_counts[intent]}")

    print("\nPer-file results:")
    for ok, r in per_file_results:
        status = "PASS" if ok else "FAIL"
        print(f"\n[{status}] {r['file']}")
        print(f"  lines: {r['lines_checked']}")
        if r["intent_distribution"]:
            print("  intents:")
            for k, v in sorted(r["intent_distribution"].items()):
                print(f"    - {k}: {v}")
        if r["errors"]:
            print("  errors:")
            for e in r["errors"]:
                print(f"    * {e}")

    sys.exit(0 if not any_fail else 1)

if __name__ == "__main__":
    main()
