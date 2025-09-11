#!/usr/bin/env python3
"""
dt_equivalence.py

Detects date-like columns in a CSV, infers their types/formats, normalizes them to UTC,
and groups columns that represent the same instant (equivalent) across rows.

Usage:
  python dt_equivalence.py --in data.csv [--delimiter ,] [--naive-tz America/Chicago] [--max-rows 1000000]

Outputs:
  - Prints a human-readable report to stdout
  - Writes a JSON report next to the CSV: <csvname>.dt_report.json
  - Optionally writes a normalized preview CSV: <csvname>.dt_preview.csv

Notes:
  - This is a practical MVP aimed at messy real-world tables.
  - It tries fast, vectorized paths first (pandas.to_datetime), with fallbacks.
  - Equivalence is determined by row-wise equality of normalized UTC values
    (within a tolerance of 1 second), over the intersection of non-null rows.
"""

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil import parser as du_parser
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# ---------- Heuristics & helpers ----------

CANDIDATE_NAME_HINTS = [
    "time", "date", "timestamp", "ts", "_at", "created", "updated", "modified", "event", "logged"
]

TZ_ABBR_MAP = {
    # Common US DST/standard pairs
    "UTC": "UTC", "Z": "UTC", "GMT": "UTC",
    "PST": "America/Los_Angeles", "PDT": "America/Los_Angeles",
    "MST": "America/Denver",      "MDT": "America/Denver",
    "CST": "America/Chicago",     "CDT": "America/Chicago",
    "EST": "America/New_York",    "EDT": "America/New_York",
    # Add more if your data uses them (AEST/AEDT, IST, etc.)
}

# For epoch unit detection
def detect_epoch_unit(series: pd.Series) -> Optional[str]:
    """Guess the epoch unit for a numeric-like series by magnitude."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    if s.empty:
        return None
    mn, mx = s.min(), s.max()
    # Years range sanity check helpers
    def plausible_year_range(epoch_values: pd.Series, unit: str) -> bool:
        if unit == "s":
            vals = epoch_values
        elif unit == "ms":
            vals = epoch_values / 1e3
        elif unit == "us":
            vals = epoch_values / 1e6
        else:
            vals = epoch_values / 1e9  # ns
        # convert to year roughly
        years = pd.to_datetime(vals, unit="s", utc=True, errors="coerce").dt.year
        if years.isna().all():
            return False
        # Acceptable year range (1970..2100) by default
        ok = years.between(1970, 2100).mean()
        return ok > 0.95

    # Magnitude heuristics
    # ns ~1e18, us ~1e15, ms ~1e12, s ~1e9 (around now)
    if mn > 1e17 and mx < 1e19:
        candidate = "ns"
        return candidate if plausible_year_range(s, candidate) else None
    if mn > 1e14 and mx < 1e16:
        candidate = "us"
        return candidate if plausible_year_range(s, candidate) else None
    if mn > 1e11 and mx < 1e13:
        candidate = "ms"
        return candidate if plausible_year_range(s, candidate) else None
    if mn > 1e8 and mx < 1e10:
        candidate = "s"
        return candidate if plausible_year_range(s, candidate) else None

    # Fallback: try candidates by plausibility
    for cand in ["s", "ms", "us", "ns"]:
        if plausible_year_range(s, cand):
            return cand
    return None



def classify_date_string(val: str) -> str:
    """Best-effort classification of a date string format."""
    if not isinstance(val, str):
        return ""
    s = val.strip()
    if not s:
        return ""
    # ISO8601 with Z
    if re.match(r'^\d{4}-\d{2}-\d{2}T.*Z$', s):
        return "iso8601_z"
    # ISO8601 with offset
    if re.match(r'^\d{4}-\d{2}-\d{2}T.*[+-]\d{2}:?\d{2}$', s):
        return "iso8601_offset"
    # Naive ISO-like YYYY-MM-DD HH:MM:SS
    if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', s):
        return "naive_iso"
    # RFC 5322 / email-style (starts with day-of-week, has GMT/UTC)
    if re.match(r'^[A-Za-z]{3}, \d{1,2} [A-Za-z]{3} \d{4}', s):
        return "rfc5322"
    # HTTP-date (same as RFC1123)
    if re.match(r'^[A-Za-z]{3}, \d{2} [A-Za-z]{3} \d{4}', s):
        return "http_date"
    # Slashy
    if re.match(r'^\d{1,2}/\d{1,2}/\d{4}', s):
        # crude: if first token > 12 then must be DMY
        try:
            first = int(s.split("/")[0])
            if first > 12:
                return "slash_dmy"
            else:
                return "slash_mdy"
        except:
            return "slash"
    return "other"

def looks_like_date_string(s: str) -> bool:
    """Quick-and-dirty checks to avoid attempting to parse every random string."""
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s:
        return False
    # contains digits and either '-' or '/' or ':' or 'T' or timezone letters
    if any(ch in s for ch in "-/:T+") and re.search(r"\d", s):
        return True
    # RFC-like day/month names
    if re.search(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun),", s, re.I):
        return True
    if re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", s, re.I):
        return True
    return False


def parse_string_to_utc(series: pd.Series, naive_tz: Optional[str]) -> Tuple[pd.Series, Dict]:
    """
    Parse a string series into UTC datetimes.
    Strategy:
      1) pandas.to_datetime with utc=True (fast path for ISO/RFC)
      2) dateutil parser per row fallback (handles tz abbreviations via TZ_ABBR_MAP)
      3) If naive (no tz info), apply naive_tz if provided; otherwise assume UTC and flag it.
    Returns:
      (utc_series, meta)
    """
    meta = {"parser": None, "naive_policy": None, "notes": []}

    # Try vectorized fast path
    try:
        dt = pd.to_datetime(series, errors="coerce", utc=True, )
        if dt.notna().mean() > 0.8:
            meta["parser"] = "pandas.to_datetime(utc=True)"
            # Handle naive detection (pandas will treat some naive strings as local? It won't; utc=True keeps as UTC assuming naive)
            # We can't easily know which were naive vs tz-aware here; document policy:
            if naive_tz:
                meta["naive_policy"] = f"naive→{naive_tz} (best-effort)"
            return dt, meta
    except Exception as e:
        meta["notes"].append(f"fast_path_error: {e!r}")

    # Fallback: row-wise with dateutil
    out = []
    naive_count = 0
    for val in series.astype(str).tolist():
        val = val.strip()
        if not val or not looks_like_date_string(val):
            out.append(pd.NaT)
            continue
        try:
            # Replace TZ abbrev with IANA where obvious (simple token replace)
            tokens = val.split()
            tokens = [TZ_ABBR_MAP.get(tok, tok) for tok in tokens]
            val_norm = " ".join(tokens)
            dt = du_parser.parse(val_norm, tzinfos=TZ_ABBR_MAP)
            if dt.tzinfo is None:
                naive_count += 1
                if naive_tz:
                    dt = dt.replace(tzinfo=ZoneInfo(naive_tz))
                else:
                    # Assume UTC if not provided
                    dt = dt.replace(tzinfo=timezone.utc)
            out.append(dt.astimezone(timezone.utc))
        except Exception:
            out.append(pd.NaT)
    utc_series = pd.to_datetime(pd.Series(out), errors="coerce", utc=True)
    meta["parser"] = "dateutil.parse (row-wise)"
    meta["naive_policy"] = f"naive→{naive_tz or 'UTC'} (row-wise)"
    meta["notes"].append(f"naive_rows={naive_count}")
    return utc_series, meta


def parse_epoch_to_utc(series: pd.Series, unit: str) -> Tuple[pd.Series, Dict]:
    meta = {"parser": f"epoch({unit})", "naive_policy": None, "notes": []}
    s = pd.to_numeric(series, errors="coerce")
    dt = pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
    return dt, meta


@dataclass
class ColumnReport:
    name: str
    role: str  # 'date_string' | 'epoch' | 'non_date'
    parse_rate: float
    detected_unit: Optional[str] = None
    parser: Optional[str] = None
    naive_policy: Optional[str] = None
    format_guess: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def detect_date_columns(df: pd.DataFrame, naive_tz: Optional[str]) -> Tuple[Dict[str, pd.Series], Dict[str, ColumnReport]]:
    parsed_utc: Dict[str, pd.Series] = {}
    reports: Dict[str, ColumnReport] = {}

    for col in df.columns:
        s = df[col]
        # Skip obvious non-candidates quickly
        name_l = col.lower()
        name_hint = any(hint in name_l for hint in CANDIDATE_NAME_HINTS)

        # If numeric-ish and name hint or magnitude suggests epoch
        is_numeric_like = pd.to_numeric(s, errors="coerce").notna().mean() > 0.8
        detected_unit = None

        role = "non_date"
        parser = None
        naive_policy = None
        notes: List[str] = []

        utc_series = None

        if is_numeric_like:
            detected_unit = detect_epoch_unit(s)
            if detected_unit:
                utc_series, meta = parse_epoch_to_utc(s, detected_unit)
                role = "epoch"
                parser = meta["parser"]
                parse_rate = utc_series.notna().mean()
                reports[col] = ColumnReport(name=col, role=role, parse_rate=parse_rate, detected_unit=detected_unit, parser=parser, naive_policy=None, format_guess=f'epoch:{detected_unit}', notes=notes)
                parsed_utc[col] = utc_series
                continue  # move to next column

        # Try string parsing if name hints or many stringy date-like values
        # Check sample for date-ish strings
        sample = s.dropna().astype(str).head(200).tolist()
        stringy_rate = 0.0
        if sample:
            stringy_rate = sum(looks_like_date_string(v) for v in sample) / len(sample)

        if name_hint or stringy_rate > 0.3:
            utc_series, meta = parse_string_to_utc(s.astype(str), naive_tz)
            role = "date_string"
            parser = meta["parser"]
            naive_policy = meta.get("naive_policy")
            parse_rate = utc_series.notna().mean()
            notes.extend(meta.get("notes", []))
            fmt_guess = ''
            if role == 'date_string':
                sample_vals = [v for v in sample if v]
                if sample_vals:
                    guesses = [classify_date_string(v) for v in sample_vals[:20]]
                    # pick most common non-empty guess
                    if guesses:
                        from collections import Counter
                        fmt_guess, _ = Counter([g for g in guesses if g]).most_common(1)[0]
            reports[col] = ColumnReport(name=col, role=role, parse_rate=parse_rate, detected_unit=detected_unit, parser=parser, naive_policy=naive_policy, format_guess=fmt_guess, notes=notes)
            parsed_utc[col] = utc_series
        else:
            # not a date column
            reports[col] = ColumnReport(name=col, role=role, parse_rate=0.0, detected_unit=None, parser=None, naive_policy=None, format_guess=None, notes=notes)

    return parsed_utc, reports


def equivalence_groups(parsed: Dict[str, pd.Series], tolerance_seconds: int = 1, min_overlap: int = 100, min_match_ratio: float = 0.98):
    """
    Build groups of columns that represent the same instant after normalization.
    Two columns A,B are considered equivalent if:
      - Overlap (rows where both not null) >= min_overlap (or 1% of rows if bigger)
      - Among the overlap, proportion of |A-B| <= tolerance <= min_match_ratio
    Returns a list of groups (each a list of column names) and a similarity matrix.
    """
    cols = list(parsed.keys())
    n = len(cols)
    if n == 0:
        return [], {}

    # build matrix
    sim: Dict[Tuple[str, str], Dict[str, float]] = {}
    total_rows = max(len(v) for v in parsed.values())
    dyn_min_overlap = max(min_overlap, int(0.01 * total_rows))

    # Convert to int seconds for speed; use NaN for NaT
    as_int = {}
    for c in cols:
        s = parsed[c]
        # Pandas stores UTC timestamps as datetime64[ns]; convert to int seconds
        si = (s.astype("int64", copy=False) // 10**9).astype("float64")
        si[s.isna()] = np.nan
        as_int[c] = si

    # pairwise comparison
    parent = {c: c for c in cols}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            a, b = cols[i], cols[j]
            va, vb = as_int[a], as_int[b]
            mask = (~np.isnan(va)) & (~np.isnan(vb))
            overlap = int(mask.sum())
            if overlap < dyn_min_overlap:
                sim[(a, b)] = {"overlap": overlap, "match_ratio": 0.0, "equivalent": False}
                continue
            diff = np.abs(va[mask] - vb[mask])
            matches = int((diff <= tolerance_seconds).sum())
            match_ratio = matches / overlap if overlap else 0.0
            eq = match_ratio >= min_match_ratio
            sim[(a, b)] = {"overlap": overlap, "match_ratio": match_ratio, "equivalent": bool(eq)}
            if eq:
                union(a, b)

    # build groups from union-find
    groups_map: Dict[str, List[str]] = {}
    for c in cols:
        r = find(c)
        groups_map.setdefault(r, []).append(c)
    groups = list(groups_map.values())

    return groups, sim



def print_report(csv_path: str, reports: Dict[str, ColumnReport], groups: List[List[str]], sim_matrix: Dict[Tuple[str, str], Dict[str, float]], preview_path: Optional[str] = None):
    print("\n=== dt-equivalence report ===")
    print(f"Source: {csv_path}")
    if preview_path:
        print(f"Normalized preview (first 50 rows) written to: {preview_path}")
    print("\n-- Detected columns --")
    rows = []
    for name, rep in reports.items():
        rows.append([
            name,
            rep.role,
            (rep.format_guess or ""),
            (rep.detected_unit or ""),
            f"{rep.parse_rate:.1%}",
            (rep.parser or ""),
            (rep.naive_policy or ""),
            ("; ".join(rep.notes) if rep.notes else "")
        ])
    # Pretty print as a simple table
    header = ["column","role","format","unit","parse_rate","parser","naive_policy","notes"]
    if not rows:
        colw = [len(h) for h in header]
    else:
        colw = [max(len(str(x[i])) for x in rows + [header]) for i in range(len(header))]
    print(" | ".join(h.ljust(colw[i]) for i,h in enumerate(header)))
    print("-+-".join("-"*w for w in colw))
    for r in rows:
        print(" | ".join(str(r[i]).ljust(colw[i]) for i in range(len(header))))

    print("\n-- Equivalence groups (UTC, tol=1s, min overlap=1% or >=100 rows) --")
    if groups:
        for idx, g in enumerate(groups, 1):
            print(f"Group {idx}: {', '.join(g)}")
    else:
        print("(no equivalent groups found)")

    # Optionally show top mismatched pairs with high overlap but low match
    suspicious = []
    for (a,b), m in sim_matrix.items():
        if m.get("overlap", 0) >= 100 and m.get("match_ratio", 0.0) < 0.5:
            suspicious.append((a, b, m["overlap"], m["match_ratio"]))
    if suspicious:
        print("\n-- Highly overlapping but non-matching pairs (possible double-UTC or wrong unit) --")
        for a,b,ov,mr in sorted(suspicious, key=lambda x:(-x[2], x[3]))[:10]:
            print(f"{a} vs {b}: overlap={ov}, match_ratio={mr:.1%}")

    print("\nDone.\n")


def main():
    ap = argparse.ArgumentParser(description="Detect and group equivalent datetime columns in a CSV.")
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV path")
    ap.add_argument("--delimiter", default=None, help="CSV delimiter (auto-detect if not set)")
    ap.add_argument("--naive-tz", default=None, help="Time zone to apply to naive strings (e.g., America/Chicago). If not set, assume UTC for naive strings.")
    ap.add_argument("--encoding", default="utf-8", help="File encoding (default utf-8)")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit rows to read (for sampling/scale)")
    ap.add_argument("--preview", action="store_true", help="Write a small normalized preview CSV for the detected columns (first 50 rows).")
    args = ap.parse_args()

    csv_path = args.inp
    if not os.path.exists(csv_path):
        print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    # Read CSV (let pandas guess dtypes; we'll coerce as needed)
    try:
        df = pd.read_csv(csv_path, delimiter=args.delimiter, encoding=args.encoding, nrows=args.max_rows, low_memory=False)
    except Exception as e:
        print(f"ERROR: failed to read CSV: {e}", file=sys.stderr)
        sys.exit(2)

    parsed, reports = detect_date_columns(df, args.naive_tz)

    # Compute equivalence groups
    groups, sim = equivalence_groups(parsed)

    # JSON report
    json_report = {
        "source": os.path.abspath(csv_path),
        "rows_analyzed": int(len(df)),
        "detected_columns": {k: asdict(v) for k, v in reports.items()},
        "equivalence_groups": groups,
        "pairwise": {f"{a}||{b}": v for (a,b), v in sim.items()},
        "params": {"tolerance_seconds": 1, "min_overlap": 100, "min_match_ratio": 0.98},
    }
    json_path = os.path.splitext(csv_path)[0] + ".dt_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)

    # Optional normalized preview CSV
    preview_path = None
    if args.preview and parsed:
        prev = pd.DataFrame({k: v.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").str[:-4] + "Z" for k, v in parsed.items()}).head(50)
        preview_path = os.path.splitext(csv_path)[0] + ".dt_preview.csv"
        prev.to_csv(preview_path, index=False)

    # Print human-readable
    print_report(csv_path, reports, groups, sim, preview_path=preview_path)


if __name__ == "__main__":
    main()


def classify_string_formats(values_sample):
    """
    Classify likely date string formats from a sample of strings.
    Returns (format_guess, counts_dict, notes)
    """
    patterns = {
        "iso8601_z": re.compile(r"^\d{4}-\d{2}-\d{2}T.*Z$"),
        "iso8601_offset": re.compile(r"^\d{4}-\d{2}-\d{2}[ T].*[+-]\d{2}:\d{2}$"),
        "naive_iso": re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?$"),
        "rfc5322_http": re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s+\d{2}:\d{2}:\d{2}\s+\w+$", re.I),
        "iso_week": re.compile(r"^\d{4}-W\d{2}-\d(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?$"),
        "iso_ordinal": re.compile(r"^\d{4}-\d{3}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?$"),
        "text_month": re.compile(r"^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}(?:\s+\d{2}:\d{2}:\d{2})?$", re.I),
    }

    def slash_style(s):
        # Determine if looks like MDY or DMY
        m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})(?:\s+\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)?$", s)
        if not m: return None
        a, b, y = m.groups()
        a, b = int(a), int(b)
        # Heuristic: if both <=12, ambiguous; we'll count separately
        if 1 <= a <= 12 and 1 <= b <= 12:
            return "slash_ambiguous"
        return "slash_mdy" if a <= 12 and b > 12 else "slash_dmy"

    counts = {}
    notes = []

    for v in values_sample:
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s:
            continue

        # Slash styles first
        style = slash_style(s)
        if style:
            counts[style] = counts.get(style, 0) + 1
            continue

        matched = False
        for name, pat in patterns.items():
            if pat.search(s):
                counts[name] = counts.get(name, 0) + 1
                matched = True
                break
        if matched:
            continue

        # tz abbrev hint
        if re.search(r"\b([A-Z]{2,4})\b", s) and re.search(r"\d", s):
            counts["tz_abbrev_string"] = counts.get("tz_abbrev_string", 0) + 1
            continue

        # Fallback bucket
        counts["unclassified"] = counts.get("unclassified", 0) + 1

    # Resolve overall guess
    if not counts:
        return None, {}, notes

    # Ambiguity note for slash
    if counts.get("slash_mdy", 0) and counts.get("slash_dmy", 0):
        notes.append("mixed_mdy_dmy")
        guess = "ambiguous_slash"
    else:
        # Choose the label with highest count
        guess = max(counts.items(), key=lambda kv: kv[1])[0]

    return guess, counts, notes
