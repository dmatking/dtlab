#!/usr/bin/env python3
"""
dt_equivalence.py

Detects date-like columns in a CSV, infers their types/formats, normalizes them to UTC,
and groups columns that represent the same instant (equivalent) across rows.

Usage:
  python dt_equivalence.py --in data.csv
                           [--delimiter ,]
                           [--naive-tz America/Chicago]
                           [--tolerance 1]
                           [--min-overlap 100]
                           [--min-match-ratio 0.98]
                           [--include-columns col1,col2]
                           [--exclude-columns col3,col4]
                           [--max-rows 1000000]
                           [--preview]

Outputs:
  - Prints a human-readable report to stdout
  - Writes a JSON report next to the CSV: <csvname>.dt_report.json
  - Optionally writes a normalized preview CSV: <csvname>.dt_preview.csv
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict, field
from datetime import timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dateutil import parser as du_parser

# ---------- Heuristics & helpers ----------

CANDIDATE_NAME_HINTS = [
    "time", "date", "timestamp", "ts", "_at", "created", "updated", "modified", "event", "logged"
]

TZ_ABBR_MAP = {
    "UTC": "UTC", "Z": "UTC", "GMT": "UTC",
    "PST": "America/Los_Angeles", "PDT": "America/Los_Angeles",
    "MST": "America/Denver",      "MDT": "America/Denver",
    "CST": "America/Chicago",     "CDT": "America/Chicago",
    "EST": "America/New_York",    "EDT": "America/New_York",
    # Add more if your data uses them (AEST/AEDT, IST, etc.)
}


def detect_epoch_unit(series: pd.Series) -> Tuple[Optional[str], List[str]]:
    """Guess the epoch unit for a numeric-like series by magnitude. Returns (unit, notes)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None, ["no numeric values"]
    mn, mx = s.min(), s.max()

    def plausible_year_range(epoch_values: pd.Series, unit: str) -> bool:
        divisors = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
        vals = epoch_values / divisors[unit]
        years = pd.to_datetime(vals, unit="s", utc=True, errors="coerce").dt.year
        if years.isna().all():
            return False
        valid = years[years.between(1970, 2100)]
        if len(valid) / len(years) <= 0.95:
            return False
        # Reject if nearly all dates cluster at the epoch origin (year 1970) —
        # a strong signal that we're looking at small integer IDs, not timestamps.
        if len(valid) > 0 and (valid == 1970).mean() > 0.9:
            return False
        return True

    brackets = [
        ("ns", 1e17, 1e19),
        ("us", 1e14, 1e16),
        ("ms", 1e11, 1e13),
        ("s",  1e8,  1e10),
    ]
    for unit, lo, hi in brackets:
        if mn > lo and mx < hi:
            if plausible_year_range(s, unit):
                return unit, []
            return None, [
                f"magnitude suggests {unit} but year-range check failed "
                f"(min={mn:.2e}, max={mx:.2e})"
            ]

    # Outside all normal brackets — try fallback by plausibility
    for unit in ("s", "ms", "us", "ns"):
        if plausible_year_range(s, unit):
            return unit, [
                f"magnitude {mn:.2e}–{mx:.2e} outside normal brackets; "
                f"inferred {unit} by year-range fallback"
            ]

    return None, [f"could not determine epoch unit (magnitude {mn:.2e}–{mx:.2e})"]


def classify_string_formats(
    values_sample: List[str],
) -> Tuple[Optional[str], Dict[str, int], List[str]]:
    """
    Classify likely date string formats from a sample of strings.
    Returns (format_guess, counts_dict, notes).
    """
    patterns = {
        "iso8601_z":      re.compile(r"^\d{4}-\d{2}-\d{2}T.*Z$"),
        "iso8601_offset": re.compile(r"^\d{4}-\d{2}-\d{2}[ T].*[+-]\d{2}:\d{2}$"),
        "naive_iso":      re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?$"),
        "rfc5322":        re.compile(
            r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\d{1,2}\s+"
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s+"
            r"\d{2}:\d{2}:\d{2}\s+\S+$",
            re.I,
        ),
        "iso_week":    re.compile(r"^\d{4}-W\d{2}-\d(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?$"),
        "iso_ordinal": re.compile(r"^\d{4}-\d{3}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?$"),
        "text_month":  re.compile(
            r"^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}"
            r"(?:\s+\d{2}:\d{2}:\d{2})?$",
            re.I,
        ),
    }

    def slash_style(s: str) -> Optional[str]:
        m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})(?:\s+\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)?$", s)
        if not m:
            return None
        a, b = int(m.group(1)), int(m.group(2))
        if 1 <= a <= 12 and 1 <= b <= 12:
            return "slash_ambiguous"
        return "slash_mdy" if a <= 12 else "slash_dmy"

    counts: Dict[str, int] = {}
    for v in values_sample:
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s:
            continue
        style = slash_style(s)
        if style:
            counts[style] = counts.get(style, 0) + 1
            continue
        for name, pat in patterns.items():
            if pat.match(s):
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            if re.search(r"\b([A-Z]{2,4})\b", s) and re.search(r"\d", s):
                counts["tz_abbrev_string"] = counts.get("tz_abbrev_string", 0) + 1
            else:
                counts["unclassified"] = counts.get("unclassified", 0) + 1

    if not counts:
        return None, {}, []

    notes: List[str] = []
    if counts.get("slash_mdy", 0) and counts.get("slash_dmy", 0):
        notes.append("mixed_mdy_dmy")
        guess: Optional[str] = "ambiguous_slash"
    elif counts.get("slash_ambiguous", 0) and not counts.get("slash_mdy") and not counts.get("slash_dmy"):
        notes.append("ambiguous_slash_format")
        guess = "slash_ambiguous"
    else:
        guess = max(counts.items(), key=lambda kv: kv[1])[0]

    return guess, counts, notes


def looks_like_date_string(s: str) -> bool:
    """Quick check to avoid attempting to parse every random string."""
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s:
        return False
    if any(ch in s for ch in "-/:T+") and re.search(r"\d", s):
        return True
    if re.search(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun),", s, re.I):
        return True
    if re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", s, re.I):
        return True
    return False


def parse_string_to_utc(series: pd.Series, naive_tz: Optional[str]) -> Tuple[pd.Series, Dict]:
    """
    Parse a string series to UTC datetimes.

    Fast path (pandas.to_datetime) used when >= 80% of values parse:
      - Without naive_tz: parse with utc=True (naive values treated as UTC).
      - With naive_tz: parse without forcing UTC, then localize naive values to
        naive_tz. Mixed-tz columns fall through to row-wise fallback.

    Fallback: row-wise dateutil parsing with TZ abbreviation substitution.
    """
    meta: Dict = {"parser": None, "naive_policy": None, "notes": []}

    # ---- Fast path ----
    try:
        if naive_tz is None:
            dt = pd.to_datetime(series, errors="coerce", utc=True)
            if dt.notna().mean() > 0.8:
                meta["parser"] = "pandas.to_datetime(utc=True)"
                meta["naive_policy"] = "naive→UTC"
                return dt, meta
        else:
            # Parse without assuming UTC so we can correctly handle naive values
            dt_raw = pd.to_datetime(series, errors="coerce")
            if dt_raw.notna().mean() > 0.8:
                try:
                    if dt_raw.dt.tz is None:
                        # Entire series is naive — localize to naive_tz
                        naive_count = int(dt_raw.notna().sum())
                        dt_utc = dt_raw.dt.tz_localize(naive_tz).dt.tz_convert("UTC")
                        meta["parser"] = "pandas.to_datetime + tz_localize"
                        meta["naive_policy"] = f"naive→{naive_tz}"
                        meta["notes"].append(f"naive_rows={naive_count}")
                        return dt_utc, meta
                    else:
                        # Entire series is tz-aware — naive_tz unused
                        dt_utc = dt_raw.dt.tz_convert("UTC")
                        meta["parser"] = "pandas.to_datetime + tz_convert"
                        meta["naive_policy"] = "all tz-aware (naive_tz unused)"
                        return dt_utc, meta
                except AttributeError:
                    # Mixed tz (object dtype) — fall through to row-wise
                    meta["notes"].append("mixed tz detected in fast path; using row-wise fallback")
    except Exception as e:
        meta["notes"].append(f"fast_path_error: {e!r}")

    # ---- Row-wise fallback ----
    out = []
    naive_count = 0
    for val in series.astype(str).tolist():
        val = val.strip()
        if not val or not looks_like_date_string(val):
            out.append(pd.NaT)
            continue
        try:
            tokens = val.split()
            tokens = [TZ_ABBR_MAP.get(tok, tok) for tok in tokens]
            val_norm = " ".join(tokens)
            dt = du_parser.parse(val_norm, tzinfos=TZ_ABBR_MAP)
            if dt.tzinfo is None:
                naive_count += 1
                dt = dt.replace(tzinfo=ZoneInfo(naive_tz or "UTC"))
            out.append(dt.astimezone(timezone.utc))
        except (ValueError, OverflowError, AttributeError, TypeError):
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


def detect_date_columns(
    df: pd.DataFrame,
    naive_tz: Optional[str],
    include_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
) -> Tuple[Dict[str, pd.Series], Dict[str, ColumnReport]]:
    parsed_utc: Dict[str, pd.Series] = {}
    reports: Dict[str, ColumnReport] = {}

    columns = list(df.columns)
    if include_columns:
        columns = [c for c in columns if c in include_columns]
    if exclude_columns:
        exclude_set = set(exclude_columns)
        columns = [c for c in columns if c not in exclude_set]

    for col in columns:
        s = df[col]
        name_l = col.lower()
        name_hint = any(hint in name_l for hint in CANDIDATE_NAME_HINTS)
        is_numeric_like = pd.to_numeric(s, errors="coerce").notna().mean() > 0.8
        notes: List[str] = []

        if is_numeric_like:
            detected_unit, epoch_notes = detect_epoch_unit(s)
            notes.extend(epoch_notes)
            if detected_unit:
                utc_series, meta = parse_epoch_to_utc(s, detected_unit)
                parse_rate = float(utc_series.notna().mean())
                reports[col] = ColumnReport(
                    name=col, role="epoch", parse_rate=parse_rate,
                    detected_unit=detected_unit, parser=meta["parser"],
                    format_guess=f"epoch:{detected_unit}", notes=notes,
                )
                parsed_utc[col] = utc_series
            else:
                if name_hint:
                    notes.append("name suggests datetime but epoch unit could not be determined")
                reports[col] = ColumnReport(name=col, role="non_date", parse_rate=0.0, notes=notes)
            continue  # numeric columns skip string detection

        # Check sample for date-ish strings
        sample = s.dropna().astype(str).head(200).tolist()
        stringy_rate = sum(looks_like_date_string(v) for v in sample) / len(sample) if sample else 0.0

        if name_hint or stringy_rate > 0.3:
            utc_series, meta = parse_string_to_utc(s.astype(str), naive_tz)
            parse_rate = float(utc_series.notna().mean())
            notes.extend(meta.get("notes", []))
            fmt_guess, _, fmt_notes = classify_string_formats([v for v in sample if v][:50])
            notes.extend(fmt_notes)
            reports[col] = ColumnReport(
                name=col, role="date_string", parse_rate=parse_rate,
                parser=meta["parser"], naive_policy=meta.get("naive_policy"),
                format_guess=fmt_guess or "", notes=notes,
            )
            parsed_utc[col] = utc_series
        else:
            reports[col] = ColumnReport(name=col, role="non_date", parse_rate=0.0, notes=notes)

    return parsed_utc, reports


def equivalence_groups(
    parsed: Dict[str, pd.Series],
    tolerance_seconds: int = 1,
    min_overlap: int = 100,
    min_match_ratio: float = 0.98,
) -> Tuple[List[List[str]], Dict[Tuple[str, str], Dict]]:
    """
    Build groups of columns that represent the same instant after normalization.
    Two columns are equivalent if:
      - Row overlap (both non-null) >= max(min_overlap, 1% of total rows)
      - Fraction of overlapping rows where |A-B| <= tolerance_seconds >= min_match_ratio
    """
    cols = list(parsed.keys())
    n = len(cols)
    if n == 0:
        return [], {}

    total_rows = max(len(v) for v in parsed.values())
    # Boost min_overlap for large files (1% floor), but cap it so it's
    # always achievable — never more than half the file size.
    dyn_min_overlap = max(min_overlap, int(0.01 * total_rows))
    dyn_min_overlap = min(dyn_min_overlap, max(5, total_rows // 2))

    # Convert to float seconds; NaT -> NaN.
    # Use total_seconds() from epoch to avoid resolution assumptions
    # (pandas 3.0+ can return datetime64[us] or [ms] rather than [ns]).
    _epoch = pd.Timestamp("1970-01-01", tz="UTC")
    as_int: Dict[str, np.ndarray] = {}
    for c in cols:
        s = parsed[c]
        si = (s - _epoch).dt.total_seconds().where(s.notna())
        as_int[c] = si.values

    # Union-Find with path compression and union-by-rank
    parent = {c: c for c in cols}
    rank: Dict[str, int] = {c: 0 for c in cols}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    sim: Dict[Tuple[str, str], Dict] = {}
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
            match_ratio = float((diff <= tolerance_seconds).sum()) / overlap
            eq = match_ratio >= min_match_ratio
            sim[(a, b)] = {"overlap": overlap, "match_ratio": match_ratio, "equivalent": bool(eq)}
            if eq:
                union(a, b)

    groups_map: Dict[str, List[str]] = {}
    for c in cols:
        groups_map.setdefault(find(c), []).append(c)
    return list(groups_map.values()), sim


class AnalysisResult:
    """
    Result of analyze(). Designed for use in notebooks and scripts.

    Attributes:
      reports  — dict of column name → ColumnReport (all analyzed columns)
      groups   — list of equivalence groups (each a list of column names);
                 singletons are groups of length 1
      sim      — pairwise similarity matrix: (col_a, col_b) → {overlap, match_ratio, equivalent}
      params   — the analysis parameters used
      parsed   — dict of column name → UTC-normalized pd.Series (only detected datetime columns)

    Methods:
      equivalent_groups() → groups with 2+ members
      summary()           → pd.DataFrame of column metadata (role, format, parse_rate, …)
      normalized()        → pd.DataFrame of detected columns normalized to UTC strings
      report(source)      → print the human-readable report
    """

    def __init__(
        self,
        reports: Dict[str, "ColumnReport"],
        groups: List[List[str]],
        sim: Dict[Tuple[str, str], Dict],
        params: Dict,
        parsed: Dict[str, pd.Series],
    ) -> None:
        self.reports = reports
        self.groups = groups
        self.sim = sim
        self.params = params
        self.parsed = parsed

    def equivalent_groups(self) -> List[List[str]]:
        """Return only groups with 2 or more columns."""
        return [g for g in self.groups if len(g) > 1]

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame of detected column metadata — handy for notebook display."""
        return pd.DataFrame([
            {
                "column":       rep.name,
                "role":         rep.role,
                "format":       rep.format_guess or "",
                "unit":         rep.detected_unit or "",
                "parse_rate":   rep.parse_rate,
                "parser":       rep.parser or "",
                "naive_policy": rep.naive_policy or "",
                "notes":        "; ".join(rep.notes),
            }
            for rep in self.reports.values()
        ])

    def normalized(self) -> pd.DataFrame:
        """Return a DataFrame of all detected datetime columns normalized to UTC ISO strings."""
        if not self.parsed:
            return pd.DataFrame()
        return pd.DataFrame({
            k: v.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            for k, v in self.parsed.items()
        })

    def report(self, source: str = "<DataFrame>") -> None:
        """Print the human-readable equivalence report."""
        _print_report(source, self.reports, self.groups, self.sim, self.params)


def analyze(
    df: pd.DataFrame,
    naive_tz: Optional[str] = None,
    tolerance_seconds: int = 1,
    min_overlap: int = 100,
    min_match_ratio: float = 0.98,
    include_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
) -> AnalysisResult:
    """
    Analyze a DataFrame for equivalent datetime columns.

    Parameters:
      df               — input DataFrame
      naive_tz         — IANA timezone to apply to naive datetime strings
                         (e.g. "America/Chicago"). Default: treat as UTC.
      tolerance_seconds — max difference in seconds to consider two timestamps equivalent
      min_overlap      — minimum row overlap required to compare two columns
      min_match_ratio  — fraction of overlapping rows that must match within tolerance
      include_columns  — if set, only analyze these columns
      exclude_columns  — columns to skip

    Returns an AnalysisResult.

    Example (notebook)::

        import pandas as pd
        from dt_equivalence import analyze

        df = pd.read_parquet("events.parquet")
        result = analyze(df, naive_tz="America/Chicago")
        result.report()
        display(result.summary())
        display(result.normalized().head())
        print(result.equivalent_groups())
    """
    params = {
        "tolerance_seconds": tolerance_seconds,
        "min_overlap": min_overlap,
        "min_match_ratio": min_match_ratio,
    }
    parsed, reports = detect_date_columns(df, naive_tz, include_columns, exclude_columns)
    groups, sim = equivalence_groups(parsed, **params)
    return AnalysisResult(reports=reports, groups=groups, sim=sim, params=params, parsed=parsed)


def _print_report(
    source: str,
    reports: Dict[str, ColumnReport],
    groups: List[List[str]],
    sim_matrix: Dict[Tuple[str, str], Dict],
    params: Dict,
    preview_path: Optional[str] = None,
) -> None:
    print("\n=== dt-equivalence report ===")
    print(f"Source: {source}")
    if preview_path:
        print(f"Normalized preview (first 50 rows): {preview_path}")

    print("\n-- Detected columns --")
    header = ["column", "role", "format", "unit", "parse_rate", "parser", "naive_policy", "notes"]
    rows = [
        [
            rep.name,
            rep.role,
            rep.format_guess or "",
            rep.detected_unit or "",
            f"{rep.parse_rate:.1%}",
            rep.parser or "",
            rep.naive_policy or "",
            "; ".join(rep.notes) if rep.notes else "",
        ]
        for rep in reports.values()
    ]
    if rows:
        colw = [max(len(str(x[i])) for x in rows + [header]) for i in range(len(header))]
    else:
        colw = [len(h) for h in header]
    print(" | ".join(h.ljust(colw[i]) for i, h in enumerate(header)))
    print("-+-".join("-" * w for w in colw))
    for r in rows:
        print(" | ".join(str(r[i]).ljust(colw[i]) for i in range(len(header))))

    tol = params["tolerance_seconds"]
    min_ov = params["min_overlap"]
    mmr = params["min_match_ratio"]
    print(f"\n-- Equivalence groups (tol={tol}s, min_overlap={min_ov}, min_match_ratio={mmr:.0%}) --")
    date_groups = [g for g in groups if len(g) > 1]
    singletons = [g[0] for g in groups if len(g) == 1]
    if date_groups:
        for idx, g in enumerate(date_groups, 1):
            print(f"  Group {idx}: {', '.join(g)}")
    else:
        print("  (no equivalent groups found)")
    if singletons:
        print(f"  Singletons (no match): {', '.join(singletons)}")

    suspicious = [
        (a, b, m["overlap"], m["match_ratio"])
        for (a, b), m in sim_matrix.items()
        if m.get("overlap", 0) >= min_ov and 0 < m.get("match_ratio", 0.0) < 0.5
    ]
    if suspicious:
        print("\n-- Highly overlapping but non-matching pairs (possible wrong unit or interpretation) --")
        for a, b, ov, mr in sorted(suspicious, key=lambda x: (-x[2], x[3]))[:10]:
            print(f"  {a} vs {b}: overlap={ov}, match_ratio={mr:.1%}")

    print("\nDone.\n")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Detect and group equivalent datetime columns in a CSV.")
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV path")
    ap.add_argument("--delimiter", default=None, help="CSV delimiter (auto-detect if not set)")
    ap.add_argument("--naive-tz", default=None,
                    help="Timezone for naive datetime strings (e.g. America/Chicago). Default: treat as UTC.")
    ap.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit rows read (for sampling)")
    ap.add_argument("--tolerance", type=int, default=1, metavar="SECONDS",
                    help="Max seconds difference to consider two timestamps equivalent (default: 1)")
    ap.add_argument("--min-overlap", type=int, default=100,
                    help="Minimum row overlap to compare two columns (default: 100)")
    ap.add_argument("--min-match-ratio", type=float, default=0.98,
                    help="Fraction of overlapping rows that must match within tolerance (default: 0.98)")
    ap.add_argument("--include-columns", default=None,
                    help="Comma-separated column names to analyze (default: all)")
    ap.add_argument("--exclude-columns", default=None,
                    help="Comma-separated column names to skip")
    ap.add_argument("--preview", action="store_true",
                    help="Write a normalized preview CSV (first 50 rows) alongside the report.")
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        print(f"ERROR: file not found: {args.inp}", file=sys.stderr)
        sys.exit(2)

    try:
        df = pd.read_csv(
            args.inp, delimiter=args.delimiter, encoding=args.encoding,
            nrows=args.max_rows, low_memory=False,
        )
    except Exception as e:
        print(f"ERROR: failed to read CSV: {e}", file=sys.stderr)
        sys.exit(2)

    include_cols = [c.strip() for c in args.include_columns.split(",")] if args.include_columns else None
    exclude_cols = [c.strip() for c in args.exclude_columns.split(",")] if args.exclude_columns else None

    result = analyze(
        df,
        naive_tz=args.naive_tz,
        tolerance_seconds=args.tolerance,
        min_overlap=args.min_overlap,
        min_match_ratio=args.min_match_ratio,
        include_columns=include_cols,
        exclude_columns=exclude_cols,
    )

    json_path = os.path.splitext(args.inp)[0] + ".dt_report.json"
    json_report = {
        "source": os.path.abspath(args.inp),
        "rows_analyzed": int(len(df)),
        "params": result.params,
        "detected_columns": {k: asdict(v) for k, v in result.reports.items()},
        "equivalence_groups": result.groups,
        "pairwise": {f"{a}||{b}": v for (a, b), v in result.sim.items()},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)

    preview_path = None
    if args.preview and result.parsed:
        result.normalized().head(50).to_csv(
            os.path.splitext(args.inp)[0] + ".dt_preview.csv", index=False
        )
        preview_path = os.path.splitext(args.inp)[0] + ".dt_preview.csv"

    _print_report(args.inp, result.reports, result.groups, result.sim, result.params, preview_path=preview_path)


if __name__ == "__main__":
    main()
