# dtlab — datetime equivalence detector

Finds datetime columns in a CSV or DataFrame that represent the same instant in different formats or timezones, and groups them together.

Useful when working with wide tables that have many timestamp columns — epoch seconds alongside ISO strings alongside RFC5322 headers, all meaning the same thing.

## What it detects

- ISO 8601 (with Z, numeric offsets, naive)
- RFC 5322 / HTTP-date
- Epoch integers (seconds, milliseconds, microseconds, nanoseconds — inferred by magnitude)
- Slash-style dates (MDY/DMY, with ambiguity flagging)
- ISO week / ordinal formats
- TZ abbreviations (PST/PDT, CST/CDT, EST/EDT, MST/MDT, UTC/GMT)

## Install

```bash
pip install pandas numpy python-dateutil
```

Requires Python 3.9+ (uses `zoneinfo`).

## CLI

```bash
python dt_equivalence.py --in data.csv
```

```
=== dt-equivalence report ===
Source: data.csv

-- Detected columns --
column         | role        | format         | unit | parse_rate | parser                       | naive_policy | notes
...

-- Equivalence groups (tol=1s, min_overlap=100, min_match_ratio=98%) --
  Group 1: ts_iso_utc, ts_iso_cdt, ts_iso_pdt, ts_epoch_s, ts_epoch_ms, ts_rfc5322
  Singletons (no match): ts_naive_local
```

Also writes `data.dt_report.json` with full pairwise details.

### Options

| Flag | Default | Description |
|---|---|---|
| `--in` | required | Input CSV path |
| `--delimiter` | auto | CSV delimiter |
| `--naive-tz` | UTC | IANA timezone for naive datetime strings (e.g. `America/Chicago`) |
| `--encoding` | utf-8 | File encoding |
| `--max-rows` | all | Limit rows read |
| `--tolerance` | 1 | Max seconds difference to consider two timestamps equivalent |
| `--min-overlap` | 100 | Minimum non-null row overlap required to compare two columns |
| `--min-match-ratio` | 0.98 | Fraction of overlapping rows that must match within tolerance |
| `--include-columns` | all | Comma-separated list of columns to analyze |
| `--exclude-columns` | none | Comma-separated list of columns to skip |
| `--preview` | off | Write a normalized UTC preview CSV (first 50 rows) |

## Notebook / script API

```python
import pandas as pd
from dt_equivalence import analyze

df = pd.read_parquet("events.parquet")
result = analyze(df, naive_tz="America/Chicago")

result.report()              # print text report
result.summary()             # pd.DataFrame of column metadata
result.normalized()          # pd.DataFrame of detected columns as UTC ISO strings
result.equivalent_groups()   # list of groups with 2+ members
result.groups                # all groups including singletons
result.parsed                # dict of col → UTC pd.Series
result.sim                   # pairwise {overlap, match_ratio, equivalent}
```

`analyze()` accepts the same parameters as the CLI flags:

```python
result = analyze(
    df,
    naive_tz="America/New_York",
    tolerance_seconds=5,
    min_overlap=50,
    min_match_ratio=0.95,
    include_columns=["created_at", "event_ts", "ts_epoch"],
    exclude_columns=["id"],
)
```

## How equivalence works

All detected columns are normalized to UTC. Two columns are considered equivalent if:

1. They share at least `min_overlap` non-null rows (or 1% of total rows for large files)
2. At least `min_match_ratio` of those rows have timestamps within `tolerance_seconds` of each other

Grouping uses union-find, so transitivity is handled correctly (if A≡B and B≡C, all three end up in the same group).

## Split datetime + timezone column pairs

A common pattern in messy tables is a naive datetime column paired with a separate timezone column:

| passing_dt          | passing_tz         |
|---------------------|--------------------|
| 2025-03-09 07:00:00 | America/Chicago    |
| 2025-03-09 08:15:00 | America/New_York   |

dtlab detects these pairs automatically by name convention. It looks for columns ending in `_tz` / `_timezone` / `_zone` whose values look like IANA zone names or TZ abbreviations, then searches for a companion column with the same prefix and a datetime suffix (`_dt`, `_date`, `_time`, `_ts`, `_at`).

When a pair is found, the two columns are combined row-wise (each row gets its own timezone applied) before UTC normalization. The timezone column appears in the report with `role=tz_column` and is excluded from equivalence grouping.

If the timezone column has null values for some rows, those rows fall back to `--naive-tz` (or UTC if not set).

## Caveats

- **Naive timestamps**: without `--naive-tz`, naive strings are assumed to be UTC. If your data has naive local times, set `--naive-tz` to get correct grouping.
- **Ambiguous slash dates**: `03/08/2025` is ambiguous (MDY vs DMY). These are flagged in the notes column but still parsed by pandas using its default interpretation.
- **Floating-point epoch loss**: epoch values stored as floats with few significant digits may not match string timestamps exactly — raise `--tolerance` if needed.
