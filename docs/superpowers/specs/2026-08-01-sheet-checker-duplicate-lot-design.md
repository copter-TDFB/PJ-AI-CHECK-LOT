# Sheet Checker — Duplicate Lot Row Resolution — Design

**Date:** 2026-08-01
**Scope:** `utils/sheet_checker.py` only (no sheet/data changes, no other pipeline files)
**Source:** debugging session on `retail_sachet` image (Lot `109W`, Rootroute Wheatgrass Powder 5g) —
`/predict` reported `exp_match: false` even though OCR correctly read the expiry printed on the
sachet (`01/02/2027`).

## Problem

`SheetChecker._find_row_by_lot()` returns the **first** row in the sheet whose `Lot.` column
matches the OCR-extracted lot, then uses that single row for every downstream check
(`exp_match`, `product_match`, `sheet_product_name`).

Real-world sheets accumulate **duplicate rows for the same Lot**: a row is added when the Lot is
reserved (before the label is printed, `MFG`/`EXP` still blank), then a second row is added later
once the real dates are known — without deleting the first. Confirmed live example (sheet
`1kp_4uXIWnhDuCz1obMqzSOnKFBP9BFqMP-OqCAiTuIE`, gid `666410502`, "01" tab):

| Row | Lot | MFG | EXP | Product |
|---|---|---|---|---|
| 10 | `109W` | *(blank)* | *(blank)* | Rootroute Wheatgrass Powder 5 g |
| 12 | `109W` | `01/08/2026` | `01022027` | Rootroute Wheatgrass Powder 5 g |

Because row 10 comes first, `_find_row_by_lot` returns it, and `exp_match` compares the OCR's
`2027-02-01` against an empty string → `False`, even though row 12 (same Lot) agrees with the
photo exactly.

## Decisions (locked with user)

| Topic | Decision |
|---|---|
| Fix location | Code (`sheet_checker.py`), not a sheet/process change. Sheet cleanup is a separate, optional follow-up. |
| Row selection when Lot matches >1 row | Loop through **every** row matching the Lot (in sheet order); the first one whose normalized `EXP` equals the OCR-extracted `exp` wins. |
| Blank EXP vs. non-blank-but-wrong EXP | **Same rule, no special-casing.** A row that doesn't match — whether its `EXP` cell is empty or holds a different date — is skipped identically; the loop just keeps going to the next Lot-matching row. |
| Lot filter vs. EXP filter ordering | Lot equality is the **hard filter first** (unchanged from today — `lot_match` is decided purely by "does any row have this Lot", before any EXP comparison). EXP is only ever used to pick *among* rows that already passed the Lot filter — never to search the sheet independently of Lot. |
| No candidate row matches the OCR'd EXP | Fall back to the **first** Lot-matching row (today's behavior) and report `exp_match = False`. This also covers the case where two Lot-duplicate rows have two different real (non-blank) EXP values and OCR's reading doesn't equal either — conservative default, surfaces as a genuine mismatch for a human to check. |
| Visibility | Log one `logger.warning` when more than one row matches a Lot (regardless of which sub-case), so sheet hygiene issues stay visible without needing a new alerting mechanism. |
| `product`/`sachet` checks | Unaffected in behavior — they simply read from whichever row was resolved as above (today they read from the single found row). |
| `lot_short_fallback` retry path | Same duplicate-resolution logic applies to the short-lot retry (reuses the same helper). |

## Design

### `_find_row_by_lot` → `_find_rows_by_lot`

Rename and change return type from `dict | None` to `list[dict]` (possibly empty). Same
case-insensitive comparison against the `Lot.` column as today; only difference is it collects
**all** matches instead of returning on the first hit.

```python
def _find_rows_by_lot(self, lot: str | None, sheet_id: str, gid: int) -> list[dict]:
    if not lot:
        return []
    lot_key = lot.upper().strip()
    return [
        row for row in self._get_rows(sheet_id, gid)
        if str(row.get("Lot.", "")).upper().strip() == lot_key
    ]
```

### `check()` — new row-resolution step

Replace the single `row = self._find_row_by_lot(...)` call with:

```python
rows = self._find_rows_by_lot(lot, sheet_id, gid)

if not rows and config.lot_short_fallback and lot and len(lot) >= 13:
    short_lot = lot[9:13]
    logger.info("%s lot not found — retrying with short lot %s", config.key, short_lot)
    rows = self._find_rows_by_lot(short_lot, sheet_id, gid)

if "lot" in checks:
    result["lot_match"] = bool(rows)

row = None
if rows:
    if len(rows) > 1:
        logger.warning(
            "%s: %d rows matched Lot %s — resolving by EXP",
            config.key, len(rows), lot,
        )
    if "exp" in checks:
        exp_key = "exp_box" if is_container else "exp_date"
        exp = kwargs.get(exp_key)
        if exp:
            row = next(
                (r for r in rows if _normalize_sheet_date(r.get("EXP", "")) == exp),
                None,
            )
    row = row or rows[0]  # fallback: first row (today's behavior) when no candidate matched,
                           # or when "exp" isn't checked / OCR didn't read one
```

The rest of `check()` (`sheet_product_name`, `exp_match`, `product_match`, the `row is None`
branch, and the whole `sachet` block) is **unchanged** — it already operates on a single `row`
variable; only how that variable gets populated changes.

### Worked examples (from the decisions table)

- **1 row matches Lot** → `rows = [that row]`, loop is a no-op, identical to today.
- **2 rows, one blank EXP one real** → loop skips the blank one (doesn't equal OCR's exp),
  matches the real one → `exp_match = True`, correct product/exp both come from the real row.
- **2 rows, both non-blank but different EXP values** (Lot genuinely reused across two batches)
  → whichever row's EXP equals what OCR read wins; if OCR's reading matches neither, falls back
  to the first row and reports `exp_match = False`.
- **0 rows match** → unchanged: `lot_match = False`, `exp_match`/`product_match = False` if
  checked.

## Out of scope (explicit)

- Cleaning up the actual duplicate rows in the live Google Sheet (separate, optional follow-up —
  not required for this fix to work).
- Any change to `product`/`sachet` matching logic beyond "read from the resolved row."
- Deduping or flagging duplicates anywhere other than a log line (no new UI/alert surface).
- Any change to `_normalize_sheet_date`, `PackagingConfig`, or the `/predict` response shape.

## Verification

New `tests/test_sheet_checker.py` (no existing coverage for this module):

- Mock `_get_rows` to return 2 rows for the same Lot (one blank `EXP`, one filled) → `check()`
  resolves to the filled row (`exp_match=True`, correct `sheet_product_name`).
- Mock 2 rows with **different non-blank** `EXP` values → `check()` picks whichever matches the
  OCR `exp` kwarg; a third case where OCR's `exp` matches neither → falls back to first row,
  `exp_match=False`.
- Single-row match (no duplicates) → behavior identical to current tests-would-be (regression
  guard).
- Zero rows match → `lot_match=False`, `exp_match=False` (if checked), unchanged from today.
- `lot_short_fallback` retry path still finds rows via the short lot when the full lot has no
  match.

Manual: re-run `scripts/run_real_pipeline.py` on the `IMG_20260801_103346.jpg` sachet image
against the real sheet/gid used in this session, confirm `exp_match` is now `True` via a direct
`SheetChecker().check(...)` call (already reproduced as `False` before this fix, in-session).

## Risks

- **Silent wrong-batch match**: if OCR misreads an expiry and that misread value coincidentally
  equals a *different* Lot-duplicate row's real EXP, the wrong row would be picked. This is an
  inherent ambiguity in the sheet itself (same Lot reused for two batches) that no code-level
  heuristic can fully resolve; the conservative fallback (first row, reported mismatch) is the
  overall safety net for the case where OCR matches nobody.
- **Performance**: `_find_rows_by_lot` is a Python list comprehension over cached rows already
  held in memory (`_get_rows`, 5-minute TTL) — no new sheet reads, cost is negligible even for
  large sheets.
