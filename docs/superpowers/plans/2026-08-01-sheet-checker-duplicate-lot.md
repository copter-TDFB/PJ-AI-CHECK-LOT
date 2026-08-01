# Sheet Checker Duplicate-Lot Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `SheetChecker.check()` reporting a false `exp_match: False` when a Lot number
appears in more than one row of the Google Sheet (e.g. a blank placeholder row added before
printing, plus a later row with the real MFG/EXP).

**Architecture:** `SheetChecker._find_row_by_lot` (returns the first row matching a Lot) becomes
`_find_rows_by_lot` (returns every row matching that Lot). `check()` then picks, among those
candidate rows, the one whose `EXP` equals what OCR actually read; if none match, it falls back
to the first row exactly like today. Lot equality is always the hard filter first — EXP is only
ever used to choose among rows that already matched the Lot, never to search independently.

**Tech Stack:** Python 3.11, pytest, `gspread` (already a dependency, no new libraries).

## Global Constraints

- Fix is confined to `utils/sheet_checker.py` (plus the one existing test that pokes its
  internals) — no sheet/data changes, no other pipeline file touched.
- No special-casing between "blank EXP" and "non-blank but different EXP" — both are just rows
  that don't match the OCR value and get skipped the same way.
- `lot_match` semantics are unchanged: `True` iff at least one row matches the Lot.
- `print()` is forbidden in this repo — use `logger` (already imported in this file).
- Run tests with `python -m pytest` (`pytest` is not on PATH in this repo).

---

### Task 1: Duplicate-lot resolution in `SheetChecker`

**Files:**
- Modify: `utils/sheet_checker.py:71-79` (rename `_find_row_by_lot` → `_find_rows_by_lot`)
- Modify: `utils/sheet_checker.py:109-143` (row-resolution logic inside `check()`)
- Modify: `tests/test_routing.py:53-69` (`test_sheet_checker_lot_source` pokes the renamed method)
- Create: `tests/test_sheet_checker.py`

**Interfaces:**
- Produces: `SheetChecker._find_rows_by_lot(self, lot: str | None, sheet_id: str, gid: int) -> list[dict]`
  — replaces `_find_row_by_lot` (which returned `dict | None`). Returns `[]` when `lot` is falsy
  or nothing matches; returns every row (in sheet order) whose `Lot.` column equals `lot`
  (case-insensitive, stripped) otherwise.
- Consumes: `SheetChecker._get_rows` (unchanged, already exists at `utils/sheet_checker.py:41`),
  `_normalize_sheet_date` module function (unchanged, already exists at
  `utils/sheet_checker.py:19`).

- [ ] **Step 1: Write the failing tests in a new file `tests/test_sheet_checker.py`**

```python
from pipeline.packaging_registry import PackagingConfig
from utils.sheet_checker import SheetChecker


def _cfg(sheet_checks=("lot", "exp"), lot_short_fallback=False):
    return PackagingConfig(
        key="retail_sachet",
        display_name="Retail Sachet",
        pipeline="detector_ocr",
        lot_patterns=[],
        fields_extracted=["lot", "exp"],
        sheet_checks=list(sheet_checks),
        post_ocr_fixes=[],
        message_template_key="lot_exp",
        model_classifier_label="retail_sachet",
        detector_yolo_prefixes=["retail_sachet_lot"],
        conf_threshold=0.6,
        accuracy=None,
        gate_on_lot=True,
        lot_short_fallback=lot_short_fallback,
        sub_regions=[],
        detection_mode="single",
    )


def _row(lot, mfg="", exp="", product="Rootroute Wheatgrass Powder 5 g"):
    return {"Lot.": lot, "MFG": mfg, "EXP": exp, "Product Name": product}


def test_single_matching_row_behaves_like_before(monkeypatch):
    checker = SheetChecker()
    monkeypatch.setattr(
        checker, "_find_rows_by_lot",
        lambda lot, sheet_id, gid: [_row("109W", exp="01022027")],
    )
    result = checker.check(
        _cfg(), sheet_id="sid", gid=0,
        lot_number="109W", exp_date="2027-02-01",
    )
    assert result["lot_match"] is True
    assert result["exp_match"] is True
    assert result["sheet_product_name"] == "Rootroute Wheatgrass Powder 5 g"


def test_duplicate_rows_blank_placeholder_then_filled_resolves_to_filled(monkeypatch):
    checker = SheetChecker()
    rows = [
        _row("109W", exp=""),  # placeholder row added before the label was printed
        _row("109W", mfg="01/08/2026", exp="01022027"),  # real data added later
    ]
    monkeypatch.setattr(checker, "_find_rows_by_lot", lambda lot, sheet_id, gid: rows)
    result = checker.check(
        _cfg(), sheet_id="sid", gid=0,
        lot_number="109W", exp_date="2027-02-01",
    )
    assert result["lot_match"] is True
    assert result["exp_match"] is True


def test_duplicate_rows_conflicting_exp_picks_whichever_matches_ocr(monkeypatch):
    checker = SheetChecker()
    rows = [
        _row("109W", exp="01/02/2027"),
        _row("109W", exp="05/03/2027"),
    ]
    monkeypatch.setattr(checker, "_find_rows_by_lot", lambda lot, sheet_id, gid: rows)

    result_first = checker.check(
        _cfg(), sheet_id="sid", gid=0, lot_number="109W", exp_date="2027-02-01"
    )
    assert result_first["exp_match"] is True

    result_second = checker.check(
        _cfg(), sheet_id="sid", gid=0, lot_number="109W", exp_date="2027-03-05"
    )
    assert result_second["exp_match"] is True


def test_duplicate_rows_none_match_ocr_falls_back_to_first_row(monkeypatch):
    checker = SheetChecker()
    rows = [
        _row("109W", exp="01/02/2027"),
        _row("109W", exp="05/03/2027"),
    ]
    monkeypatch.setattr(checker, "_find_rows_by_lot", lambda lot, sheet_id, gid: rows)
    result = checker.check(
        _cfg(), sheet_id="sid", gid=0, lot_number="109W", exp_date="2099-01-01"
    )
    assert result["lot_match"] is True   # lot itself still matched
    assert result["exp_match"] is False  # but no row's EXP agrees with OCR


def test_no_rows_match_lot(monkeypatch):
    checker = SheetChecker()
    monkeypatch.setattr(checker, "_find_rows_by_lot", lambda lot, sheet_id, gid: [])
    result = checker.check(
        _cfg(), sheet_id="sid", gid=0, lot_number="ZZZZ", exp_date="2027-02-01"
    )
    assert result["lot_match"] is False
    assert result["exp_match"] is False


def test_lot_short_fallback_retries_via_new_method(monkeypatch):
    checker = SheetChecker()
    calls = []

    def fake_find(lot, sheet_id, gid):
        calls.append(lot)
        if lot == "AAAAAAAAA1234XX":
            return []
        return [_row("1234", exp="01022027")]

    monkeypatch.setattr(checker, "_find_rows_by_lot", fake_find)
    result = checker.check(
        _cfg(lot_short_fallback=True), sheet_id="sid", gid=0,
        lot_number="AAAAAAAAA1234XX", exp_date="2027-02-01",
    )
    assert calls == ["AAAAAAAAA1234XX", "1234"]
    assert result["lot_match"] is True
    assert result["exp_match"] is True
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python -m pytest tests/test_sheet_checker.py -v`
Expected: every test **errors** with `AttributeError: <SheetChecker instance ...> does not have
the attribute '_find_rows_by_lot'` (monkeypatch's `setattr` fails because the method doesn't
exist yet — this is the "RED" signal for this rename).

- [ ] **Step 3: Rename `_find_row_by_lot` to `_find_rows_by_lot` and change it to return all matches**

In `utils/sheet_checker.py`, replace lines 71-79:

```python
    def _find_row_by_lot(self, lot: str | None, sheet_id: str, gid: int) -> dict | None:
        """หา row แรกที่ column 'Lot.' ตรงกับ lot (case-insensitive)"""
        if not lot:
            return None
        lot_key = lot.upper().strip()
        for row in self._get_rows(sheet_id, gid):
            if str(row.get("Lot.", "")).upper().strip() == lot_key:
                return row
        return None
```

with:

```python
    def _find_rows_by_lot(self, lot: str | None, sheet_id: str, gid: int) -> list[dict]:
        """หาทุก row ที่ column 'Lot.' ตรงกับ lot (case-insensitive) — อาจเจอมากกว่า 1 แถว
        ถ้าชีทมีข้อมูลซ้ำ (เช่น แถว placeholder ที่ยังไม่กรอก MFG/EXP + แถวจริงที่กรอกภายหลัง)"""
        if not lot:
            return []
        lot_key = lot.upper().strip()
        return [
            row for row in self._get_rows(sheet_id, gid)
            if str(row.get("Lot.", "")).upper().strip() == lot_key
        ]
```

- [ ] **Step 4: Update `check()` to resolve duplicate rows by EXP**

In `utils/sheet_checker.py`, replace lines 109-143 (from `lot = kwargs.get(...)` through the end
of the `else:` branch that sets `exp_match`/`product_match` to `False`):

```python
            lot = kwargs.get("lot_box") if is_container else kwargs.get("lot_number")
            rows = self._find_rows_by_lot(lot, sheet_id, gid)

            if not rows and config.lot_short_fallback and lot and len(lot) >= 13:
                short_lot = lot[9:13]
                logger.info(
                    "%s lot not found — retrying with short lot %s",
                    config.key,
                    short_lot,
                )
                rows = self._find_rows_by_lot(short_lot, sheet_id, gid)

            if "lot" in checks:
                result["lot_match"] = bool(rows)

            if rows:
                if len(rows) > 1:
                    logger.warning(
                        "%s: %d rows matched Lot %s — resolving by EXP",
                        config.key, len(rows), lot,
                    )

                exp_key = "exp_box" if is_container else "exp_date"
                exp = kwargs.get(exp_key)

                row = None
                if "exp" in checks and exp:
                    row = next(
                        (r for r in rows if _normalize_sheet_date(r.get("EXP", "")) == exp),
                        None,
                    )
                row = row or rows[0]

                result["sheet_product_name"] = row.get("Product Name", "").strip() or None

                if "exp" in checks:
                    sheet_exp = _normalize_sheet_date(row.get("EXP", ""))
                    result["exp_match"] = sheet_exp == exp if exp else False

                if "product" in checks:
                    product = kwargs.get("product_name")
                    sheet_product = str(row.get("Product Name", "")).strip().lower()
                    result["product_match"] = bool(
                        product and product.strip().lower() == sheet_product
                    )
            else:
                if "exp" in checks:
                    result["exp_match"] = False
                if "product" in checks:
                    result["product_match"] = False
```

(The `"sachet" in checks` block after this, lines 145-158, is untouched — leave it exactly as
it is.)

- [ ] **Step 5: Update the existing test that pokes the renamed method**

In `tests/test_routing.py`, the `test_sheet_checker_lot_source` test monkeypatches
`_find_row_by_lot` and has its fake return `None`. Update it for the new name/return type.
Replace lines 53-69:

```python
def test_sheet_checker_lot_source(monkeypatch, mode, expected_lot_key):
    checker = SheetChecker()
    seen = {}

    def fake_find(lot, sheet_id, gid):
        seen["lot"] = lot
        return None

    monkeypatch.setattr(checker, "_find_row_by_lot", fake_find)
    checker.check(
        _cfg(detection_mode=mode),
        sheet_id="sid", gid=0,
        lot_number="LOTNUM", lot_box="LOTBOX", lot_sachet="LOTSACHET",
        exp_date="2026-01-01", exp_box="2026-01-01", exp_sachet="2026-01-01",
    )
    expected = {"lot_number": "LOTNUM", "lot_box": "LOTBOX"}[expected_lot_key]
    assert seen["lot"] == expected
```

with:

```python
def test_sheet_checker_lot_source(monkeypatch, mode, expected_lot_key):
    checker = SheetChecker()
    seen = {}

    def fake_find(lot, sheet_id, gid):
        seen["lot"] = lot
        return []

    monkeypatch.setattr(checker, "_find_rows_by_lot", fake_find)
    checker.check(
        _cfg(detection_mode=mode),
        sheet_id="sid", gid=0,
        lot_number="LOTNUM", lot_box="LOTBOX", lot_sachet="LOTSACHET",
        exp_date="2026-01-01", exp_box="2026-01-01", exp_sachet="2026-01-01",
    )
    expected = {"lot_number": "LOTNUM", "lot_box": "LOTBOX"}[expected_lot_key]
    assert seen["lot"] == expected
```

- [ ] **Step 6: Run the full test suite for both files to verify everything passes**

Run: `python -m pytest tests/test_sheet_checker.py tests/test_routing.py -v`
Expected: all tests **PASS** (6 new tests in `test_sheet_checker.py` + the 3 parametrized cases
of `test_sheet_checker_lot_source` in `test_routing.py`).

- [ ] **Step 7: Run the broader test suite to check for unrelated regressions**

Run: `python -m pytest -x -q`
Expected: no new failures beyond the pre-existing known failure (`tests/test_classifier.py`,
3 setup errors unrelated to this change — see `CLAUDE.md` "Known pre-existing failure").

- [ ] **Step 8: Commit**

```bash
git add utils/sheet_checker.py tests/test_sheet_checker.py tests/test_routing.py
git commit -m "fix: resolve duplicate Lot rows in sheet check by matching OCR's EXP

_find_row_by_lot returned only the first sheet row matching a Lot, so a
blank placeholder row (added before a label was printed) could shadow
a later row with the real MFG/EXP for the same Lot — reporting a false
exp_match=False even when OCR read the expiry correctly. Now every
Lot-matching row is considered, and the one whose EXP agrees with what
OCR read wins; falls back to the first row (today's behavior) only when
no candidate matches."
```

---

### Task 2: Verify against the real debugging session (manual, no code changes)

**Files:** none (verification only)

**Interfaces:** none

- [ ] **Step 1: Reproduce the original bug is fixed with the real sheet and the real OCR result**

This repeats the ad-hoc reproduction from the debugging session, which had confirmed
`exp_match: False` before this fix (Lot `109W`, `retail_sachet`, sheet
`1kp_4uXIWnhDuCz1obMqzSOnKFBP9BFqMP-OqCAiTuIE`, gid `666410502`). Run from the repo root
(needs `GOOGLE_APPLICATION_CREDENTIALS=./gcp-key.json`, already set in `.env`):

```bash
python -c "
import os
os.environ.setdefault('GOOGLE_APPLICATION_CREDENTIALS', './gcp-key.json')
from pipeline.packaging_registry import PackagingRegistry
from utils.sheet_checker import SheetChecker

registry = PackagingRegistry()
config = registry.get('retail_sachet')
checker = SheetChecker()
result = checker.check(config, '1kp_4uXIWnhDuCz1obMqzSOnKFBP9BFqMP-OqCAiTuIE', 666410502, lot_number='109W', exp_date='2027-02-01')
print(result)
"
```

Expected: `{'lot_match': True, 'exp_match': True, 'product_match': None, 'sachet_match': None,
'sheet_product_name': 'Rootroute Wheatgrass Powder 5 g'}` — `exp_match` is now `True` (it was
`False` before this fix, with the exact same inputs).

- [ ] **Step 2: Confirm the duplicate-row warning fires**

Re-run the app locally with `LOG_LEVEL=INFO` (or check logs from Step 1 if the logger is
already configured to print to console) and confirm a line like
`retail_sachet: 2 rows matched Lot 109W — resolving by EXP` appears — this is the visibility
signal from `utils/sheet_checker.py` Step 4, confirming the duplicate was detected (not just
silently worked around).

No commit for this task — it's a manual confirmation, not a code change.
