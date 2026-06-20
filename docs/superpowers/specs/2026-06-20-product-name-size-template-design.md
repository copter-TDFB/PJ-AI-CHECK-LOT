# Product Name Composition via `{size}` Template

**Date:** 2026-06-20
**Status:** Approved — ready for implementation

## Problem

When a Google Sheet's `Product Name` column embeds the size (e.g. `Medium 40 g`),
the wizard's `product_aliases` mechanism cannot produce a matching name without
either (a) hardcoding the size into each `canonical` and splitting one product
into one row per size, or (b) relying on a rigid auto-append that cannot be
turned off per product.

Two concrete needs from the user:
1. A single product (`medium`) sold in several sizes (`40 g`, `100 g`) should be
   handled by **one** alias row, not one row per size.
2. Some products need fixed text appended (`Houjicha` → `Houjicha Powder`), some
   need a size appended (`Medium {size}`), and some need **nothing** appended
   (`Excellent`). The choice must be **per product**, not global.

## Current behavior (important)

`pipeline/ocr_engine.py:65-69` already composes product + size — but rigidly:

```python
size = find_size(full_text) if extract_size else None
aliases = config.product_aliases if config else None
product_name = find_product_name(full_text, aliases) if extract_product else None
if product_name and size:
    product_name = f"{product_name} {size}"   # size ALWAYS appended, at the end, no opt-out
```

This always-on append:
- locks size to the end with a single space, and
- cannot be disabled per product — so a packaging that has both `Houjicha Powder`
  (sheet has no size) and `Medium 40 g` (sheet has size) cannot coexist once the
  `size` field is enabled.

`back_label` and `grade_bag` are the only shipped classes that extract
`[lot, exp, product, size]`, and **both use the hardcoded fallback** (no
`product_aliases`). Production depends on the auto-append for them
(e.g. `Houjicha Powder 40 g`). **This behavior must not change.**

## Decision

Make `canonical` a **free-form template** with one special token, `{size}`.
Branch on whether the config uses `product_aliases`:

- **product_aliases path** (config-driven, wizard packagings): resolve the
  template. `{size}` is replaced by the extracted size; all other text is
  literal. **No automatic append** — each row decides via the token.
  - `Houjicha Powder` → `Houjicha Powder` (no token → literal)
  - `Medium {size}` + size `40 g` → `Medium 40 g`
  - `Excellent {size} Powder` + size `200 g` → `Excellent 200 g Powder`
  - `Medium {size}` + size **not found** → `None` (product treated as no-match)
- **hardcoded fallback path** (no `product_aliases`, i.e. `back_label`/
  `grade_bag`): keep the existing `f"{name} {size}"` auto-append unchanged.

Rule for missing size (user decision **ก**): when a template contains `{size}`
but no size was extracted, `find`/resolve returns `None`, which makes
`product_match` false in `sheet_checker` (cannot verify → do not guess).

## Backend design

1. `utils/validators.py` — add a small pure helper:
   ```python
   def resolve_product_template(template: str | None, size: str | None) -> str | None:
       if not template:
           return None
       if "{size}" not in template:
           return template                      # literal (e.g. "Houjicha Powder")
       if not size:
           return None                          # token present but size missing → rule ก
       resolved = template.replace("{size}", size)
       return " ".join(resolved.split())        # collapse stray double-spaces, strip
   ```
   `find_product_name` is unchanged — it still returns the matched `canonical`
   (now possibly containing `{size}`).

2. `pipeline/ocr_engine.py:65-69` — replace the rigid append with a branch:
   ```python
   size = find_size(full_text) if extract_size else None
   aliases = config.product_aliases if config else None
   product_name = find_product_name(full_text, aliases) if extract_product else None
   if product_name:
       if aliases:
           product_name = resolve_product_template(product_name, size)
       elif size:
           product_name = f"{product_name} {size}"   # legacy fallback path, unchanged
   ```

No schema / store / YAML changes — `canonical` is already an opaque string;
`Medium {size}` is a valid value today.

## Frontend design (`web/wizard.html`)

1. **Rewrite the `.pa-hint` callout** added on 2026-06-20. Its current text tells
   users to type the size into `canonical` manually and split rows per size —
   that advice is now wrong. Replace with a `{size}` token explanation.
2. Update the `canonical` input placeholder/label to signal it is a template
   (e.g. placeholder `Medium {size}`).
3. **Live per-row preview**: as the user types a template, show the resolved
   result using a sample size (`40 g`):
   - `Medium {size}` → `Medium 40 g`
   - `Houjicha Powder` → `Houjicha Powder`
   - if `{size}` is used while the `size` field is **not** enabled, the preview
     surfaces that (so the footgun is visible without separate validation logic).

## Out of scope (known gap, not speculative work)

`pipeline/pipeline_runner.py` (`_run_multi_region` / `_run_multi_field`) does
**not** compose product + size and is left unchanged. No shipped `cross_check`
or `multi_field` packaging extracts product + size with `product_aliases`, so
adding template resolution there would be code for a case that does not exist.
Recorded here so a future grouped/cross-check packaging that needs it knows to
reuse `resolve_product_template`.

## Success criteria

- Unit tests in `tests/test_ocr.py` cover `resolve_product_template`:
  - `{size}` substitution → `Medium {size}` + `40 g` → `Medium 40 g`
  - no token → literal returned unchanged
  - `{size}` present + size `None` → `None`
  - double-space collapse for awkward templates
- An `ocr_engine`-level test (or existing equivalent) confirms:
  - aliases path with `{size}` resolves via the template
  - non-aliases path (`back_label`/`grade_bag` style) still appends size
    (`Houjicha Powder` + `40 g` → `Houjicha Powder 40 g`)
- Wizard renders the new hint, template placeholder, and a working live preview
  (verified by serving `web/` and inspecting the `#pa-card` step).

## Files touched

- `utils/validators.py` — add `resolve_product_template`
- `pipeline/ocr_engine.py` — branch aliases vs legacy append
- `web/wizard.html` — hint rewrite, placeholder, live preview
- `tests/test_ocr.py` — new cases
