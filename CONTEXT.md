# OCR Lot Checker

Classifies a product packaging photo, OCRs its lot/expiry region, and
cross-checks the values against a Google Sheet. One context: the whole
pipeline from photo to verify-message.

## Language

**Packaging**:
A recognizable product packaging type (e.g. back_label, container_label) that the classifier can identify and the pipeline knows how to read.
_Avoid_: class, category, product type

**Conf threshold**:
The per-packaging minimum classifier confidence required to run the pipeline; below it the photo is rejected as low-confidence. User-tunable between 0.50 and 0.95.
_Avoid_: confidence, conf (ambiguous — see Detector confidence)

**Detector confidence**:
YOLO's internal certainty when locating a lot region inside a photo. System-wide, internal, not user-tunable.
_Avoid_: conf threshold (that is the classifier gate)

**Runtime tuning field**:
A packaging setting that changes pipeline behavior without retraining any model and is reversible instantly. Currently only the conf threshold. Runtime tuning fields are exempt from the clone-edit flow.

**Override**:
A persisted, user-set value for a runtime tuning field that wins over the packaging's baked-in configuration.

**Active / Draft / Archived**:
An active packaging serves production traffic. A draft is a packaging being built in the wizard, not yet deployed. An archived packaging is soft-disabled — recognized by the classifier but refused by the pipeline.

**Edit-draft**:
A draft cloned from an active packaging in order to retrain it; deploying it overwrites the parent.

**Detector class** (`{key}_{region}`):
A physical place on the packaging that the YOLO detector is trained to locate and crop, baked into `detector.pt`. A packaging can own SEVERAL detector classes — `back_label` has `back_label_lot` + `back_label_name` + `back_label_size`; `grade_bag` has `grade_bag_lot` + `grade_bag_product`. At inference the detector crops EVERY class of the predicted key (matched by `{key}_` prefix in `detector.py`), stacks all crops vertically, and OCRs once. So multiple detector classes do NOT mean multiple OCR passes — they are pieces of one combined image. `detector_yolo_prefixes` in the YAML must list these classes (used only by edit-draft prelabel filtering, not by inference). A product still carries only ONE lot value — extra classes are just other text areas (name/size), never a second lot.

**Sub-region** (จุด crop):
A *routing* declaration in `sub_regions`, NOT the count of detector classes. `len(sub_regions) > 1` is the ONLY thing that switches the pipeline into multi-region cross-check mode; `[]` or one entry = single-region (stack all detector crops → one OCR). The sole multi-region case is `container_label` (`box` + `sachet`): the same lot/date is printed on both the outer box and the inner sachet, and the two crops are OCR'd separately then cross-checked (`exp_box == exp_sachet`) to catch mispackaging — a sachet from a different lot boxed by mistake. `back_label`'s three detector classes are NOT sub-regions — it stays `sub_regions: []` (single-region routing).
_Avoid_: equating sub-region with detector class — `back_label` has 3 detector classes but 0 sub-regions.

**Field**:
A value extracted from the combined OCR text by regex/validators — `lot`, `exp`, `product`, `size`. The text can come from one or several detector-class crops stacked together (e.g. `back_label`'s lot/name/size crops yield all four fields). Declared in `fields_extracted`, independent of routing. A field only resolves if its text is actually inside one of the cropped regions.
_Avoid_: sub-region, crop point (those are locations/routing, not values).

## Flagged ambiguities

- **"conf"** alone is ambiguous between the classifier gate and YOLO's
  detection certainty. Resolved 2026-06-11: when users say "ปรับค่า conf"
  they mean the **Conf threshold** (classifier gate). Always qualify which
  one is meant.
- **"จุดที่ crop" vs field**. Resolved 2026-06-12: a single crop region that
  OCRs both lot and expiry is the *default* (single-region routing, many
  **Fields**) — the `back_label` model. It is NOT a case for multiple
  **sub-regions** (routing). Multiple sub-regions are only for
  physically-separate locations that need cross-check (`box`/`sachet`). The
  wizard must not offer field names (`lot`, `exp`, `name`, `size`) as
  *sub-region* (routing) choices.
- **Sub-region vs detector class**. Clarified 2026-06-12: these are different
  layers. `back_label` has THREE **detector classes** (`back_label_lot/name/size`
  in `detector.pt`) yet `sub_regions: []` (single-region routing) — the three
  crops stack into one OCR. "Multiple detector classes" ≠ "multiple
  sub-regions". `detector_yolo_prefixes` lists the detector classes and is read
  ONLY by edit-draft prelabel; live inference matches by `{key}_` prefix in
  `detector.py` and ignores that field.

## Example dialogue

> **Dev:** A user says back_label photos keep coming back low-confidence.
> **Expert:** Then lower its conf threshold — that's a runtime tuning field,
> set the override from the wizard, no retrain needed.
> **Dev:** Don't I need an edit-draft for that?
> **Expert:** No — edit-drafts are for retraining. If the *detector* were
> missing the lot region, that's detector confidence territory and a
> retrain via edit-draft would be the fix.
