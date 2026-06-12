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

## Flagged ambiguities

- **"conf"** alone is ambiguous between the classifier gate and YOLO's
  detection certainty. Resolved 2026-06-11: when users say "ปรับค่า conf"
  they mean the **Conf threshold** (classifier gate). Always qualify which
  one is meant.

## Example dialogue

> **Dev:** A user says back_label photos keep coming back low-confidence.
> **Expert:** Then lower its conf threshold — that's a runtime tuning field,
> set the override from the wizard, no retrain needed.
> **Dev:** Don't I need an edit-draft for that?
> **Expert:** No — edit-drafts are for retraining. If the *detector* were
> missing the lot region, that's detector confidence territory and a
> retrain via edit-draft would be the fix.
