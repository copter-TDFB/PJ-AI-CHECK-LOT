# 0006 — Drive dataset write runs as an OAuth user, not the service account

Date: 2026-06-12
Status: Accepted (fixes the storage-quota flaw in ADR 0003; the direct-write
data path of ADR 0003 stays in force — only the identity changes)

## Context

ADR 0003 had the backend write the reference dataset directly into Drive using
the service account (`google_auth_default()` in `services/drive_client.py`),
and claimed "files created by the SA count against the SA's 15 GB quota". That
premise is wrong: **service accounts have no Drive storage quota of their own.**
Any SA-owned upload to My Drive fails with:

```
HttpError 403 ... "Service Accounts do not have storage quota. Leverage
shared drives ..., or use OAuth delegation ... instead."
(reason: storageQuotaExceeded)
```

So Full Training never actually published — the dataset upload 403'd on the
first file. The download paths (manifest, models) kept working because reading
does not consume quota.

Three ways to give the upload a real storage quota were considered:

1. **Shared Drive** — files are owned by the drive, not the SA. Requires a
   Workspace Shared Drive and `supportsAllDrives=True` on every Drive call.
2. **Domain-wide delegation** — the SA impersonates a Workspace user. Requires
   a Workspace admin to authorize the SA's client id; we do not have admin.
3. **OAuth user credentials** — the backend acts as a real Workspace user whose
   Drive quota owns the dataset.

`tdfb.co` is a Google Workspace domain, so an OAuth consent screen configured as
**Internal** may use the restricted full `drive` scope without Google's
verification process, and its refresh token does not expire (the 7-day
testing-mode expiry only applies to External apps). This sidesteps the
"restricted scope" blocker that pushed ADR 0001 to the addition-bundle design.

## Decision

`DriveClient` authenticates as an OAuth user when these env vars are set:

- `DRIVE_OAUTH_CLIENT_ID`
- `DRIVE_OAUTH_CLIENT_SECRET`
- `DRIVE_OAUTH_REFRESH_TOKEN`

When present it builds `google.oauth2.credentials.Credentials` from them;
otherwise it falls back to `google_auth_default()` (service account / ADC), so
read-only deployments and other callers are unaffected. The scope stays full
`drive` — full scope lets the user write into the existing, hand-created
reference folders without re-uploading them.

The refresh token is minted once with `scripts/generate_drive_token.py`
(`InstalledAppFlow`, loopback redirect). Uploads are owned by the consenting
user (`sarutipong@tdfb.co`, ~3.2 TB Workspace quota).

`sheet_checker.py` and `cloudrun_deployer.py` keep using the service account —
only `DriveClient` changes.

## Consequences

**Positive** — Full Training can publish again; uploads draw on the user's
Workspace quota; no Shared Drive migration, no admin involvement, no re-upload
of existing reference data; token does not expire.

**Negative** — a long-lived refresh token + client secret are now deployment
secrets (store in Cloud Run env / Secret Manager, never commit); uploads are
tied to one human's account, so off-boarding that user means re-minting the
token; the backend still holds full-drive scope (same blast radius as ADR 0003,
mitigated by append-only names + commit-point ordering).

**Prerequisite** — the OAuth user must have Editor access to the dataset
folders in `DRIVE_DETECTOR_DATASET_FOLDER_ID` /
`DRIVE_CLASSIFIER_DATASET_FOLDER_ID` (own them or have them shared).
