"""Google Drive API v3 wrapper — download + upload (full drive scope)."""

import io
import json
import logging
import mimetypes
import os
from pathlib import Path

from google.auth import default as google_auth_default
from google.oauth2.credentials import Credentials as UserCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload

logger = logging.getLogger(__name__)

# Full drive scope — required to write into the hand-created reference dataset
# folders. Allowed without Google verification because the OAuth consent screen
# is configured as Internal within the tdfb.co Workspace (see ADR 0005).
_SCOPES = ["https://www.googleapis.com/auth/drive"]

_TOKEN_URI = "https://oauth2.googleapis.com/token"

# Files at/under this size use a single-request (simple) upload instead of a
# resumable one — resumable adds a session-init round-trip that only pays off
# for large/flaky transfers.
_SIMPLE_UPLOAD_MAX_BYTES = 5 * 1024 * 1024


def _user_oauth_credentials() -> UserCredentials | None:
    """Build user OAuth creds from env if all three vars are set, else None.

    Service accounts have zero Drive storage quota, so SA-owned uploads fail
    with 403 storageQuotaExceeded (the bug ADR 0003 missed). When these vars
    are present the client acts as a real Workspace user whose Drive quota
    owns the dataset; otherwise it falls back to ADC/service account, which
    still works for read-only paths (manifest/model download). See ADR 0005.
    """
    client_id = os.environ.get("DRIVE_OAUTH_CLIENT_ID")
    client_secret = os.environ.get("DRIVE_OAUTH_CLIENT_SECRET")
    refresh_token = os.environ.get("DRIVE_OAUTH_REFRESH_TOKEN")
    if not (client_id and client_secret and refresh_token):
        return None
    logger.info("DriveClient: using OAuth user credentials (Drive quota owner)")
    return UserCredentials(
        token=None,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        token_uri=_TOKEN_URI,
        scopes=_SCOPES,
    )


class DriveClient:
    def __init__(self) -> None:
        creds = _user_oauth_credentials()
        if creds is None:
            creds, _ = google_auth_default(scopes=_SCOPES)
        # cache_discovery=False ป้องกัน warning ใน serverless env
        self._svc = build("drive", "v3", credentials=creds, cache_discovery=False)

    # ─── Download ────────────────────────────────────────

    def _download_bytes(self, file_id: str) -> bytes:
        """Stream file content from Drive into memory and return raw bytes."""
        request = self._svc.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        dl = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = dl.next_chunk()
        return buf.getvalue()

    def read_json(self, file_id: str) -> dict:
        """Download file from Drive และ parse เป็น dict."""
        return json.loads(self._download_bytes(file_id))

    def read_text(self, file_id: str) -> str:
        """Download file from Drive เป็น UTF-8 text."""
        return self._download_bytes(file_id).decode("utf-8")

    def download_file(self, file_id: str, dest: Path) -> None:
        """Download file from Drive ไปเก็บที่ dest (เขียนทับถ้ามีอยู่แล้ว)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        request = self._svc.files().get_media(fileId=file_id)
        with dest.open("wb") as f:
            dl = MediaIoBaseDownload(f, request, chunksize=10 * 1024 * 1024)
            done = False
            while not done:
                status, done = dl.next_chunk()
                if status:
                    logger.debug("  %.0f%%", status.progress() * 100)
        logger.info("Downloaded → %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)

    # ─── Upload + folders ────────────────────────────────

    def create_folder(self, name: str, parent_id: str | None = None) -> str:
        """สร้าง folder ใน Drive — return file_id. ถ้า parent_id None → My Drive root."""
        metadata = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
        if parent_id:
            metadata["parents"] = [parent_id]
        created = self._svc.files().create(body=metadata, fields="id").execute()
        logger.info("Drive folder created: %s (id=%s)", name, created["id"])
        return created["id"]

    def upload_file(
        self,
        src: Path,
        parent_id: str | None = None,
        name: str | None = None,
        public: bool = False,
    ) -> str:
        """Upload local file → return file_id.

        public=True → set permission to 'anyone with link can view' (สำหรับให้
        notebook ใน Colab download ผ่าน gdown โดยไม่ต้อง auth).
        """
        metadata: dict = {"name": name or src.name}
        if parent_id:
            metadata["parents"] = [parent_id]
        mime, _ = mimetypes.guess_type(str(src))
        # Resumable uploads cost an extra session-init round-trip. For small
        # files (the common case — packaging photos < 5MB) a simple multipart
        # upload is a single request, roughly halving per-image latency on the
        # sequential dataset publish. Reserve resumable for genuinely large files.
        resumable = src.stat().st_size > _SIMPLE_UPLOAD_MAX_BYTES
        media = MediaFileUpload(
            str(src), mimetype=mime or "application/octet-stream", resumable=resumable
        )
        created = self._svc.files().create(body=metadata, media_body=media, fields="id").execute()
        file_id = created["id"]
        logger.info("Uploaded %s → file_id=%s (%.1f MB)", src.name, file_id, src.stat().st_size / 1e6)
        if public:
            self._make_public(file_id)
        return file_id

    def copy_file(
        self, file_id: str, parent_id: str, name: str | None = None
    ) -> str:
        """Server-side copy an existing Drive file into parent_id → new file_id.

        Avoids re-uploading bytes that already live in Drive (e.g. the dataset
        publish needs the same image in both the detector and classifier
        folders). A metadata-only operation — far cheaper than a second upload.
        """
        body: dict = {"parents": [parent_id]}
        if name:
            body["name"] = name
        created = self._svc.files().copy(fileId=file_id, body=body, fields="id").execute()
        return created["id"]

    def upload_bytes(
        self,
        content: bytes,
        name: str,
        parent_id: str | None = None,
        mime_type: str = "application/octet-stream",
        public: bool = False,
        progress_cb=None,
    ) -> str:
        """Upload bytes (ไม่ผ่าน disk) → return file_id.

        progress_cb(sent_bytes, total_bytes) → switches to a resumable
        chunked upload so the caller gets per-chunk progress (ใช้กับ
        bundle zip ใหญ่ๆ ให้ wizard แสดง % ได้).
        """
        metadata: dict = {"name": name}
        if parent_id:
            metadata["parents"] = [parent_id]
        if progress_cb is not None:
            media = MediaIoBaseUpload(
                io.BytesIO(content), mimetype=mime_type,
                resumable=True, chunksize=1024 * 1024,
            )
            request = self._svc.files().create(body=metadata, media_body=media, fields="id")
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    progress_cb(status.resumable_progress, len(content))
            progress_cb(len(content), len(content))
            created = response
        else:
            media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime_type, resumable=False)
            created = self._svc.files().create(body=metadata, media_body=media, fields="id").execute()
        file_id = created["id"]
        logger.info("Uploaded bytes %s → file_id=%s (%.1f KB)", name, file_id, len(content) / 1024)
        if public:
            self._make_public(file_id)
        return file_id

    def update_file_content(
        self, file_id: str, content: bytes, mime_type: str = "text/plain"
    ) -> None:
        """เขียนทับเนื้อหาไฟล์เดิม (file_id คงเดิม)."""
        media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime_type, resumable=False)
        self._svc.files().update(fileId=file_id, media_body=media).execute()
        logger.info("Updated file content → %s (%.1f KB)", file_id, len(content) / 1024)

    def ensure_folder(self, name: str, parent_id: str) -> str:
        """Find folder by name under parent — create ถ้ายังไม่มี. Return folder_id."""
        safe = name.replace("'", "\\'")
        q = (
            f"name = '{safe}' and '{parent_id}' in parents and trashed = false "
            "and mimeType = 'application/vnd.google-apps.folder'"
        )
        resp = self._svc.files().list(q=q, fields="files(id)", pageSize=1).execute()
        files = resp.get("files", [])
        if files:
            return files[0]["id"]
        return self.create_folder(name, parent_id)

    def _make_public(self, file_id: str) -> None:
        """Grant anyone-with-link read access — required for Colab gdown."""
        self._svc.permissions().create(
            fileId=file_id,
            body={"role": "reader", "type": "anyone"},
        ).execute()
        logger.info("Granted public read → %s", file_id)

    # ─── Search ──────────────────────────────────────────

    def list_folder(self, parent_id: str) -> list[dict]:
        """List ทุกไฟล์/โฟลเดอร์ใน parent — return [{id, name, mimeType}]."""
        files: list[dict] = []
        token = None
        while True:
            resp = self._svc.files().list(
                q=f"'{parent_id}' in parents and trashed = false",
                fields="nextPageToken, files(id,name,mimeType)",
                pageSize=1000,
                pageToken=token,
            ).execute()
            files.extend(resp.get("files", []))
            token = resp.get("nextPageToken")
            if not token:
                return files

    def find_in_folder(self, parent_id: str, name: str) -> str | None:
        """Find file by name within a parent folder — return file_id or None."""
        # Escape single quotes in name
        safe_name = name.replace("'", "\\'")
        q = f"name = '{safe_name}' and '{parent_id}' in parents and trashed = false"
        resp = self._svc.files().list(q=q, fields="files(id,name)", pageSize=1).execute()
        files = resp.get("files", [])
        return files[0]["id"] if files else None
