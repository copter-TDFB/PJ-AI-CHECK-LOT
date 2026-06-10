"""Google Drive API v3 wrapper — download + upload (full drive scope)."""

import io
import json
import logging
import mimetypes
from pathlib import Path

from google.auth import default as google_auth_default
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload

logger = logging.getLogger(__name__)

# Full drive scope — required to write into the reference dataset folders
# that the user shares with this service account. SAs do not go through
# OAuth consent verification, so the "restricted scope" problem in ADR 0001
# does not apply (see ADR 0003).
_SCOPES = ["https://www.googleapis.com/auth/drive"]


class DriveClient:
    def __init__(self) -> None:
        creds, _ = google_auth_default(scopes=_SCOPES)
        # cache_discovery=False ป้องกัน warning ใน serverless env
        self._svc = build("drive", "v3", credentials=creds, cache_discovery=False)

    # ─── Download ────────────────────────────────────────

    def _download_bytes(self, file_id: str) -> bytes:
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
        media = MediaFileUpload(str(src), mimetype=mime or "application/octet-stream", resumable=True)
        created = self._svc.files().create(body=metadata, media_body=media, fields="id").execute()
        file_id = created["id"]
        logger.info("Uploaded %s → file_id=%s (%.1f MB)", src.name, file_id, src.stat().st_size / 1e6)
        if public:
            self._make_public(file_id)
        return file_id

    def upload_bytes(
        self,
        content: bytes,
        name: str,
        parent_id: str | None = None,
        mime_type: str = "application/octet-stream",
        public: bool = False,
    ) -> str:
        """Upload bytes (ไม่ผ่าน disk) → return file_id."""
        metadata: dict = {"name": name}
        if parent_id:
            metadata["parents"] = [parent_id]
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
