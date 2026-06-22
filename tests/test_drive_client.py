"""Unit tests for DriveClient extensions — Drive API fully mocked."""

import io
from unittest.mock import MagicMock, patch

import pytest


class FakeDownloader:
    """Stands in for MediaIoBaseDownload — writes fixed bytes into the buffer."""

    payload = b"names: [a, b]\n"

    def __init__(self, buf, request, chunksize=None):
        self._buf = buf

    def next_chunk(self):
        self._buf.write(self.payload)
        return None, True


@pytest.fixture
def client_and_svc():
    with patch("services.drive_client.google_auth_default",
               return_value=(MagicMock(), None)), \
         patch("services.drive_client.build") as mock_build:
        svc = MagicMock()
        mock_build.return_value = svc
        from services.drive_client import DriveClient
        yield DriveClient(), svc


def test_scope_is_full_drive():
    from services import drive_client
    assert drive_client._SCOPES == ["https://www.googleapis.com/auth/drive"]


_OAUTH_ENV = {
    "DRIVE_OAUTH_CLIENT_ID": "cid.apps.googleusercontent.com",
    "DRIVE_OAUTH_CLIENT_SECRET": "GOCSPX-secret",
    "DRIVE_OAUTH_REFRESH_TOKEN": "1//refresh",
}


def test_user_oauth_credentials_none_when_env_missing(monkeypatch):
    from services import drive_client
    for key in _OAUTH_ENV:
        monkeypatch.delenv(key, raising=False)
    assert drive_client._user_oauth_credentials() is None


def test_user_oauth_credentials_none_when_partial(monkeypatch):
    from services import drive_client
    monkeypatch.setenv("DRIVE_OAUTH_CLIENT_ID", _OAUTH_ENV["DRIVE_OAUTH_CLIENT_ID"])
    monkeypatch.delenv("DRIVE_OAUTH_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("DRIVE_OAUTH_REFRESH_TOKEN", raising=False)
    assert drive_client._user_oauth_credentials() is None


def test_user_oauth_credentials_built_when_all_set(monkeypatch):
    from services import drive_client
    for key, val in _OAUTH_ENV.items():
        monkeypatch.setenv(key, val)
    creds = drive_client._user_oauth_credentials()
    assert creds is not None
    assert creds.refresh_token == "1//refresh"
    assert creds.client_id == "cid.apps.googleusercontent.com"
    assert creds.scopes == ["https://www.googleapis.com/auth/drive"]


def test_driveclient_prefers_oauth_over_adc(monkeypatch):
    """When OAuth env is set, DriveClient must NOT fall back to ADC/service account."""
    for key, val in _OAUTH_ENV.items():
        monkeypatch.setenv(key, val)
    with patch("services.drive_client.google_auth_default") as mock_adc, \
         patch("services.drive_client.build") as mock_build:
        from services.drive_client import DriveClient
        DriveClient()
        mock_adc.assert_not_called()
        # build() received the user OAuth credentials, not an SA
        _, kwargs = mock_build.call_args
        assert kwargs["credentials"].refresh_token == "1//refresh"


def test_read_text_decodes_utf8(client_and_svc):
    client, _ = client_and_svc
    with patch("services.drive_client.MediaIoBaseDownload", FakeDownloader):
        assert client.read_text("fid") == "names: [a, b]\n"


class FakeJsonDownloader(FakeDownloader):
    """FakeDownloader variant with a JSON payload."""

    payload = b'{"version": "v1"}'


def test_read_json_parses_json(client_and_svc):
    client, _ = client_and_svc
    with patch("services.drive_client.MediaIoBaseDownload", FakeJsonDownloader):
        assert client.read_json("fid") == {"version": "v1"}


def test_update_file_content_calls_files_update(client_and_svc):
    client, svc = client_and_svc
    client.update_file_content("fid123", b"new content", mime_type="text/yaml")
    _, kwargs = svc.files.return_value.update.call_args
    assert kwargs["fileId"] == "fid123"
    assert "media_body" in kwargs
    media = kwargs["media_body"]
    assert media.mimetype() == "text/yaml"
    assert media.getbytes(0, media.size()) == b"new content"


def test_list_folder_paginates(client_and_svc):
    client, svc = client_and_svc
    svc.files.return_value.list.return_value.execute.side_effect = [
        {"files": [{"id": "1", "name": "a.jpg"}], "nextPageToken": "tok"},
        {"files": [{"id": "2", "name": "b.jpg"}]},
    ]
    files = client.list_folder("parent")
    assert [f["name"] for f in files] == ["a.jpg", "b.jpg"]
    assert svc.files.return_value.list.call_count == 2


def test_ensure_folder_returns_existing(client_and_svc):
    client, svc = client_and_svc
    svc.files.return_value.list.return_value.execute.return_value = {
        "files": [{"id": "existing-id"}]
    }
    assert client.ensure_folder("train", "root") == "existing-id"
    svc.files.return_value.create.assert_not_called()


def test_ensure_folder_creates_when_missing(client_and_svc):
    client, svc = client_and_svc
    svc.files.return_value.list.return_value.execute.return_value = {"files": []}
    svc.files.return_value.create.return_value.execute.return_value = {"id": "new-id"}
    assert client.ensure_folder("train", "root") == "new-id"
