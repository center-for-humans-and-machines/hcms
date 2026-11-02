"""Tests for retrying rate-limited Google Drive uploads (BAN-484)."""

# pylint: disable=missing-function-docstring,too-few-public-methods

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import httplib2
import pytest
from googleapiclient.errors import HttpError

import upload_to_gdrive
from upload_to_gdrive import UPLOAD_MAX_TRIES, upload_file, upload_files_by_pattern


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Backoff waits through ``time.sleep``; the tests should not."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


def _http_error(
    status: int,
    message: str = "User rate limit exceeded.",
    reason: str = "userRateLimitExceeded",
) -> HttpError:
    content = {
        "error": {
            "code": status,
            "message": message,
            "errors": [{"domain": "usageLimits", "reason": reason, "message": message}],
        }
    }
    return HttpError(
        resp=httplib2.Response({"status": status}),
        content=json.dumps(content).encode(),
    )


class _FakeFiles:
    """Drive ``files()`` resource whose ``create().execute()`` fails on cue."""

    def __init__(self, failures: List[Exception]) -> None:
        self.failures = list(failures)
        self.calls = 0
        self.create_params: List[dict] = []

    def create(self, **params: Any) -> SimpleNamespace:
        self.create_params.append(params)
        return SimpleNamespace(execute=self._execute)

    def _execute(self) -> dict:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return {"id": f"file-{self.calls}"}


def _service(failures: List[Exception]):
    files = _FakeFiles(failures)
    return SimpleNamespace(files=lambda: files), files


@pytest.fixture(name="data_file")
def fixture_data_file(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.json"
    path.write_text("{}")
    return path


def test_upload_retries_rate_limit_then_succeeds(data_file):
    service, files = _service([_http_error(403), _http_error(403)])

    assert upload_file(service, str(data_file), "folder-id") == "file-3"
    assert files.calls == 3


@pytest.mark.parametrize("status", [429, 500, 503])
def test_upload_retries_transient_statuses(data_file, status):
    service, files = _service([_http_error(status, "Backend Error", "backendError")])

    assert upload_file(service, str(data_file), "folder-id") == "file-2"
    assert files.calls == 2


def test_upload_gives_up_after_max_tries(data_file):
    service, files = _service([_http_error(403)] * (UPLOAD_MAX_TRIES + 1))

    assert upload_file(service, str(data_file), "folder-id") is None
    assert files.calls == UPLOAD_MAX_TRIES


@pytest.mark.parametrize(
    "error",
    [
        _http_error(403, "Insufficient permissions", "insufficientFilePermissions"),
        _http_error(404, "File not found", "notFound"),
        _http_error(400, "Bad request", "badRequest"),
    ],
)
def test_upload_does_not_retry_permanent_errors(data_file, error):
    service, files = _service([error])

    assert upload_file(service, str(data_file), "folder-id") is None
    assert files.calls == 1


def test_upload_passes_shared_drive_flag(data_file):
    service, files = _service([])

    upload_file(service, str(data_file), "folder-id", drive_id="drive-id")

    assert files.create_params[0]["supportsAllDrives"] is True


def _patch_drive_setup(monkeypatch, service) -> None:
    monkeypatch.setattr(
        upload_to_gdrive, "authenticate_drive_service_account", lambda: service
    )
    monkeypatch.setattr(
        upload_to_gdrive, "find_folder_by_path", lambda *_a, **_k: "target-id"
    )
    monkeypatch.setattr(
        upload_to_gdrive, "get_or_create_folder", lambda *_a, **_k: "date-id"
    )


def test_run_succeeds_when_retries_recover(monkeypatch, tmp_path):
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")
    service, files = _service([_http_error(403)])
    _patch_drive_setup(monkeypatch, service)

    exit_code = upload_files_by_pattern(
        str(tmp_path / "*.json"), credentials_file=str(tmp_path / "creds.json")
    )

    assert exit_code == 0
    assert files.calls == 3


def test_run_fails_when_a_file_still_fails_after_retries(monkeypatch, tmp_path):
    (tmp_path / "a.json").write_text("{}")
    service, _ = _service([_http_error(403)] * UPLOAD_MAX_TRIES)
    _patch_drive_setup(monkeypatch, service)

    exit_code = upload_files_by_pattern(
        str(tmp_path / "*.json"), credentials_file=str(tmp_path / "creds.json")
    )

    assert exit_code == 1
