"""Shared helpers for the dataset acquisition and audit scripts (Week 2).

Design rules:
  * every path is derived from the repository root (no absolute/personal paths);
  * raw data is never modified: downloads go to ``<file>.part`` and are renamed on success;
  * every downloaded file and every remote API query is registered in a versioned manifest.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
SOURCES_DIR = MANIFEST_DIR / "sources"
AUDIT_DIR = REPO_ROOT / "reports" / "audit"
RAW_FILES_CSV = MANIFEST_DIR / "raw_files.csv"
REMOTE_QUERIES_CSV = MANIFEST_DIR / "remote_queries.csv"

RAW_FILES_COLUMNS = [
    "dataset_id", "relative_path", "filename", "file_type", "size_bytes",
    "sha256", "source_url", "download_date", "notes",
]
REMOTE_QUERIES_COLUMNS = [
    "dataset_id", "query_id", "url", "params", "query_date_utc", "http_status",
    "response_sha256", "snapshot_path", "summary",
]

# Contact information is deliberately generic; do not add personal data here.
USER_AGENT = "edge-ai-virtus-dataset-audit/0.1 (academic research; metadata-first audit)"

log = logging.getLogger("virtus.data")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def rel(path: Path) -> str:
    """Repository-relative POSIX path (what manifests store)."""
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET", "HEAD"), respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = USER_AGENT
    # Never let `requests` transparently decode a stored object: raw bytes (and their SHA256) must match the source.
    s.headers["Accept-Encoding"] = "identity"
    return s


# --------------------------------------------------------------------------- manifests
def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_rows(path: Path, columns: list[str], rows: Iterable[dict[str, Any]], key: tuple[str, ...]) -> None:
    rows = sorted(rows, key=lambda r: tuple(str(r.get(k, "")) for k in key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def upsert_rows(path: Path, columns: list[str], new_rows: list[dict[str, Any]], key: tuple[str, ...]) -> None:
    """Insert or replace rows identified by ``key`` in a CSV manifest."""
    current = {tuple(r.get(k, "") for k in key): r for r in _read_rows(path)}
    for r in new_rows:
        current[tuple(str(r.get(k, "")) for k in key)] = r
    _write_rows(path, columns, current.values(), key)


def register_raw_file(dataset_id: str, path: Path, source_url: str = "", notes: str = "",
                      download_date: str | None = None, sha256: str | None = None) -> dict[str, Any]:
    """Compute size + SHA256 of a local raw file and upsert it into ``raw_files.csv``."""
    row = {
        "dataset_id": dataset_id,
        "relative_path": rel(path),
        "filename": path.name,
        "file_type": "".join(path.suffixes).lstrip(".") or "NA",
        "size_bytes": path.stat().st_size,
        "sha256": sha256 or sha256_file(path),
        "source_url": source_url,
        "download_date": download_date or utc_today(),
        "notes": notes,
    }
    prev = existing_raw_row(path)
    if download_date is None and prev and prev["sha256"] == row["sha256"]:
        row["download_date"] = prev["download_date"]   # same bytes as before: keep the original acquisition date
    upsert_rows(RAW_FILES_CSV, RAW_FILES_COLUMNS, [row], key=("dataset_id", "relative_path"))
    return row


def existing_raw_row(path: Path) -> dict[str, str] | None:
    for r in _read_rows(RAW_FILES_CSV):
        if r["relative_path"] == rel(path):
            return r
    return None


# --------------------------------------------------------------------------- downloads
def download_file(session: requests.Session, url: str, dest: Path, expected_size: int | None = None,
                  headers: dict[str, str] | None = None, timeout: int = 120) -> bool:
    """Stream ``url`` to ``dest``. Returns True if a download happened, False if skipped.

    Existing files are never overwritten: if ``dest`` exists (and matches ``expected_size`` when
    known) the download is skipped. A partial ``.part`` file is discarded and restarted.
    """
    if dest.exists():
        if expected_size is None or dest.stat().st_size == expected_size:
            log.info("skip (already present): %s", rel(dest))
            return False
        raise RuntimeError(
            f"{rel(dest)} exists with size {dest.stat().st_size} != expected {expected_size}; "
            "refusing to overwrite raw data - inspect it manually."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    part.unlink(missing_ok=True)
    log.info("download %s -> %s", url, rel(dest))
    with session.get(url, stream=True, timeout=timeout, headers=headers or {}) as r:
        r.raise_for_status()
        with part.open("wb") as fh:
            # decode_content=False: some servers (e.g. S3 objects stored with "Content-Encoding: gzip") would make
            # `requests` gunzip transparently; we want the exact bytes the source serves so hashes match.
            for block in r.raw.stream(1 << 20, decode_content=False):
                fh.write(block)
    if expected_size is not None and part.stat().st_size != expected_size:
        raise RuntimeError(f"size mismatch for {url}: got {part.stat().st_size}, expected {expected_size}")
    part.replace(dest)
    return True


def head_size(session: requests.Session, url: str) -> int | None:
    try:
        r = session.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        return int(r.headers["Content-Length"]) if "Content-Length" in r.headers else None
    except Exception as exc:  # noqa: BLE001 - informative only
        log.warning("HEAD failed for %s: %s", url, exc)
        return None


def store_snapshot(dataset_id: str, query_id: str, url: str, params: dict[str, Any] | None, data: Any,
                   status: int | str = 200, summary: str = "") -> Path:
    """Persist a JSON-serialisable query result as a raw snapshot and log it in the manifests."""
    path = RAW_DIR / dataset_id / "api" / f"{query_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True)
    path.write_text(body, encoding="utf-8")
    now = utc_now()
    upsert_rows(REMOTE_QUERIES_CSV, REMOTE_QUERIES_COLUMNS, [{
        "dataset_id": dataset_id, "query_id": query_id, "url": url,
        "params": json.dumps(params or {}, sort_keys=True), "query_date_utc": now,
        "http_status": status, "response_sha256": sha256_bytes(body.encode("utf-8")),
        "snapshot_path": rel(path), "summary": summary,
    }], key=("dataset_id", "query_id"))
    register_raw_file(dataset_id, path, source_url=url, notes=f"Instantánea de la consulta '{query_id}'",
                      download_date=now[:10])
    return path


def snapshot_json(session: requests.Session, dataset_id: str, query_id: str, url: str,
                  params: dict[str, Any] | None = None, refresh: bool = False,
                  sleep: float = 1.0, summary_key: str | None = None,
                  headers: dict[str, str] | None = None) -> Any:
    """Run a small remote API query once, store the JSON response as a raw snapshot and log it.

    The response is stored under ``data/raw/<dataset_id>/api/<query_id>.json`` and the query is
    logged (URL, parameters, timestamp, response hash) in ``data/manifests/remote_queries.csv``.
    Repeated runs reuse the stored snapshot unless ``refresh`` is set, so that the audit is
    computed on exactly the same responses.
    """
    path = RAW_DIR / dataset_id / "api" / f"{query_id}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    r = session.get(url, params=params, timeout=60, headers=headers or {})
    r.raise_for_status()
    data = r.json()
    summary = f"{summary_key}={data[summary_key]}" if summary_key and isinstance(data, dict) and summary_key in data else ""
    store_snapshot(dataset_id, query_id, r.url.split("?")[0], params, data, r.status_code, summary)
    time.sleep(sleep)
    return data
