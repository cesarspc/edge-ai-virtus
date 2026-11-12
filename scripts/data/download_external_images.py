"""Download the images of an external evaluation set (resumable, polite, verifiable).

    python scripts/data/download_external_images.py --set ext_co_001     # iNaturalist CO + Mushroom Observer CO
    python scripts/data/download_external_images.py --set ext_full_001   # + MIND.Funga (complete test_external)
    python scripts/data/download_external_images.py --set ext_full_001 --limit 20   # trial
    python scripts/data/download_external_images.py --set ext_full_001 --retry-failed

Reads ``data/processed/<set>/images.csv`` (checked against ``data/versions/<set>/manifest.json``).

* iNaturalist / Mushroom Observer: each ``locator`` URL is fetched exactly as served (iNaturalist ``medium`` photos from
  the public Open Data bucket, Mushroom Observer 640 px). Mushroom Observer is run by volunteers: two workers with a
  pause between requests.
* MIND.Funga v2: the images live inside the official Mendeley ZIP (3,2 GB, DOI 10.17632/sfrbdjvxcc.2). Only the needed
  members are fetched, with HTTP Range requests through the system ``curl`` (Mendeley answers python-requests with a
  Cloudflare challenge; see ``acquire_datasets.curl_range``), merging neighbouring members into one request. Each
  member is decompressed and checked against the ZIP CRC-32 and against the SHA-256 published by the source.

Every file is decoded with Pillow; size, SHA-256 and dimensions go to the shared manifest.

Outputs (under --out, default data/raw/images/external_v001, shared by every set; ignored by Git):
    <source_id>/<source_image_id>.<ext>     images
    download_manifest.csv                   one row per image (all sets)
When every image of the set has been attempted, the hashes of its usable images are frozen in
``data/versions/<set>/image_integrity.csv``. An existing integrity file is never overwritten with different content.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, log, make_session, rel, setup_logging, sha256_file, utc_now  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "data" / "raw" / "images" / "external_v001"
MIN_SIDE = 128   # configs/dataset_v001.yaml: image_policy.min_side_px (kept and reported, as in v002)
COLUMNS = ["image_uid", "observation_uid", "source_id", "filename", "status", "error", "http_status", "size_bytes",
           "sha256", "width", "height", "mode", "url", "downloaded_utc"]
USABLE = ("ok", "too_small")
HOSTS = {"inaturalist-open-data.s3.amazonaws.com": (8, 0.0), "mushroomobserver.org": (2, 0.5)}
DEFAULT_HOST = (2, 0.5)
MIND = "mind_funga_v2"
MERGE_GAP = 256 * 1024
MAX_CHUNK = 16 * 1024 * 1024
LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")


def verified_images(set_id: str) -> pd.DataFrame:
    ver = REPO_ROOT / "data" / "versions" / set_id
    images_csv = REPO_ROOT / "data" / "processed" / set_id / "images.csv"
    man = json.loads((ver / "manifest.json").read_text(encoding="utf-8"))
    expected = next(o["sha256"] for o in man["outputs"] if o["name"] == "images.csv")
    if sha256_file(images_csv) != expected:
        raise SystemExit(f"{rel(images_csv)} does not match {rel(ver / 'manifest.json')}; rebuild the set")
    return pd.read_csv(images_csv, dtype=str, keep_default_na=False)


def target(row: pd.Series) -> str:
    ext = Path(urlparse(row["locator"]).path).suffix.lower() or ".jpg"
    return f"{row['source_id']}/{row['source_image_id']}{ext}"


def base_record(row: pd.Series, url: str) -> dict:
    return {"image_uid": row["image_uid"], "observation_uid": row["observation_uid"], "source_id": row["source_id"],
            "filename": target(row), "url": url, "downloaded_utc": utc_now(), "error": "", "http_status": "",
            "size_bytes": "", "sha256": "", "width": "", "height": "", "mode": "", "status": ""}


def store(out: dict, body: bytes, root: Path, expected_sha: str = "") -> dict:
    """Decode, check and write one image; fills the manifest record."""
    digest = hashlib.sha256(body).hexdigest()
    if expected_sha and digest != expected_sha.lower():
        out.update(status="hash_mismatch", error=f"sha256 {digest} != published {expected_sha}")
        return out
    try:
        with Image.open(io.BytesIO(body)) as im:
            im.load()
            w, h, mode = im.width, im.height, im.mode
    except Exception as exc:  # noqa: BLE001
        out.update(status="decode_error", error=str(exc)[:200])
        return out
    dest = root / out["filename"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(body)
    part.replace(dest)
    out.update(status="ok" if min(w, h) >= MIN_SIDE else "too_small", size_bytes=len(body), sha256=digest,
               width=w, height=h, mode=mode)
    return out


# --------------------------------------------------------------------------- URL sources
def fetch_url(session, row: pd.Series, root: Path, pause: float) -> dict:
    out = base_record(row, row["locator"])
    try:
        r = session.get(row["locator"], timeout=60)
        out["http_status"] = r.status_code
        if r.status_code != 200:
            out.update(status=f"http_{r.status_code}", error=r.reason or "")
            return out
        return store(out, r.content, root)
    except Exception as exc:  # noqa: BLE001 - recorded per image, the run continues
        out.update(status="error", error=f"{type(exc).__name__}: {exc}"[:200])
        return out
    finally:
        if pause:
            time.sleep(pause)


def run_url_sources(todo: pd.DataFrame, root: Path, record) -> None:
    by_host = {h: g for h, g in todo.groupby(todo["locator"].map(lambda u: urlparse(u).netloc))}

    def run_host(host: str, frame: pd.DataFrame) -> None:
        workers, pause = HOSTS.get(host, DEFAULT_HOST)
        local = threading.local()

        def task(row):
            if not hasattr(local, "s"):
                local.s = make_session()
            return fetch_url(local.s, row, root, pause)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(task, r) for _, r in frame.iterrows()]):
                record(fut.result())

    with ThreadPoolExecutor(max_workers=max(1, len(by_host))) as hosts:
        for f in [hosts.submit(run_host, h, g) for h, g in by_host.items()]:
            f.result()


# --------------------------------------------------------------------------- MIND.Funga (members of the Mendeley ZIP)
def run_mind_funga(todo: pd.DataFrame, root: Path, record) -> None:
    from acquire_datasets import HttpRangeFile, MENDELEY_API, MENDELEY_ID, MENDELEY_VERSION, curl_range
    url = f"{MENDELEY_API}/zip/{MENDELEY_ID}/download/{MENDELEY_VERSION}"
    fetch = lambda a, b: curl_range(url, a, b)  # noqa: E731
    rf = HttpRangeFile(fetch)
    with zipfile.ZipFile(rf) as z:
        infos = z.infolist()
    log.info("MIND.Funga ZIP: %d members, %.2f GB, central directory read with %d requests",
             len(infos), rf.size / 1e9, rf.n_requests)
    by_suffix: dict[str, zipfile.ZipInfo] = {}
    for i in infos:
        by_suffix.setdefault(i.filename.split("/", 1)[-1], i)   # path below the top folder = v001 locator
    jobs = []
    for _, row in todo.iterrows():
        info = by_suffix.get(row["locator"])
        if info is None:
            out = base_record(row, f"{url}#{row['locator']}")
            record(out | {"status": "missing_in_zip", "error": "locator not found in the official ZIP"})
            continue
        jobs.append((info, row))
    jobs.sort(key=lambda j: j[0].header_offset)

    def span(info):   # local header + name + extra (unknown length: slack) + data
        return info.header_offset, info.header_offset + LOCAL_HEADER.size + len(info.filename.encode()) + 1024 + info.compress_size

    chunks, cur = [], []
    for job in jobs:
        s, e = span(job[0])
        if cur and (s - cur_end > MERGE_GAP or e - cur_start > MAX_CHUNK):
            chunks.append((cur_start, cur_end, cur))
            cur = []
        if not cur:
            cur_start = s
            cur_end = e
        cur.append(job)
        cur_end = max(cur_end, e)
    if cur:
        chunks.append((cur_start, cur_end, cur))
    log.info("MIND.Funga: %d images in %d range requests (%.1f MB)", len(jobs), len(chunks),
             sum(e - s for s, e, _ in chunks) / 1e6)

    def member(buf: bytes, base: int, info: zipfile.ZipInfo) -> bytes:
        o = info.header_offset - base
        sig, _, _, _, _, _, _, _, _, nlen, xlen = LOCAL_HEADER.unpack_from(buf, o)
        if sig != b"PK\x03\x04":
            raise ValueError("bad local header signature")
        start = o + LOCAL_HEADER.size + nlen + xlen
        data = buf[start:start + info.compress_size]
        if len(data) < info.compress_size:      # local extra field longer than the slack: fetch the rest
            more, _ = fetch(base + start + len(data), base + start + info.compress_size - 1)
            data += more
        if info.compress_type == zipfile.ZIP_STORED:
            raw = data
        elif info.compress_type == zipfile.ZIP_DEFLATED:
            raw = zlib.decompress(data, -15)
        else:
            raise ValueError(f"unsupported compression {info.compress_type}")
        if zlib.crc32(raw) & 0xFFFFFFFF != info.CRC:
            raise ValueError("CRC-32 mismatch")
        return raw

    def do_chunk(chunk) -> list[dict]:
        s, e, members = chunk
        results = []
        try:
            buf, _ = fetch(s, e - 1)
        except Exception as exc:  # noqa: BLE001
            for info, row in members:
                results.append(base_record(row, f"{url}#{row['locator']}")
                               | {"status": "error", "error": f"range {s}-{e}: {exc}"[:200]})
            return results
        for info, row in members:
            out = base_record(row, f"{url}#{row['locator']}")
            try:
                results.append(store(out, member(buf, s, info), root, row.get("sha256_reported", "")))
            except Exception as exc:  # noqa: BLE001
                results.append(out | {"status": "zip_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
        time.sleep(0.5)
        return results

    with ThreadPoolExecutor(max_workers=2) as ex:     # a few polite parallel ranges to Mendeley
        for fut in as_completed([ex.submit(do_chunk, c) for c in chunks]):
            for r in fut.result():
                record(r)


# --------------------------------------------------------------------------- manifest + freeze
def write_manifest(path: Path, rows: dict[str, dict]) -> None:
    df = pd.DataFrame(list(rows.values()), columns=COLUMNS).sort_values("image_uid")
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def freeze(set_id: str, rows: dict[str, dict], uids: set[str]) -> None:
    df = pd.DataFrame([rows[u] for u in uids if u in rows], columns=COLUMNS)
    if len(df) != len(uids):
        log.info("integrity of %s not frozen: %d of %d images attempted", set_id, len(df), len(uids))
        return
    ok = df[df["status"].isin(USABLE)][["image_uid", "sha256", "width", "height", "status"]].sort_values("image_uid")
    body = ok.to_csv(index=False, lineterminator="\n").encode("utf-8")
    dest = REPO_ROOT / "data" / "versions" / set_id / "image_integrity.csv"
    if dest.exists() and dest.read_bytes() != body:
        raise SystemExit(f"{rel(dest)} already frozen with different content; downloaded files changed. "
                         "Inspect download_manifest.csv; do not overwrite a published set.")
    dest.write_bytes(body)
    log.info("frozen %d usable image hashes of %d in %s", len(ok), len(uids), rel(dest))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="external set id, e.g. ext_co_001 or ext_full_001")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=None, help="only the first N images of each source (trial)")
    ap.add_argument("--retry-failed", action="store_true", help="retry images whose previous attempt failed")
    args = ap.parse_args()
    setup_logging()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "download_manifest.csv"
    img = verified_images(args.set)
    rows: dict[str, dict] = {}
    if manifest_path.exists():
        for r in pd.read_csv(manifest_path, dtype=str, keep_default_na=False).to_dict("records"):
            path = root / r["filename"]
            if r["status"] in USABLE and (not path.is_file() or sha256_file(path) != r["sha256"]):
                continue                                   # file gone or changed: download again
            if r["status"] not in USABLE and args.retry_failed:
                continue
            rows[r["image_uid"]] = r
    todo = img[~img["image_uid"].isin(rows)]
    if args.limit:
        todo = todo.groupby("source_id", group_keys=False).head(args.limit)
    log.info("%s: %d images, %d already done, %d to fetch", args.set, len(img), int(img["image_uid"].isin(rows).sum()),
             len(todo))

    lock, done = threading.Lock(), [0]

    def record(res: dict) -> None:
        with lock:
            rows[res["image_uid"]] = res
            done[0] += 1
            if done[0] % 250 == 0:
                write_manifest(manifest_path, rows)
                log.info("%d/%d fetched", done[0], len(todo))

    mind = todo[todo["source_id"] == MIND]
    urls = todo[todo["source_id"] != MIND]
    if len(urls):
        run_url_sources(urls, root, record)
    if len(mind):
        run_mind_funga(mind, root, record)
    write_manifest(manifest_path, rows)
    df = pd.DataFrame([rows[u] for u in img["image_uid"] if u in rows], columns=COLUMNS)
    print(df.groupby(["source_id", "status"]).size().to_string())
    if not args.limit:
        freeze(args.set, rows, set(img["image_uid"]))
    print(rel(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
