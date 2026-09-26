"""Download the FungiTastic images of a processed dataset version (one command, resumable).

Usage:
    python scripts/data/download_images.py                 # all FungiTastic images of v001 (~44k, ~1 GB)
    python scripts/data/download_images.py --dry-run       # only read the ZIP indexes and report what would be downloaded
    python scripts/data/download_images.py --limit 50      # trial run: first 50 images per split
    python scripts/data/download_images.py --out D:/virtus_images   # store outside the repository (recommended with OneDrive)

How it works:
  The official archives (FungiTastic-{train,val,test,dna-test}-300p.zip, 14.5 GB in total) contain every FungiTastic
  image, but v001 needs only ~7 % of them. The server supports HTTP Range requests, so the script reads the central
  directory of each ZIP (a few MB, cached locally) and then fetches only the byte ranges of the needed members,
  merging neighbouring members into a single request. Each image is decompressed, checked against the CRC-32 stored
  in the ZIP, decoded with Pillow and recorded in a manifest. Nothing in data/processed or data/versions is modified.

Outputs (under --out, default data/raw/images/fungitastic, ignored by Git):
    300p/<filename>                 images, exactly as stored in the official ZIPs
    download_manifest.csv           one row per image: status, size, sha256, crc, width, height, mode, phash64
    download_summary.json           counts by status and split, duplicates, run metadata
    _zip_index/<zip>.csv            cached central directories

Only FungiTastic is handled here (the development source of v001: train/val/test). The external test sources
(iNaturalist, Mushroom Observer, MIND.Funga) are left for a later phase; their URLs are in images.csv (`locator`).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import struct
import sys
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, log, make_session, rel, setup_logging, utc_now  # noqa: E402

FT_ROOT = "https://cmp.felk.cvut.cz/datagrid/FungiTastic/shared/download"
RESOLUTION = "300p"
# official CSV (first part of `locator` in images.csv) -> official image archive
CSV_TO_ZIP = {"Train": "train", "ClosedSet-Val": "val", "OpenSet-Val": "val", "ClosedSet-Test": "test",
              "OpenSet-Test": "test", "DNA-Test": "dna-test"}
MANIFEST_COLUMNS = ["image_uid", "filename", "split", "zip", "member", "status", "error", "size_bytes", "sha256",
                    "crc32_ok", "width", "height", "mode", "phash64", "downloaded_utc"]
MERGE_GAP = 64 * 1024          # merge members whose byte ranges are closer than this
MAX_CHUNK = 8 * 1024 * 1024    # but never request more than this in one go
MIN_SIDE = 128                 # configs/dataset_v001.yaml: image_policy.min_side_px


# --------------------------------------------------------------------------- remote ZIP index
class RangeFile:
    """Minimal read-only file object over HTTP Range requests (enough for zipfile to read the central directory)."""

    def __init__(self, session: requests.Session, url: str):
        self.s, self.url, self.pos = session, url, 0
        r = session.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        data = get_range(self.s, self.url, self.pos, end - 1)
        self.pos += len(data)
        return data


def get_range(s: requests.Session, url: str, start: int, end: int, tries: int = 5) -> bytes:
    for k in range(tries):
        try:
            r = s.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            if r.status_code == 206 and len(r.content) == end - start + 1:
                return r.content
            raise IOError(f"HTTP {r.status_code}, {len(r.content)} bytes for range {start}-{end}")
        except Exception as exc:  # noqa: BLE001
            if k == tries - 1:
                raise
            wait = 2 ** k
            log.warning("range %s-%s of %s failed (%s); retry in %ss", start, end, url.rsplit("/", 1)[-1], exc, wait)
            time.sleep(wait)
    raise AssertionError


def zip_index(s: requests.Session, zip_name: str, cache_dir: Path) -> pd.DataFrame:
    cache = cache_dir / f"{zip_name}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    url = f"{FT_ROOT}/{zip_name}"
    log.info("reading central directory of %s", zip_name)
    z = zipfile.ZipFile(RangeFile(s, url))
    rows = [{"member": i.filename, "filename": i.filename.rsplit("/", 1)[-1], "header_offset": i.header_offset,
             "compress_size": i.compress_size, "file_size": i.file_size, "compress_type": i.compress_type,
             "crc32": i.CRC} for i in z.infolist() if not i.is_dir()]
    df = pd.DataFrame(rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


# --------------------------------------------------------------------------- image checks
def phash64(img) -> str:
    """64-bit perceptual hash (DCT of a 32x32 grayscale image, top-left 8x8 block vs. its median)."""
    g = np.asarray(img.convert("L").resize((32, 32)), dtype=np.float64)
    n = 32
    k = np.arange(n)
    dct = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    block = (dct @ g @ dct.T)[:8, :8].flatten()
    bits = block > np.median(block[1:])
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def inspect_image(data: bytes) -> dict[str, Any]:
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        w, h, mode = im.width, im.height, im.mode
        ph = phash64(im)
    return {"width": w, "height": h, "mode": mode, "phash64": ph}


def extract_member(chunk: bytes, chunk_start: int, row: pd.Series) -> bytes:
    """Decompress one member from a downloaded byte range that contains its local header."""
    off = int(row["header_offset"]) - chunk_start
    sig, *_ = struct.unpack("<I", chunk[off:off + 4])
    if sig != 0x04034B50:
        raise IOError("bad local header signature")
    name_len, extra_len = struct.unpack("<HH", chunk[off + 26:off + 30])
    start = off + 30 + name_len + extra_len
    raw = chunk[start:start + int(row["compress_size"])]
    if len(raw) != int(row["compress_size"]):
        raise IOError("truncated member data")
    if int(row["compress_type"]) == zipfile.ZIP_STORED:
        return raw
    if int(row["compress_type"]) == zipfile.ZIP_DEFLATED:
        return zlib.decompress(raw, -15)
    raise IOError(f"unsupported compression {row['compress_type']}")


# --------------------------------------------------------------------------- planning
def plan_chunks(rows: pd.DataFrame) -> list[tuple[int, int, pd.DataFrame]]:
    """Group members (sorted by offset) into byte ranges; the end includes local header slack (name + extra)."""
    rows = rows.sort_values("header_offset")
    chunks, cur, c_start, c_end = [], [], None, None
    for _, r in rows.iterrows():
        s0 = int(r["header_offset"])
        e0 = s0 + 30 + len(str(r["member"]).encode()) + 256 + int(r["compress_size"])   # 256 B for the extra field
        if cur and (s0 - c_end <= MERGE_GAP) and (e0 - c_start <= MAX_CHUNK):
            cur.append(r)
            c_end = max(c_end, e0)
        else:
            if cur:
                chunks.append((c_start, c_end, pd.DataFrame(cur)))
            cur, c_start, c_end = [r], s0, e0
    if cur:
        chunks.append((c_start, c_end, pd.DataFrame(cur)))
    return chunks


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v001", help="processed dataset version (reads data/processed/<version>/images.csv)")
    ap.add_argument("--out", default=None, help="output directory (default data/raw/images/fungitastic)")
    ap.add_argument("--workers", type=int, default=4, help="parallel HTTP requests (be gentle with the academic server)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N images of each split (trial run)")
    ap.add_argument("--dry-run", action="store_true", help="read the ZIP indexes and report the plan; download nothing")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    images_csv = REPO_ROOT / "data" / "processed" / args.version / "images.csv"
    if not images_csv.exists():
        log.error("%s not found: run `python scripts/data/build_dataset.py` first", rel(images_csv))
        return 1
    out = Path(args.out) if args.out else REPO_ROOT / "data" / "raw" / "images" / "fungitastic"
    out = out.resolve()
    img_dir, idx_dir = out / RESOLUTION, out / "_zip_index"
    if "onedrive" in str(out).lower():
        log.warning("the output folder is inside OneDrive (%s): ~1 GB will be synchronised. "
                    "Consider --out outside OneDrive (and set image_root accordingly in the training config).", out)

    img = pd.read_csv(images_csv, low_memory=False)
    ft = img[img["source_id"] == "fungitastic"].copy()
    ft["csv_file"] = ft["locator"].str.split("/", n=1).str[0]
    ft["zip"] = "FungiTastic-" + ft["csv_file"].map(CSV_TO_ZIP) + f"-{RESOLUTION}.zip"
    ft["filename"] = ft["source_image_id"]
    if args.limit:
        ft = ft.sort_values("image_uid").groupby("split", group_keys=False).head(args.limit)
    log.info("%d FungiTastic images requested (%s)", len(ft), ft["split"].value_counts().to_dict())

    s = make_session()
    indexes = {z: zip_index(s, z, idx_dir) for z in sorted(ft["zip"].unique())}
    zip_sizes = {z: RangeFile(s, f"{FT_ROOT}/{z}").size for z in indexes}
    parts = []
    for z, idx in indexes.items():
        want = ft[ft["zip"] == z][["image_uid", "filename", "split", "zip"]]
        parts.append(want.merge(idx, on="filename", how="left"))
    plan = pd.concat(parts, ignore_index=True)
    if plan["image_uid"].duplicated().any():
        log.error("some filenames appear more than once inside a ZIP; inspect %s", rel(idx_dir))
        return 1
    not_found = plan["member"].isna()
    total_bytes = int(plan.loc[~not_found, "compress_size"].sum())
    log.info("found in the ZIPs: %d of %d images; %d missing; %.2f GB to download",
             int((~not_found).sum()), len(plan), int(not_found.sum()), total_bytes / 1e9)

    # resume: keep previous manifest rows whose file is still present
    man_path = out / "download_manifest.csv"
    done: dict[str, dict[str, Any]] = {}
    if man_path.exists():
        for r in pd.read_csv(man_path, dtype=str, keep_default_na=False).to_dict("records"):
            if r["status"] in ("ok", "too_small") and (img_dir / r["filename"]).exists():
                done[r["image_uid"]] = r
    todo = plan[~not_found & ~plan["image_uid"].isin(done)]
    log.info("%d already downloaded; %d to download", len(done), len(todo))

    if args.dry_run:
        free = shutil.disk_usage(out.anchor or ".").free / 1e9
        log.info("dry run: %d requests planned (after merging neighbours); free disk %.1f GB",
                 sum(len(plan_chunks(todo[todo["zip"] == z])) for z in todo["zip"].unique()), free)
        return 0

    img_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = dict(done)
    for _, r in plan[not_found].iterrows():
        results[r["image_uid"]] = {"image_uid": r["image_uid"], "filename": r["filename"], "split": r["split"],
                                   "zip": r["zip"], "status": "not_in_zip", "error": "member not found in the official ZIP"}
    lock = threading.Lock()

    def write_manifest() -> None:
        rows = sorted(results.values(), key=lambda r: r["image_uid"])
        tmp = man_path.with_suffix(".csv.part")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        tmp.replace(man_path)

    def work(zip_name: str, start: int, end: int, members: pd.DataFrame) -> list[dict[str, Any]]:
        url = f"{FT_ROOT}/{zip_name}"
        out_rows = []
        try:
            chunk = get_range(s, url, start, min(end, zip_sizes[zip_name]) - 1)
        except Exception as exc:  # noqa: BLE001
            return [{**m, "status": "download_error", "error": str(exc)[:200]} for m in members.to_dict("records")]
        for m in members.to_dict("records"):
            row = {k: m.get(k) for k in ("image_uid", "filename", "split", "zip", "member")}
            try:
                data = extract_member(chunk, start, pd.Series(m))
                crc_ok = (zlib.crc32(data) & 0xFFFFFFFF) == int(m["crc32"])
                info = inspect_image(data)
                status = "ok" if crc_ok else "crc_mismatch"
                if crc_ok and min(info["width"], info["height"]) < MIN_SIDE:
                    status = "too_small"
                if crc_ok:
                    dest = img_dir / m["filename"]
                    part = dest.with_name(dest.name + ".part")
                    part.write_bytes(data)
                    part.replace(dest)
                row.update(status=status, error="", size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                           crc32_ok=crc_ok, downloaded_utc=utc_now(), **info)
            except Exception as exc:  # noqa: BLE001
                row.update(status="decode_error", error=str(exc)[:200])
            out_rows.append(row)
        return out_rows

    jobs = [(z, a, b, m) for z in sorted(todo["zip"].unique()) for a, b, m in plan_chunks(todo[todo["zip"] == z])]
    log.info("downloading %d images in %d requests with %d workers", len(todo), len(jobs), args.workers)
    t0, n_done = time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, *j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            rows = f.result()
            with lock:
                for r in rows:
                    results[r["image_uid"]] = r
                n_done += len(rows)
                if k % 200 == 0 or k == len(futs):
                    write_manifest()
                    rate = n_done / max(time.time() - t0, 1e-9)
                    log.info("%d/%d images (%.0f img/s, ETA %.0f min)", n_done, len(todo), rate,
                             (len(todo) - n_done) / max(rate, 1e-9) / 60)
    write_manifest()

    man = pd.read_csv(man_path, dtype=str, keep_default_na=False)
    ok = man[man["status"] == "ok"]
    dup = ok[ok["sha256"].duplicated(keep=False)]
    summary = {
        "version": args.version, "finished_utc": utc_now(), "output_dir": str(out), "resolution": RESOLUTION,
        "requested": len(plan), "by_status": man["status"].value_counts().to_dict(),
        "ok_by_split": ok["split"].value_counts().to_dict(),
        "exact_duplicate_files": int(len(dup)),
        "exact_duplicate_groups_spanning_splits": int((dup.groupby("sha256")["split"].nunique() > 1).sum()),
        "note": "Near-duplicate analysis (phash64) and any exclusion belong to a new dataset version (v002); "
                "v001 is not modified by this script.",
    }
    (out / "download_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("summary: %s", summary)
    bad = len(man) - int(man["status"].isin(["ok", "too_small"]).sum())
    if bad:
        log.warning("%d images not downloaded correctly; re-run the same command to retry them", bad)
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
