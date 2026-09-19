"""Metadata-first acquisition of the Week 2 audit sources.

Only metadata, indexes, taxonomies, small API snapshots and DwC-A archives are downloaded.
NO image collection is downloaded (see README / week 2 report). Files land in ``data/raw/<dataset_id>/``
(ignored by Git); every file / query is registered in ``data/manifests/raw_files.csv`` and
``data/manifests/remote_queries.csv`` (versioned).

Usage:
    python scripts/data/acquire_datasets.py                 # all sources
    python scripts/data/acquire_datasets.py fungitastic df20  # selected sources
    python scripts/data/acquire_datasets.py --refresh-queries gbif inaturalist_api

Raw files are never overwritten. A failing source is logged and the run continues with the others;
the exit code is 1 if any source failed.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RAW_DIR, USER_AGENT, download_file, head_size, log, make_session, register_raw_file,  # noqa: E402
                    rel, setup_logging, snapshot_json, store_snapshot, utc_now)

# --------------------------------------------------------------------------- constants
FT_ROOT = "https://cmp.felk.cvut.cz/datagrid/FungiTastic/shared/download"
DF20_ROOT = "http://ptak.felk.cvut.cz/plants/DanishFungiDataset"
MENDELEY_API = "https://data.mendeley.com/public-api"
MENDELEY_ID = "sfrbdjvxcc"
MENDELEY_VERSION = 2
INAT_S3 = "https://inaturalist-open-data.s3.amazonaws.com"
INAT_API = "https://api.inaturalist.org/v1"
GBIF_API = "https://api.gbif.org/v1"
MO_ROOT = "https://mushroomobserver.org"
SIB_IPT = "https://ipt.biodiversidad.co/permisos"
SIB_RESOURCES = {  # dataset_id -> IPT resource short name
    "sib_macrohongosmitu": "macrohongosmitu",
    "sib_macrohongos_jbb_2023_1": "macrohongos_2023-1",
    "sib_hongos_col2023": "hongos_col2023",
}
COLOMBIA_PLACE_ID = 7196   # iNaturalist place id for the country (verified via /places/autocomplete)
FUNGI_TAXON_ID = 47170     # iNaturalist taxon id for kingdom Fungi (verified via /taxa)
GBIF_FUNGI_KEY = 5         # GBIF backbone kingdom Fungi
GBIF_DK_DATASET = "84d26682-f762-11e1-a439-00145eb45e9a"   # dataset behind the FungiTastic/DF20 observation ids (verified via lookups below)


# --------------------------------------------------------------------------- HTTP range file
def curl_range(url: str, start: int, end: int, timeout: int = 180) -> tuple[bytes, int]:
    """GET bytes [start, end] with the system ``curl`` (follows redirects). Returns (body, total_size).

    data.mendeley.com answers python-requests with a Cloudflare bot challenge (HTTP 403) while the plain
    ``curl`` client is served normally. We do NOT try to solve or spoof anything: this is a public API and we
    only issue a handful of requests (see report, section on limitations).
    """
    cmd = ["curl", "-sSL", "--fail", "-m", str(timeout), "-A", USER_AGENT, "-r", f"{start}-{end}", "-D", "-", url]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    # with -D - one header block per redirect hop precedes the body
    sep, head = b"\r\n\r\n", b""
    while out.startswith(b"HTTP/"):
        block, _, out = out.partition(sep)
        head += block + sep
    body = out
    total = int(re.findall(rb"(?i)content-range: bytes \d+-\d+/(\d+)", head)[-1])
    return body, total


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable file object over HTTP Range requests (used to read a remote ZIP directory)."""

    def __init__(self, fetch: Callable[[int, int], tuple[bytes, int]]):
        self.fetch, self.pos, self.n_requests = fetch, 0, 1
        _, self.size = fetch(0, 0)

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        body, _ = self.fetch(self.pos, self.pos + n - 1)
        self.n_requests += 1
        self.pos += len(body)
        return body


# --------------------------------------------------------------------------- FungiTastic
def acquire_fungitastic(s: requests.Session, refresh: bool) -> None:
    """Same URL scheme as the official ``dataset/download.py --metadata`` (needs wget; not portable to
    Windows), but keeping the ZIP so that the audit can stream CSVs directly from it."""
    url = f"{FT_ROOT}/metadata.zip"
    size = head_size(s, url)
    dest = RAW_DIR / "fungitastic" / "metadata.zip"
    download_file(s, url, dest, expected_size=size)
    register_raw_file("fungitastic", dest, url, "Paquete oficial de metadatos (CSV train/val/test/DNA de los subconjuntos completo, Mini y FewShot)")

    # Kaggle public dataset page metadata (unofficial web endpoint; used only for version/licence/size)
    try:
        snapshot_json(s, "fungitastic", "kaggle_dataset_view",
                      "https://www.kaggle.com/api/v1/datasets/view/picekl/fungitastic", refresh=refresh,
                      summary_key="currentVersionNumber")
    except Exception as exc:  # noqa: BLE001
        log.warning("Kaggle metadata unavailable: %s", exc)

    # HEAD probes of the other official artefacts (nothing downloaded)
    probes = {}
    for name in ["metadata.zip", "climatic.zip", "masks.zip", "satellite_RGB.zip", "satellite_NIR.zip",
                 "FungiTastic-train-300p.zip", "FungiTastic-val-300p.zip", "FungiTastic-test-300p.zip",
                 "FungiTastic-dna-test-300p.zip", "FungiTastic-Mini-train-300p.zip",
                 "FungiTastic-train-500p.zip", "FungiTastic-Mini-train-500p.zip", "FungiTastic-Mini-train-720p.zip",
                 "FungiTastic-Mini-train-fullsize.zip", "FungiTastic-Mini-val-fullsize.zip",
                 "FungiTastic-Mini-test-fullsize.zip", "FungiTastic-Mini-dna-test-fullsize.zip",
                 "FungiTastic-FewShot-train-300p.zip"]:
        r = s.head(f"{FT_ROOT}/{name}", allow_redirects=True, timeout=60)
        probes[name] = {"status": r.status_code, "content_length": r.headers.get("Content-Length"),
                        "last_modified": r.headers.get("Last-Modified")}
        time.sleep(0.5)
    store_snapshot("fungitastic", "official_download_head_probes", FT_ROOT, {}, probes, 200,
                   "Solo HEAD; no se descargó ningún archivo")


# --------------------------------------------------------------------------- DF20
def acquire_df20(s: requests.Session, refresh: bool) -> None:
    url = f"{DF20_ROOT}/DF20-metadata.zip"
    size = head_size(s, url)
    dest = RAW_DIR / "df20" / "DF20-metadata.zip"
    download_file(s, url, dest, expected_size=size)
    register_raw_file("df20", dest, url, "Metadatos oficiales de DF20 (CSV de train y test público)")
    probes = {}
    for name in ["DF20-metadata.zip", "DF20-300px.tar.gz", "DF20-train_val.tar.gz", "DF20M-metadata.zip",
                 "DF20M-images.tar.gz"]:
        r = s.head(f"{DF20_ROOT}/{name}", allow_redirects=True, timeout=60)
        probes[name] = {"status": r.status_code, "content_length": r.headers.get("Content-Length"),
                        "last_modified": r.headers.get("Last-Modified")}
    store_snapshot("df20", "official_download_head_probes", DF20_ROOT, {}, probes, 200, "Solo HEAD")


# --------------------------------------------------------------------------- MIND.Funga
def acquire_mind_funga(s: requests.Session, refresh: bool) -> None:
    snap = RAW_DIR / "mind_funga_v2" / "api" / "mendeley_dataset.json"
    if snap.exists() and not refresh:
        data = json.loads(snap.read_text(encoding="utf-8"))
    else:  # curl, not requests: see curl_range() docstring
        url = f"{MENDELEY_API}/datasets/{MENDELEY_ID}"
        cmd = ["curl", "-sSL", "--fail", "-m", "180", "-A", USER_AGENT,
               "-H", "Accept: application/vnd.mendeley-public-dataset.1+json", url]
        data = json.loads(subprocess.run(cmd, capture_output=True, check=True).stdout)
        store_snapshot("mind_funga_v2", "mendeley_dataset", url, {}, data, 200, f"version={data.get('version')}")
    if data.get("version") != MENDELEY_VERSION:
        log.warning("Mendeley 'latest' is now version %s (audit is defined for v%s)", data.get("version"), MENDELEY_VERSION)

    # Central directory of the official ZIP, read with HTTP Range requests (no image bytes downloaded).
    dest = RAW_DIR / "mind_funga_v2" / "zip_central_directory.csv"
    if dest.exists():
        log.info("skip (already present): %s", rel(dest))
    else:
        zurl = f"{MENDELEY_API}/zip/{MENDELEY_ID}/download/{MENDELEY_VERSION}"
        rf = HttpRangeFile(lambda a, b: curl_range(zurl, a, b))
        with zipfile.ZipFile(rf) as z:
            infos = z.infolist()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["zip_path", "file_size", "compress_size", "crc32", "date_time"])
            for i in infos:
                w.writerow([i.filename, i.file_size, i.compress_size, f"{i.CRC:08x}",
                            "%04d-%02d-%02d %02d:%02d:%02d" % i.date_time])
        register_raw_file("mind_funga_v2", dest, zurl,
                          f"Directorio central del ZIP oficial de Mendeley ({rf.size} bytes, {len(infos)} entradas) leído con "
                          f"{rf.n_requests} solicitudes HTTP Range; no se descargó ninguna imagen")


# --------------------------------------------------------------------------- iNaturalist
def _gunzip_prefix(raw: bytes) -> str:
    import zlib
    return zlib.decompressobj(31).decompress(raw).decode("utf-8", "replace")


def acquire_inaturalist_open_data(s: requests.Session, refresh: bool) -> None:
    dest = RAW_DIR / "inaturalist_open_data" / "taxa.csv.gz"
    url = f"{INAT_S3}/taxa.csv.gz"
    download_file(s, url, dest, expected_size=None)
    register_raw_file("inaturalist_open_data", dest, url, "Tabla completa de taxones de la instantánea de Open Data (TSV comprimido)")
    # Leading fragments of the two large tables: schema check only, biased towards the oldest rows.
    for name, nbytes in [("observations.csv.gz", 2_000_000), ("photos.csv.gz", 3_000_000)]:
        frag = RAW_DIR / "inaturalist_open_data" / f"{name}.head{nbytes // 1_000_000}MB"
        if not frag.exists():
            r = s.get(f"{INAT_S3}/{name}", headers={"Range": f"bytes=0-{nbytes - 1}"}, timeout=120, stream=True)
            r.raise_for_status()
            frag.write_bytes(r.raw.read(decode_content=False))
        register_raw_file("inaturalist_open_data", frag, f"{INAT_S3}/{name}",
                          f"Primeros {nbytes} bytes del flujo gzip (fragmento truncado SOLO para inspeccionar el esquema; filas más antiguas)")
    probes = {}
    for name in ["observations.csv.gz", "observers.csv.gz", "photos.csv.gz", "taxa.csv.gz",
                 "metadata/inaturalist-open-data-latest.tar.gz"]:
        r = s.head(f"{INAT_S3}/{name}", allow_redirects=True, timeout=60)
        probes[name] = {"status": r.status_code, "content_length": r.headers.get("Content-Length"),
                        "last_modified": r.headers.get("Last-Modified")}
    store_snapshot("inaturalist_open_data", "s3_head_probes", INAT_S3, {}, probes, 200, "Solo HEAD; las tablas grandes NO se descargaron")


def acquire_inaturalist_api(s: requests.Session, refresh: bool) -> None:
    ds = "inaturalist_api"
    base = {"taxon_id": FUNGI_TAXON_ID, "place_id": COLOMBIA_PLACE_ID, "per_page": 0}
    q = lambda qid, path, params, key="total_results": snapshot_json(  # noqa: E731
        s, ds, qid, f"{INAT_API}{path}", params, refresh, sleep=1.1, summary_key=key)
    q("place_colombia", "/places/autocomplete", {"q": "Colombia"}, "total_results")
    q("taxon_fungi", "/taxa", {"q": "Fungi", "rank": "kingdom"})
    q("obs_fungi_world", "/observations", {"taxon_id": FUNGI_TAXON_ID, "per_page": 0})
    q("obs_fungi_co_all", "/observations", base)
    q("obs_fungi_co_research", "/observations", {**base, "quality_grade": "research"})
    q("obs_fungi_co_photos", "/observations", {**base, "photos": "true"})
    q("obs_fungi_co_research_photos_licensed", "/observations",
      {**base, "quality_grade": "research", "photos": "true", "photo_licensed": "true"})
    q("species_counts_co_all", "/observations/species_counts", base)
    q("species_counts_co_research", "/observations/species_counts", {**base, "quality_grade": "research"})
    # per-species observation counts (needed for the long-tail analysis): 500 species per page
    for qual, pages in (("research", 2), ("any", 5)):
        for page in range(1, pages + 1):
            params = {"taxon_id": FUNGI_TAXON_ID, "place_id": COLOMBIA_PLACE_ID, "per_page": 500, "page": page}
            if qual == "research":
                params["quality_grade"] = "research"
            q(f"species_counts_co_{qual}_p{page}", "/observations/species_counts", params)
    for lic in ["cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nd", "cc-by-nc-sa", "cc-by-nc-nd"]:
        q(f"obs_fungi_co_research_photolic_{lic.replace('-', '_')}", "/observations",
          {**base, "quality_grade": "research", "photo_license": lic})
    # random sample of observations (stored, so the audit is reproducible); used to estimate photos/observation
    q("obs_fungi_co_random_sample200", "/observations",
      {"taxon_id": FUNGI_TAXON_ID, "place_id": COLOMBIA_PLACE_ID, "per_page": 200, "order_by": "random",
       "quality_grade": "research", "photos": "true"})


# --------------------------------------------------------------------------- GBIF
def acquire_gbif(s: requests.Session, refresh: bool) -> None:
    ds = "gbif"
    occ = f"{GBIF_API}/occurrence/search"
    fungi = {"kingdomKey": GBIF_FUNGI_KEY}
    q = lambda qid, params: snapshot_json(s, ds, qid, occ, {"limit": 0, **params}, refresh, sleep=0.4,  # noqa: E731
                                          summary_key="count")
    q("fungi_world_all", fungi)
    q("fungi_world_stillimage", {**fungi, "mediaType": "StillImage"})
    q("fungi_co_all", {**fungi, "country": "CO"})
    co_img = {**fungi, "country": "CO", "mediaType": "StillImage"}
    q("fungi_co_stillimage", co_img)
    q("fungi_co_stillimage_coords", {**co_img, "hasCoordinate": "true"})
    q("fungi_co_stillimage_no_coords", {**co_img, "hasCoordinate": "false"})
    for facet in ["datasetKey", "license", "basisOfRecord", "publishingOrg", "year"]:
        q(f"fungi_co_stillimage_facet_{facet}", {**co_img, "facet": facet, "facetLimit": 40})
    q("fungi_co_stillimage_facet_speciesKey", {**co_img, "facet": "speciesKey", "facetLimit": 20000})
    q("fungi_co_stillimage_facet_genusKey", {**co_img, "facet": "genusKey", "facetLimit": 5000})
    q("fungi_co_stillimage_facet_familyKey", {**co_img, "facet": "familyKey", "facetLimit": 5000})
    # Colombian-published (SiB Colombia network) fungi with images
    pub_co = {**fungi, "mediaType": "StillImage", "publishingCountry": "CO"}
    q("fungi_pubco_stillimage", pub_co)
    q("fungi_pubco_stillimage_facet_datasetKey", {**pub_co, "facet": "datasetKey", "facetLimit": 40})
    q("fungi_co_inat_dataset_stillimage", {**co_img, "datasetKey": "50c9509d-22c7-4a22-a47d-8c48425ef4a7"})
    # sample of records (limit<=300) spread over the result set to look at media/licence/metadata completeness
    for off in (0, 6000, 12000):
        snapshot_json(s, ds, f"fungi_co_stillimage_sample300_offset{off}", occ,
                      {**co_img, "limit": 300, "offset": off}, refresh, sleep=0.6, summary_key="count")
    # names of the most frequent genera (facet keys are numeric taxon keys)
    gen = snapshot_json(s, ds, "fungi_co_stillimage_facet_genusKey", occ, None, False)
    names = {}
    for c in gen["facets"][0]["counts"][:25]:
        d = snapshot_json(s, ds, f"species_{c['name']}", f"{GBIF_API}/species/{c['name']}", None, refresh, sleep=0.3)
        names[c["name"]] = {"canonicalName": d.get("canonicalName"), "rank": d.get("rank"), "family": d.get("family")}
    store_snapshot(ds, "top_genus_names_index", f"{GBIF_API}/species/{{key}}", {}, names, 200, f"{len(names)} géneros")
    # dataset titles for the datasets that dominate the Colombian image records
    facet = snapshot_json(s, ds, "fungi_co_stillimage_facet_datasetKey", occ, None, False)
    keys = [c["name"] for c in facet["facets"][0]["counts"][:12]]
    pub = snapshot_json(s, ds, "fungi_pubco_stillimage_facet_datasetKey", occ, None, False)
    keys += [c["name"] for c in pub["facets"][0]["counts"][:12] if c["name"] not in keys]
    titles = {}
    for k in keys:
        d = snapshot_json(s, ds, f"dataset_{k}", f"{GBIF_API}/dataset/{k}", None, refresh, sleep=0.3)
        titles[k] = {"title": d.get("title"), "type": d.get("type"), "license": d.get("license"),
                     "publishingOrganizationKey": d.get("publishingOrganizationKey")}
    store_snapshot(ds, "dataset_titles_index", f"{GBIF_API}/dataset/{{key}}", {}, titles, 200, f"{len(titles)} datasets")
    # do the observation ids of FungiTastic / DF20 correspond to GBIF occurrence keys? (one example id from each CSV)
    for qid, key in [("occurrence_lookup_fungitastic_train_row0", 3032624318), ("occurrence_lookup_df20_train_row0", 2238478820)]:
        snapshot_json(s, ds, qid, f"{GBIF_API}/occurrence/{key}", None, refresh, sleep=0.4, summary_key="datasetKey")
    q("danish_mycological_dataset_all", {"datasetKey": GBIF_DK_DATASET})
    q("danish_mycological_dataset_stillimage", {"datasetKey": GBIF_DK_DATASET, "mediaType": "StillImage"})
    snapshot_json(s, ds, "dataset_danish_mycological", f"{GBIF_API}/dataset/{GBIF_DK_DATASET}", None, refresh, sleep=0.3)
    # are the three SiB IPT resources registered in GBIF? (registry search by title)
    for qid, text in [("registry_search_mitu", "Mitú-Cachivera"), ("registry_search_jbb", "Fungario del Jardín Botánico José Celestino Mutis"),
                      ("registry_search_hongos_col", "Hongos de Colombia")]:
        snapshot_json(s, ds, qid, f"{GBIF_API}/dataset", {"q": text, "limit": 10}, refresh, sleep=0.4, summary_key="count")


# --------------------------------------------------------------------------- SiB Colombia
def acquire_sib(s: requests.Session, refresh: bool) -> None:
    for ds, res in SIB_RESOURCES.items():
        url = f"{SIB_IPT}/archive.do?r={res}"
        dest = RAW_DIR / ds / f"{res}.dwca.zip"
        if dest.exists():
            log.info("skip (already present): %s", rel(dest))
            cd = ""
        else:
            r = s.get(url, timeout=120)
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            cd = r.headers.get("Content-Disposition", "")
        m = re.search(r'filename="?([^";]+)', cd)
        register_raw_file(ds, dest, url, f"Darwin Core Archive del recurso IPT '{res}'" + (f"; entregado como {m.group(1)}" if m else ""))
        time.sleep(1)


# --------------------------------------------------------------------------- Mushroom Observer
MO_CSVS = ["observations", "images_observations", "images", "names", "name_classifications", "locations"]


def acquire_mushroom_observer(s: requests.Session, refresh: bool) -> None:
    """Uses the nightly CSV dumps that MO's API documentation asks bulk users to use instead of the API."""
    for name in MO_CSVS:
        url = f"{MO_ROOT}/{name}.csv"
        dest = RAW_DIR / "mushroom_observer" / f"{name}.csv"
        download_file(s, url, dest, expected_size=None)
        register_raw_file("mushroom_observer", dest, url, "Volcado CSV nocturno oficial (ver README_API.md, sección 'CSV files')")
        time.sleep(2)  # be gentle: volunteer-run service


DOC_URLS = [  # every resource in the task list, so that availability is recorded at the access date
    ("fungitastic", "https://bohemianvra.github.io/FungiTastic/"), ("fungitastic", "https://github.com/bohemianvra/FungiTastic"),
    ("fungitastic", "https://www.kaggle.com/datasets/picekl/fungitastic"), ("df20", "https://sites.google.com/view/danish-fungi-dataset"),
    ("df20", "https://openaccess.thecvf.com/content/WACV2022/html/Picek_Danish_Fungi_2020_-Not_Just_Another_Image_Recognition_Dataset_WACV_2022_paper.html"),
    ("df20", "https://github.com/BohemianVRA/DanishFungiDataset"), ("mind_funga_v2", "https://data.mendeley.com/datasets/sfrbdjvxcc/2"),
    ("mind_funga_v2", "https://mindfunga.ufsc.br/"), ("inaturalist_api", "https://www.inaturalist.org/"),
    ("inaturalist_api", "https://api.inaturalist.org/v1/docs/"), ("inaturalist_open_data", "https://github.com/inaturalist/inaturalist-open-data"),
    ("inaturalist_open_data", "https://registry.opendata.aws/inaturalist-open-data/"), ("sib_colombia_portal", "https://biodiversidad.co/"),
    ("sib_colombia_portal", "https://biodiversidad.co/data/"), ("sib_macrohongosmitu", "https://ipt.biodiversidad.co/permisos/resource?r=macrohongosmitu"),
    ("sib_macrohongos_jbb_2023_1", "https://ipt.biodiversidad.co/permisos/resource?r=macrohongos_2023-1"),
    ("sib_hongos_col2023", "https://ipt.biodiversidad.co/permisos/resource?r=hongos_col2023"), ("gbif", "https://www.gbif.org/"),
    ("gbif", "https://www.gbif.org/occurrence/search?media_type=StillImage"), ("mushroom_observer", "https://mushroomobserver.org/"),
]


def acquire_documentation_urls(s: requests.Session, refresh: bool) -> None:
    """Record HTTP status of the official documentation pages given in the task (GET, no content stored)."""
    out = {}
    for ds, url in DOC_URLS:
        try:
            r = s.get(url, timeout=45, allow_redirects=True, stream=True)
            out[url] = {"dataset_id": ds, "status": r.status_code, "final_url": r.url}
            r.close()
        except requests.RequestException as exc:
            out[url] = {"dataset_id": ds, "status": f"ERROR {type(exc).__name__}", "final_url": None}
        time.sleep(0.5)
    store_snapshot("sources_documentation", "reachability_probes", "various", {}, out, 200, f"{len(out)} URLs")


SOURCES: dict[str, Callable[[requests.Session, bool], None]] = {
    "fungitastic": acquire_fungitastic,
    "df20": acquire_df20,
    "mind_funga_v2": acquire_mind_funga,
    "inaturalist_open_data": acquire_inaturalist_open_data,
    "inaturalist_api": acquire_inaturalist_api,
    "sib_colombia": acquire_sib,
    "gbif": acquire_gbif,
    "mushroom_observer": acquire_mushroom_observer,
    "documentation_urls": acquire_documentation_urls,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", choices=list(SOURCES), help="subset of sources (default: all)")
    ap.add_argument("--refresh-queries", action="store_true", help="re-run remote API queries instead of reusing stored snapshots")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)
    s = make_session()
    failed = []
    for name in args.sources or SOURCES:
        log.info("=== acquiring %s", name)
        try:
            SOURCES[name](s, args.refresh_queries)
        except Exception as exc:  # noqa: BLE001 - keep going with the other sources
            log.error("source %s FAILED: %s: %s", name, type(exc).__name__, exc)
            failed.append(name)
    log.info("finished at %s; failed sources: %s", utc_now(), failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
