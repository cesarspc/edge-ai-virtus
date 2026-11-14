"""Fetch only a small, explicit validation review sample; verify frozen hashes.

Does not populate a training image manifest or download test photographs.
"""
from pathlib import Path
import json
import zlib

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from common import REPO_ROOT, make_session, sha256_file
from download_images import FT_ROOT, RangeFile, zip_index, get_range, extract_member, plan_chunks


def main():
    root = REPO_ROOT / "data/raw/images/s9_review_v002"
    root.mkdir(parents=True, exist_ok=True)
    output = REPO_ROOT / "reports/s9/visual_review"
    output.mkdir(parents=True, exist_ok=True)
    local = REPO_ROOT / "tmp/s9_review"
    local.mkdir(parents=True, exist_ok=True)
    predictions = REPO_ROOT / "reports/s9/validation/20261030T184438858210Z_s8_visual/observations.csv"
    obs = pd.read_csv(predictions)
    wrong = obs[~obs.correct].sort_values("confidence", ascending=False)
    discordant = obs[obs.distinct_image_predictions.gt(1)].sort_values("view_js_divergence", ascending=False)
    selected = pd.concat([wrong.head(2), discordant.head(2),
                          wrong[wrong.true_genus.eq("Hericium")].head(1),
                          obs[obs.true_genus.eq("Morchella")].head(1)]).drop_duplicates("observation_uid")
    selected.to_csv(output / "selected_observations.csv", index=False, lineterminator="\n")
    wanted = {uid for values in selected.image_uids for uid in values.split("|")}
    integrity = pd.read_csv(REPO_ROOT / "data/versions/v002/image_integrity.csv").set_index("image_uid")
    if not wanted.issubset(integrity.index):
        raise ValueError("review image is not in frozen v002")
    s = make_session()
    archive = "FungiTastic-val-300p.zip"
    url = f"{FT_ROOT}/{archive}"
    index = zip_index(s, archive, root / "_zip_index")
    wanted_rows = index[index.filename.isin([uid.removeprefix("fungitastic:") for uid in wanted])]
    if wanted_rows.filename.duplicated().any() or len(wanted_rows) != len(wanted):
        raise ValueError("missing or ambiguous review images in ZIP")
    size = RangeFile(s, url).size
    records = []
    for start, end, members in plan_chunks(wanted_rows):
        chunk = get_range(s, url, start, min(end, size) - 1)
        for _, member in members.iterrows():
            data = extract_member(chunk, start, member)
            if zlib.crc32(data) & 0xffffffff != int(member.crc32):
                raise ValueError("CRC mismatch")
            path = root / member.filename
            path.write_bytes(data)
            uid = "fungitastic:" + member.filename
            digest = sha256_file(path)
            if digest != integrity.loc[uid, "sha256"]:
                raise ValueError("frozen image SHA-256 mismatch")
            with Image.open(path) as im:
                im.verify()
            records.append({"image_uid": uid, "filename": member.filename, "sha256": digest,
                            "bytes": len(data), "crc_verified": True, "split": "val", "archive": archive})
            print(f"Verified {member.filename}", flush=True)
    pd.DataFrame(records).to_csv(output / "download_manifest.csv", index=False, lineterminator="\n")
    for row in selected.itertuples(index=False):
        uids = row.image_uids.split("|")
        width, height = 320, 350
        columns = min(4, len(uids))
        sheet = Image.new("RGB", (width * columns, 65 + height * ((len(uids) + columns - 1) // columns)), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 8), f"{row.observation_uid}  label={row.true_genus}  pred={row.pred_genus}", fill="black")
        draw.text((10, 30), f"confidence={row.confidence:.3f}  views={len(uids)}  JS={row.view_js_divergence:.3f}", fill="black")
        for i, uid in enumerate(uids):
            with Image.open(root / uid.removeprefix("fungitastic:")) as im:
                tile = ImageOps.contain(im.convert("RGB"), (310, 310))
            x, y = (i % columns) * width, 65 + (i // columns) * height
            sheet.paste(tile, (x + (width - tile.width) // 2, y))
            draw.text((x + 5, y + 315), uid.removeprefix("fungitastic:"), fill="black")
        sheet.save(local / f"{row.observation_uid.replace(':', '_')}.png")
    (output / "provenance.json").write_text(json.dumps({"source_url": url,
        "selection": "2 most confident errors + 2 largest JS disagreements + 1 Hericium error + 1 Morchella case; deduplicated",
        "selection_predictions_sha256": sha256_file(predictions), "test_images_downloaded": 0,
        "images": len(records), "observations": len(selected)}, indent=2) + "\n", encoding="utf-8")
    print(selected[["observation_uid", "true_genus", "pred_genus", "n_images"]].to_string(index=False))


if __name__ == "__main__":
    main()
