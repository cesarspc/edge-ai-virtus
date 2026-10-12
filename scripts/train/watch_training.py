"""Show a live progress bar for an existing run; Ctrl+C closes only the viewer."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]


def duration(seconds: float) -> str:
    minutes = int(seconds) // 60
    return f"{minutes // 60:02d}:{minutes % 60:02d}:{int(seconds) % 60:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Run directory (default: latest full v002 run)")
    args = parser.parse_args()
    if args.run:
        run = Path(args.run)
        if not run.is_absolute():
            run = REPO / run
    else:
        candidates = []
        for info in (REPO / "runs").glob("*/run.json"):
            record = json.loads(info.read_text(encoding="utf-8"))
            if record.get("mode") == "train" and record.get("dataset", {}).get("version") == "v002":
                candidates.append(info.parent)
        if not candidates:
            parser.error("No full v002 run found; use --run to select another run.")
        run = max(candidates, key=lambda p: p.name)
    if not (run / "run.json").is_file():
        parser.error(f"Invalid run: {run}")
    print(f"Corrida: {run}\nCtrl+C cierra el visor; el entrenamiento continua.")
    print("El estado se actualiza cada 10 lotes; la ETA corresponde a train de la epoca actual.")
    bar, previous = None, None
    try:
        while True:
            try:
                state = json.loads((run / "progress.json").read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, PermissionError):
                time.sleep(1)
                continue
            if state != previous:
                stage = state["stage"]
                if stage == "training":
                    epoch, total = state["epoch"], state["batches_total"]
                    if bar is None or epoch != previous.get("epoch"):
                        if bar is not None:
                            bar.close()
                        bar = tqdm(total=total, desc=f"Epoca {epoch} - train", unit="lote",
                                   dynamic_ncols=True,
                                   bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}")
                    count = state["batches_completed"]
                    bar.n = count
                    remaining = state["seconds"] / max(1, count) * (total - count)
                    bar.set_postfix_str(f"loss={state['train_loss']:.4f}, ETA={duration(remaining)}")
                    bar.refresh()
                else:
                    if bar is not None:
                        bar.close()
                        bar = None
                    tqdm.write(f"Estado: {stage} | actualizado: {state['updated_utc']}")
                previous = state
            if state["stage"] in ("completed", "paused"):
                print(f"Resultados: {run / 'RESULTS.md'}" if state["stage"] == "completed"
                      else f"Checkpoint para reanudar: {run / 'last.pt'}")
                return 0
            time.sleep(1)
    except KeyboardInterrupt:
        return 0
    finally:
        if bar is not None:
            bar.close()


if __name__ == "__main__":
    raise SystemExit(main())
