"""Plot saved weight drift without inference. Integer epochs, percentages and zero-based axes."""
import argparse
import hashlib
import json
from pathlib import Path

from progress_ui import bar, stage


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", required=True, type=Path)
    args = ap.parse_args()
    with stage("Preparando figura de cambios de pesos"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        frame = pd.read_csv(args.study / "parameter_drift_summary.csv")
    groups = [("backbone", "Extractor"), ("classifier", "Cabeza"), ("bn_running_statistics", "BatchNorm (buffers)")]
    with bar(total=1, desc="Figura de cambios de pesos", unit="figura") as progress:
        if frame.epoch.nunique() == 1:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for offset, branch, title in [(-.18, "control", "Control"), (.18, "cropped", "Recorte central 75 %")]:
                values = [100*frame.loc[frame.branch.eq(branch)&frame.group.eq(group), "relative_delta_l2"].item() for group, _ in groups]
                bars = ax.bar(np.arange(len(groups))+offset, values, .36, label=title)
                ax.bar_label(bars, labels=[f"{value:.2f} %" for value in values], padding=3)
            ax.set_xticks(range(len(groups)), [title for _, title in groups])
            ax.set(ylabel="Distancia L2 relativa respecto a C (%)", title=f"Piloto después de {int(frame.epoch.iloc[0])} época")
            ax.set_ylim(0, max(frame.loc[frame.group.isin([g for g, _ in groups]), "relative_delta_l2"])*125)
            ax.legend(); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, (group, title) in zip(axes, groups):
                for branch, label in [("control", "Control"), ("cropped", "Recorte 75 %")]:
                    f = frame.loc[frame.branch.eq(branch)&frame.group.eq(group)].sort_values("epoch")
                    ax.plot([0, *f.epoch], [0, *(100*f.relative_delta_l2)], marker="o", label=label)
                ax.set_xticks(sorted({0, *frame.epoch.astype(int)}))
                ax.set(title=title, xlabel="Época", ylabel="Cambio L2 relativo (%)", ylim=(0, None)); ax.legend()
        fig.tight_layout()
        target = args.study / "figures/weight_drift.png"
        temporary = target.with_suffix(".png.tmp")
        fig.savefig(temporary, format="png", dpi=160); temporary.replace(target); plt.close(fig)
        progress.update(1)
    record = {"source_sha256": hashlib.sha256((args.study / "parameter_drift_summary.csv").read_bytes()).hexdigest(),
              "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "units": "percent relative L2 distance from shared C initialization; not performance", "zero_based_axes": True}
    (args.study / "weights_plot.json").write_text(json.dumps(record, indent=2))
    print(target.resolve(), flush=True)


if __name__ == "__main__": main()
