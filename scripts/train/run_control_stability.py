"""Sequential CUDA runs for seeds 43/44; repeat this command to resume interrupted work.

python -u scripts/train/run_control_stability.py
Tracks each run, reuses completed jobs, resumes last.pt, preserves the selected seed 42.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from progress_ui import bar, stage

with stage("Preparando campana de estabilidad", active=__name__ == "__main__"):
    import yaml
    from data import REPO_ROOT, resolve, sha256_file
    from pipeline.artifacts import ProgressFile, code_hashes
    from pipeline.crop_state import atomic_json, signature


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def recipe_without_seed(cfg):
    value = copy.deepcopy(cfg)
    value.pop("run_name", None)
    value["training"].pop("seed")
    value["experiment"].pop("seed_pair", None)
    return value


def check_configs(base, configs):
    if base["input"].get("extra_crop") or base["evaluation"]["evaluate_test_at_end"]:
        raise ValueError("control stability requires original framing and val-only training")
    seeds = [base["training"]["seed"], *[c["training"]["seed"] for c in configs]]
    if seeds != [42, 43, 44]:
        raise ValueError("this prespecified campaign uses seeds 42,43,44 in that order")
    for cfg in configs:
        if recipe_without_seed(cfg) != recipe_without_seed(base):
            raise ValueError("only run name and seed metadata may differ from the reference recipe")


@contextmanager
def campaign_lock(path):
    """OS-backed lock releases on process exit; no PID-reuse or stale-lock deletion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    if path.stat().st_size == 0:
        stream.write(b"0"); stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise SystemExit("This stability campaign is already active.")
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == "nt":
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


def matching_runs(cfg):
    matches = []
    for path in sorted(resolve(cfg["output"]["runs_dir"]).glob("*/run.json")):
        info = read_json(path)
        if info.get("mode") == "train" and info["config"] == cfg:
            matches.append(path.parent)
    return matches


def job_to_run(cfg):
    matches = matching_runs(cfg)
    completed = [p for p in matches if (p / "completed.json").exists()]
    if completed:
        # Earliest completed run is fixed; never choose by its validation score.
        return "completed", completed[0]
    resumable = [p for p in matches if (p / "last.pt").exists()]
    return ("resume", resumable[-1]) if resumable else ("new", None)


def validate_completed(path, cfg, fingerprint, initial):
    info, done = read_json(path / "run.json"), read_json(path / "completed.json")
    if info["config"] != cfg or info["dataset"]["content_fingerprint"] != fingerprint:
        raise ValueError("completed job recipe or dataset differs from the campaign")
    if info["device"] != "cuda" or done["evaluated_splits"] != ["val"] or done["test_evaluations"] != 0:
        raise ValueError("campaign jobs must finish on CUDA using only validation")
    origin = read_json(path / "initialization.json")
    if origin["sha256"] != initial["sha256"] or origin["state_sha256"] != initial["state_sha256"]:
        raise ValueError("job did not start from the common selected C weights")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/control_stability_v002.yaml")
    args = ap.parse_args()
    protocol_path = resolve(args.config)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    out = resolve(protocol["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(resolve(protocol["reference_config"]).read_text(encoding="utf-8"))
    paths = [resolve(p) for p in protocol["additional_configs"]]
    configs = [yaml.safe_load(p.read_text(encoding="utf-8")) for p in paths]
    check_configs(base, configs)
    reference = resolve(protocol["reference_run"])
    reference_info = read_json(reference / "run.json")
    if reference_info["config"] != base or not (reference / "completed.json").exists():
        raise ValueError("seed 42 reference must be the prespecified completed control run")
    initial = read_json(reference / "initialization.json")
    if sha256_file(resolve(base["initialization"]["checkpoint"])) != protocol["initial_checkpoint_sha256"] or initial["sha256"] != protocol["initial_checkpoint_sha256"]:
        raise ValueError("initial C checkpoint changed")
    fingerprint = reference_info["dataset"]["content_fingerprint"]
    validate_completed(reference, base, fingerprint, initial)
    code = code_hashes()
    if reference_info["code_sha256"] != code:
        raise ValueError("training pipeline changed since seed 42; a new comparison protocol is needed")
    identity = {"protocol": protocol, "configs": [base, *configs], "code_sha256": code,
                "runner_sha256": sha256_file(Path(__file__)), "dataset_fingerprint": fingerprint,
                "reference_best_sha256": sha256_file(reference / "best.pt")}
    sig = signature(identity)
    with campaign_lock(out / "campaign.lock"):
        state_path = out / "campaign.json"
        state = read_json(state_path) if state_path.exists() else {"signature": sig, "jobs": {}}
        if state["signature"] != sig:
            raise ValueError("campaign inputs changed; restore the original configuration to resume")
        atomic_json(out / "protocol.json", {**identity, "signature": sig, "selection_policy": protocol["selection_policy"]})
        state["jobs"]["42"] = {"status": "completed", "run": str(reference.relative_to(REPO_ROOT))}
        progress = ProgressFile(out, out.name)
        atomic_json(state_path, state)
        print(f"Campana: {out}\nSemilla 42 preservada; GPU secuencial para 43 y 44.", flush=True)
        with bar(total=3, initial=1, desc="Semillas completadas", unit="semilla") as outer:
            for path, cfg in zip(paths, configs):
                seed = cfg["training"]["seed"]
                mode, run = job_to_run(cfg)
                if mode != "completed":
                    before = set(matching_runs(cfg))
                    state["jobs"][str(seed)] = {"status": "running", "mode": mode, "resume_from": str(run.relative_to(REPO_ROOT)) if run else None}
                    atomic_json(state_path, state); progress("training", seed=seed, completed_seeds=int(outer.n))
                    command = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                               str(REPO_ROOT / "scripts/train_visible.ps1"), "-SkipInstall", "-Device", "cuda", "-Config", str(path)]
                    if mode == "resume": command += ["-Resume", str(run / "last.pt")]
                    print(f"[SEMILLA {seed}] {mode}; checkpoints por epoca, test reservado.", flush=True)
                    try:
                        result = subprocess.run(command, cwd=REPO_ROOT)
                    except KeyboardInterrupt:
                        state["jobs"][str(seed)]["status"] = "interrupted"
                        atomic_json(state_path, state); progress("interrupted", seed=seed)
                        raise
                    mode, run = job_to_run(cfg)
                    if result.returncode or mode != "completed":
                        new_paths = sorted(set(matching_runs(cfg))-before)
                        state["jobs"][str(seed)].update(status="attention", run=str(new_paths[-1].relative_to(REPO_ROOT)) if new_paths else None)
                        atomic_json(state_path, state); progress("attention", seed=seed, exit_code=result.returncode)
                        raise SystemExit("Training interrupted/paused. Repeat this command to resume saved last.pt.")
                validate_completed(run, cfg, fingerprint, initial)
                state["jobs"][str(seed)] = {"status": "completed", "run": str(run.relative_to(REPO_ROOT))}
                atomic_json(state_path, state); outer.update(1)
                progress("training_jobs", completed_seeds=int(outer.n), total_seeds=3)
        state["training_complete"] = True
        atomic_json(state_path, state)
        progress("training_completed", completed_seeds=3)
        print("Tres semillas terminadas. Analizar con analyze_control_stability.py; test no fue evaluado.", flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
