"""CPU reference and per-tensor/group drift for paired fresh fine-tuning runs."""
from __future__ import annotations

import math

import torch

from pipeline.crop_state import state_digest


def initialize_drift(model, provenance):
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return {"initial_state": state, "initial_state_sha256": state_digest(state),
            "parameter_names": [name for name, _ in model.named_parameters()],
            "provenance": provenance, "rows": [], "summaries": []}


def measure_drift(model, record, epoch, batches_seen, optimizer_steps):
    """No second GPU model; float64 norms on CPU, buffers distinguished from parameters."""
    initial = record["initial_state"]
    parameters = set(record["parameter_names"])
    groups = {}
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().double()
        before = initial[name].double()
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite model state: {name}")
        group = ("classifier" if name.startswith(("classifier.", "fc.")) else "backbone") if name in parameters else "buffers"
        # BN running moments and integer counters are reported separately from learned weights.
        if group == "buffers":
            group = "bn_counters" if name.endswith("num_batches_tracked") else "bn_running_statistics"
        n0 = float(torch.sum(before*before))
        n1 = float(torch.sum(value*value))
        diff = value-before
        nd = float(torch.sum(diff*diff))
        row = {"epoch": epoch, "batches_seen": batches_seen, "optimizer_steps": optimizer_steps,
               "tensor": name, "group": group, "kind": "parameter" if name in parameters else "buffer",
               "numel": tensor.numel(), "initial_norm_l2": math.sqrt(n0), "current_norm_l2": math.sqrt(n1),
               "absolute_delta_l2": math.sqrt(nd), "relative_delta_l2": math.sqrt(nd)/(math.sqrt(n0)+1e-12)}
        record["rows"].append(row)
        totals = groups.setdefault(group, [0., 0., 0., 0])
        totals[0] += n0; totals[1] += n1; totals[2] += nd; totals[3] += tensor.numel()
    for group, (n0, n1, nd, numel) in groups.items():
        record["summaries"].append({"epoch": epoch, "batches_seen": batches_seen, "optimizer_steps": optimizer_steps,
                                   "group": group, "numel": numel, "initial_norm_l2": math.sqrt(n0),
                                   "current_norm_l2": math.sqrt(n1), "absolute_delta_l2": math.sqrt(nd),
                                   "relative_delta_l2": math.sqrt(nd)/(math.sqrt(n0)+1e-12)})


def optimizer_step_count(optimizer):
    """Actual Adam updates (GradScaler may skip a non-finite batch)."""
    return max((int(state.get("step", 0)) for state in optimizer.state.values()), default=0)
