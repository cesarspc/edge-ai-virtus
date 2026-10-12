"""Measure model size and synthetic FP32 CPU latency; never read dataset photographs."""
import argparse
import io
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from train_ma import build_model
from data import REPO_ROOT, resolve, sha256_file


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations",type=int,default=30)
    ap.add_argument("--threads",type=int,default=4)
    args=ap.parse_args()
    if args.iterations < 10 or args.threads < 1:
        ap.error("at least 10 iterations and one thread are required")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    results=[]
    for filename in ("ma_resnet18_v002.yaml","s5_mobilenet_v3_large_v002_a.yaml"):
        path=REPO_ROOT/"configs/train"/filename
        cfg=yaml.safe_load(path.read_text(encoding="utf-8"))
        model=build_model(cfg,pretrained=False).eval().cpu()
        x=torch.randn(1,3,cfg["input"]["size"],cfg["input"]["size"])
        with torch.inference_mode():
            for _ in range(5):
                model(x)
            elapsed=[]
            for _ in range(args.iterations):
                start=time.perf_counter()
                model(x)
                elapsed.append((time.perf_counter()-start)*1000)
        state=io.BytesIO()
        torch.save(model.state_dict(),state)
        results.append({"architecture":cfg["model"]["architecture"],"classes":cfg["model"]["num_classes"],
                        "parameters":sum(p.numel() for p in model.parameters()),
                        "serialized_state_mb":len(state.getbuffer())/1e6,
                        "latency_median_ms":float(np.median(elapsed)),"latency_p90_ms":float(np.percentile(elapsed,90)),
                        "latencies_ms":elapsed,"weights":"random; architecture characterization only",
                        "config":str(path.relative_to(REPO_ROOT)),"config_sha256":sha256_file(path)})
        print(json.dumps({k:v for k,v in results[-1].items() if k!="latencies_ms"},indent=2),flush=True)
    payload={"device":"PC CPU; not a mobile measurement", "precision":"FP32", "batch_size":1,
             "input":"synthetic tensor 1x3x224x224", "preprocessing_included":False,
             "mobile_latency_measured":False,"test_images_read":0,"warmup_iterations":5,
             "iterations":args.iterations,"threads":args.threads,"logical_cpus":os.cpu_count(),
             "processor":platform.processor(),"python":platform.python_version(),"torch":str(torch.__version__),
             "models":results}
    target=REPO_ROOT/"reports/s5/architecture_profile.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
