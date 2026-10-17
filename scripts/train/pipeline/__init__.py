"""Modular training/evaluation pipeline (S6).

One module per responsibility; ``scripts/train/train_ma.py`` is only the command-line orchestrator.

    data.py (sibling)        dataset version loading + SHA-256 verification against the published manifest
    pipeline/datasets.py     PyTorch Dataset over the per-split image tables
    pipeline/transforms.py   train / evaluation transforms from the ``input`` config section
    pipeline/models.py       architecture factory (pretrained weights + classification head)
    pipeline/training.py     seeds, class weighting, optimizer/scheduler, one training epoch, checkpoints
    pipeline/evaluation.py   inference, image/observation metrics, predictions export
    metrics.py (sibling)     metric definitions (macro-F1, balanced accuracy, top-k, per class, confusion)
    pipeline/taxonomy.py     metrics at higher taxonomic levels from saved predictions (no PyTorch)
    pipeline/artifacts.py    run provenance (config/code hashes, versions), progress.json, RESULTS.md

Importing this package does not import PyTorch; each module imports only what it needs, so the taxonomy
and metric layers run without a training environment.
"""
