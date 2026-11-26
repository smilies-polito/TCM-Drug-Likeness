#!/usr/bin/env python3
"""
Benchmark script for timing the training pipeline.

Run from the src/ directory:

    # Run both modes and compare (default)
    python benchmark_training.py

    # Run only the optimised version
    python benchmark_training.py --mode optimized

    # Run only the baseline (pre-Phase-2) version
    python benchmark_training.py --mode baseline

Modes
-----
optimized  : Phase-2 fixes applied
               - torch.no_grad() during test
               - optimizer.zero_grad(set_to_none=True)
               (DataLoader uses num_workers=0 in both modes to avoid
                mmap failures inside Singularity/Apptainer containers)
baseline   : original behaviour (no fixes)
compare    : runs BOTH modes sequentially and prints a diff table (default)
"""

import argparse
import os
import sys
import time
import torch
import warnings
import logging
import copy

# Suppress noisy loggers
for name in ("tensorflow", "jax", "lightning", "transformers", "deepchem",
             "deep-chem", "deep_chem"):
    logging.getLogger(name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from dataset_featurizer import MoleculeDataset
from torch_geometric.data import DataLoader
from model_architecture.tree_path import myModel
import numpy as np

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------
PARAMS = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 0.0005,
    'sgd_momentum': 0.9,
    'scheduler_gamma': 0.9,
    'pos_weight': 0.5,
    'model_embedding_size': 64,
    'model_attention_heads': 3,
    'model_layers': 2,
    'model_dropout_rate': 0.3,
    'model_top_k_ratio': 0.5,
    'model_top_k_every_n': 2,
    'model_dense_neurons': 128,
    'train_file': "data/raw/new2_ZINC_final_balanced_train_set.csv",
    'test_file':  "data/raw/new2_ZINC_final_balanced_validation_set.csv",
}
N_EPOCHS = 5

# ---------------------------------------------------------------------------
# Per-epoch profiling helpers
# ---------------------------------------------------------------------------

def _make_data_dict(batch, tree_batch):
    return {
        "mol_raws": {
            "x": batch.x.float(),
            "edge_attr": batch.edge_attr.float(),
            "edge_index": batch.edge_index,
            "batch": batch.batch,
        },
        "mol_trees": {
            "tree_node_features": tree_batch.x.float(),
            "tree_edge_attr": tree_batch.edge_attr.float(),
            "tree_edge_index": tree_batch.edge_index,
            "tree_batch_index": tree_batch.batch,
        }
    }


def run_train_epoch(model, train_loader, optimizer, loss_fn, device, set_to_none):
    """One training epoch. Returns (train_loss, timings_dict)."""
    model.train()
    timings = {"to_device": 0.0, "forward": 0.0, "backward": 0.0, "optim_step": 0.0}
    running_loss = 0.0
    step = 0
    n_batches = len(train_loader)
    print_every = max(1, n_batches // 10)  # print ~10 progress updates per epoch

    for batch, tree_batch in train_loader:
        t0 = time.perf_counter()
        batch.to(device)
        tree_batch.to(device)
        t1 = time.perf_counter()

        optimizer.zero_grad(set_to_none=set_to_none)
        data = _make_data_dict(batch, tree_batch)
        pred, _, _, _, _ = model(data, device)
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        t2 = time.perf_counter()

        loss.backward()
        t3 = time.perf_counter()

        optimizer.step()
        t4 = time.perf_counter()

        timings["to_device"] += t1 - t0
        timings["forward"]   += t2 - t1
        timings["backward"]  += t3 - t2
        timings["optim_step"]+= t4 - t3
        running_loss += loss.item()
        step += 1

        if step % print_every == 0 or step == n_batches:
            elapsed = sum(timings.values())
            print(f"      batch {step}/{n_batches}  loss={running_loss/step:.4f}  "
                  f"elapsed={elapsed:.1f}s", flush=True)

    return running_loss / max(step, 1), timings


def run_test_epoch(model, test_loader, loss_fn, device, use_no_grad):
    """One test pass. Returns (test_loss, elapsed_seconds)."""
    model.eval()
    running_loss = 0.0
    step = 0
    t_start = time.perf_counter()

    def _inner():
        nonlocal running_loss, step
        for batch, tree_batch in test_loader:
            batch.to(device)
            tree_batch.to(device)
            data = _make_data_dict(batch, tree_batch)
            pred, _, _, _, _ = model(data, device)
            loss = loss_fn(torch.squeeze(pred), batch.y.float())
            running_loss += loss.item()
            step += 1

    if use_no_grad:
        with torch.no_grad():
            _inner()
    else:
        _inner()

    return running_loss / max(step, 1), time.perf_counter() - t_start


# ---------------------------------------------------------------------------
# Full benchmark run for one mode
# ---------------------------------------------------------------------------

def run_benchmark(mode, train_dataset, test_dataset, device):
    """
    Run N_EPOCHS of train + test under the given mode.
    Returns a dict: {stage_label: elapsed_seconds}
    """
    optimized = (mode == "optimized")
    print(f"\n{'='*65}", flush=True)
    print(f"  MODE: {mode.upper()}", flush=True)
    print(f"  no_grad={optimized}  set_to_none={optimized}  num_workers=0 (both modes)", flush=True)
    print(f"{'='*65}", flush=True)

    results = {}

    # DataLoader
    # num_workers=0: single-process loading avoids multiprocessing shared-memory
    # (mmap) failures inside Singularity/Apptainer containers where /dev/shm is limited.
    # pin_memory=False: pinned-memory also uses IPC/shared-mem on some PyTorch builds.
    # num_workers=0, pin_memory=False: avoids mmap/shm failures in Singularity containers
    dl_kwargs = dict(batch_size=PARAMS["batch_size"], num_workers=4, pin_memory=True)

    t0 = time.perf_counter()
    train_loader = DataLoader(train_dataset, shuffle=True,  **dl_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **dl_kwargs)
    results["dataloader_creation"] = time.perf_counter() - t0
    print(f"  DataLoader created in {results['dataloader_creation']:.3f}s")

    # Fresh model (same init for both modes via fixed seed)
    torch.manual_seed(42)
    model_params = {k: v for k, v in PARAMS.items() if k.startswith("model_")}
    model_params["model_edge_dim"] = train_dataset[0][0].edge_attr.shape[1]
    model = myModel(feature_size=train_dataset[0][0].x.shape[1],
                    model_params=model_params).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    weight = torch.tensor([PARAMS["pos_weight"]], dtype=torch.float32).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=PARAMS["learning_rate"],
                                momentum=PARAMS["sgd_momentum"],
                                weight_decay=PARAMS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=PARAMS["scheduler_gamma"])

    train_times = []
    test_times  = []

    for epoch in range(N_EPOCHS):
        print(f"\n  --- Epoch {epoch}/{N_EPOCHS-1} ---", flush=True)

        t_train = time.perf_counter()
        train_loss, train_timings = run_train_epoch(
            model, train_loader, optimizer, loss_fn, device,
            set_to_none=optimized
        )
        epoch_train = time.perf_counter() - t_train
        train_times.append(epoch_train)

        total_t = sum(train_timings.values())
        print(f"    train: {epoch_train:.2f}s  loss={train_loss:.4f}", flush=True)
        if total_t > 0:
            parts = "  ".join(f"{k}={v:.2f}s({100*v/total_t:.0f}%)"
                              for k, v in train_timings.items())
            print(f"    breakdown: {parts}")

        print(f"    running test pass...", flush=True)
        test_loss, epoch_test = run_test_epoch(
            model, test_loader, loss_fn, device, use_no_grad=optimized
        )
        test_times.append(epoch_test)
        print(f"    test:  {epoch_test:.2f}s  loss={test_loss:.4f}"
              + ("  [no_grad=ON]" if optimized else "  [no_grad=OFF]"), flush=True)

        scheduler.step()

    results["avg_train_epoch"] = sum(train_times) / len(train_times)
    results["avg_test_epoch"]  = sum(test_times)  / len(test_times)
    results["total_train"]     = sum(train_times)
    results["total_test"]      = sum(test_times)
    results["total_training"]  = results["total_train"] + results["total_test"]

    print(f"\n  Summary ({N_EPOCHS} epochs):")
    print(f"    avg train epoch : {results['avg_train_epoch']:.2f}s")
    print(f"    avg test epoch  : {results['avg_test_epoch']:.2f}s")
    print(f"    total           : {results['total_training']:.2f}s")

    return results


# ---------------------------------------------------------------------------
# Comparison printer
# ---------------------------------------------------------------------------

def print_comparison(baseline, optimized):
    print("\n" + "=" * 72)
    print("  BEFORE vs. AFTER PHASE-2 COMPARISON")
    print("=" * 72)
    metrics = [
        ("dataloader_creation", "DataLoader creation"),
        ("avg_train_epoch",     "Avg train epoch"),
        ("avg_test_epoch",      "Avg test epoch  ← no_grad fix"),
        ("total_train",         f"Total train ({N_EPOCHS} epochs)"),
        ("total_test",          f"Total test  ({N_EPOCHS} epochs)"),
        ("total_training",      f"TOTAL ({N_EPOCHS} epochs)"),
    ]
    print(f"  {'Metric':<38s} {'Baseline':>10s}  {'Optimized':>10s}  {'Speedup':>8s}")
    print(f"  {'-'*38}  {'-'*10}  {'-'*10}  {'-'*8}")
    for key, label in metrics:
        b = baseline.get(key, 0)
        o = optimized.get(key, 0)
        if o > 0:
            speedup = f"{b/o:.2f}x"
            delta_pct = f"({100*(b-o)/b:+.0f}%)" if b > 0 else ""
        else:
            speedup = "N/A"
            delta_pct = ""
        print(f"  {label:<38s} {b:>9.2f}s  {o:>9.2f}s  {speedup:>6s} {delta_pct}")
    print("=" * 72)
    total_b = baseline.get("total_training", 0)
    total_o = optimized.get("total_training", 0)
    saved = total_b - total_o
    print(f"  Time saved over {N_EPOCHS} epochs: {saved:.1f}s"
          f"  →  projected over 700 epochs: ~{saved * 700/N_EPOCHS/60:.0f} min")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Training pipeline benchmark")
    parser.add_argument("--mode", choices=["optimized", "baseline", "compare"],
                        default="compare",
                        help="Which variant to run (default: compare)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load datasets once (shared between both modes)
    print(f"\nLoading datasets (shared)...")
    t0 = time.perf_counter()
    print("  Loading train dataset...", flush=True)
    train_dataset = MoleculeDataset(filename=PARAMS["train_file"])
    print(f"  Train loaded ({len(train_dataset)} molecules)  {time.perf_counter()-t0:.1f}s", flush=True)
    test_dataset  = MoleculeDataset(filename=PARAMS["test_file"], test=True)
    print(f"  Datasets loaded in {time.perf_counter()-t0:.1f}s  "
          f"(train={len(train_dataset)}, test={len(test_dataset)})", flush=True)

    print("  Warming up cache (pre-loading all .pt files)...", flush=True)
    train_dataset.warmup_cache()
    test_dataset.warmup_cache()
    print(f"  Cache warm in {time.perf_counter()-t0:.1f}s total", flush=True)

    if args.mode == "compare":
        print(f"\nRunning BOTH modes for {N_EPOCHS} epochs each...")
        res_optimized = run_benchmark("optimized", train_dataset, test_dataset, device)
        res_baseline  = run_benchmark("baseline",  train_dataset, test_dataset, device)
        print_comparison(res_baseline, res_optimized)
    elif args.mode == "baseline":
        run_benchmark("baseline", train_dataset, test_dataset, device)
    else:
        run_benchmark("optimized", train_dataset, test_dataset, device)


if __name__ == "__main__":
    main()

