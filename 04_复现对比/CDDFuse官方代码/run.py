#!/usr/bin/env python3
"""
One-click Evaluation & Inference for CDDFuse (IVF)
====================================================
Runs IVF fusion on all discovered datasets, computes
12 quality metrics, measures inference speed, and saves results.

Usage:
    python run.py                          # run on all datasets
    python run.py --dataset TNO            # TNO dataset only
    python run.py --dataset RoadScene      # RoadScene dataset only
"""

import os
import sys
import time
import argparse
import warnings
import logging
import numpy as np
import torch
import torch.nn as nn

# ————————————————————————————————————————————————
# Local imports
# ————————————————————————————————————————————————
from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from utils.Evaluator import Evaluator
from utils.img_read_save import img_save, image_read_cv2

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

# ================================================================
# CONFIG — edit these if your paths differ
# ================================================================
TEST_IMG_DIR   = "/home/hyhz/HT/test_img"          # must contain subdirs: <dataset>/ir/*.png  and  <dataset>/vi/*.png
TEST_OUT_DIR   = "experiments/MMIF-CDDFuse-main/test_result"       # fused images written here
RESULTS_FILE   = "results.txt"       # appended with each run's metrics

MODELS = {
    "IVF": "models/CDDFuse_IVF.pth",
}

# All 12 metrics available in utils/Evaluator.py
ALL_METRICS = ["EN", "SD", "SF", "AG", "MI", "SCD", "VIFF", "Qabf", "SSIM", "MSE", "PSNR", "CC"]

# ================================================================
# Helpers
# ================================================================

def discover_datasets(base_dir):
    """Return sorted dataset names found under base_dir.
    Each dataset must contain `ir/` and `vi/` subdirectories."""
    if not os.path.isdir(base_dir):
        return []
    datasets = []
    for name in sorted(os.listdir(base_dir)):
        sub = os.path.join(base_dir, name)
        if os.path.isdir(sub) and os.path.isdir(os.path.join(sub, "ir")) and os.path.isdir(os.path.join(sub, "vi")):
            datasets.append(name)
    return datasets


def load_models(ckpt_path, device):
    """Instantiate and load all four CDDFuse sub-networks."""
    Encoder        = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder        = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer  = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    Encoder.load_state_dict(ckpt["DIDF_Encoder"])
    Decoder.load_state_dict(ckpt["DIDF_Decoder"])
    BaseFuseLayer.load_state_dict(ckpt["BaseFuseLayer"])
    DetailFuseLayer.load_state_dict(ckpt["DetailFuseLayer"])

    for m in [Encoder, Decoder, BaseFuseLayer, DetailFuseLayer]:
        m.eval()
    return Encoder, Decoder, BaseFuseLayer, DetailFuseLayer


def fuse_one(ir_tensor, vis_tensor, Encoder, Decoder, BaseFuseLayer, DetailFuseLayer):
    """Forward pass: IR + VIS → fused image tensor (1,1,H,W)."""
    feature_V_B, feature_V_D, _ = Encoder(vis_tensor)
    feature_I_B, feature_I_D, _ = Encoder(ir_tensor)
    feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
    feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
    fused, _ = Decoder(vis_tensor, feature_F_B, feature_F_D)
    return fused


def compute_all_metrics(fi, ir, vi):
    """Return dict of 12 metrics for one fused image.
    fi, ir, vi are float 2-D numpy arrays (uint8 0-255)."""
    return {
        "EN":   Evaluator.EN(fi),
        "SD":   Evaluator.SD(fi),
        "SF":   Evaluator.SF(fi),
        "AG":   Evaluator.AG(fi),
        "MI":   Evaluator.MI(fi, ir, vi),
        "SCD":  Evaluator.SCD(fi, ir, vi),
        "VIFF": Evaluator.VIFF(fi, ir, vi),
        "Qabf": Evaluator.Qabf(fi, ir, vi),
        "SSIM": Evaluator.SSIM(fi, ir, vi),
        "MSE":  Evaluator.MSE(fi, ir, vi),
        "PSNR": Evaluator.PSNR(fi, ir, vi),
        "CC":   Evaluator.CC(fi, ir, vi),
    }


def fmt_val(v):
    """Format a float for table display."""
    if abs(v) < 1e-6:
        return "0.00"
    return f"{v:.2f}"


# ================================================================
# Core evaluation loop
# ================================================================

def evaluate_one_model(ckpt_path, model_name, dataset_name, device):
    """Run fusion + metrics on one (model, dataset) pair.
    Returns (metrics_means, speed_stats) or (None, None) on failure."""

    test_folder    = os.path.join(TEST_IMG_DIR, dataset_name)
    test_out_folder = os.path.join(TEST_OUT_DIR, dataset_name)

    # Verify inputs exist
    ir_dir  = os.path.join(test_folder, "ir")
    vi_dir  = os.path.join(test_folder, "vi")
    if not os.path.isdir(ir_dir) or not os.path.isdir(vi_dir):
        print(f"  [SKIP] Missing ir/ or vi/ in {test_folder}")
        return None, None

    image_names = sorted(os.listdir(ir_dir))
    if not image_names:
        print(f"  [SKIP] No images in {ir_dir}")
        return None, None

    print(f"\n{'='*90}")
    print(f"  Model: {model_name}  |  Dataset: {dataset_name}  |  Images: {len(image_names)}")
    print(f"{'='*90}")

    # — Load models —
    try:
        Encoder, Decoder, BaseFuseLayer, DetailFuseLayer = load_models(ckpt_path, device)
    except FileNotFoundError:
        print(f"  [ERROR] Checkpoint not found: {ckpt_path}")
        return None, None
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        return None, None

    # — Warmup (one image) —
    ir_path  = os.path.join(ir_dir, image_names[0])
    vi_path  = os.path.join(vi_dir, image_names[0])
    data_IR  = image_read_cv2(ir_path, mode="GRAY")[np.newaxis, np.newaxis, ...] / 255.0
    data_VIS = image_read_cv2(vi_path, mode="GRAY")[np.newaxis, np.newaxis, ...] / 255.0
    with torch.no_grad():
        _ = fuse_one(
            torch.FloatTensor(data_IR).to(device),
            torch.FloatTensor(data_VIS).to(device),
            Encoder, Decoder, BaseFuseLayer, DetailFuseLayer,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    # — Inference loop —
    accum_metrics = {k: 0.0 for k in ALL_METRICS}
    total_time_s   = 0.0
    per_img_times  = []

    with torch.no_grad():
        for img_name in image_names:
            # Load
            ir  = image_read_cv2(os.path.join(ir_dir, img_name), mode="GRAY")
            vi  = image_read_cv2(os.path.join(vi_dir, img_name), mode="GRAY")
            data_IR  = ir[np.newaxis, np.newaxis, ...] / 255.0
            data_VIS = vi[np.newaxis, np.newaxis, ...] / 255.0
            data_IR, data_VIS = torch.FloatTensor(data_IR).to(device), torch.FloatTensor(data_VIS).to(device)

            # Time the forward pass
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            data_Fuse = fuse_one(data_IR, data_VIS, Encoder, Decoder, BaseFuseLayer, DetailFuseLayer)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            elapsed = t1 - t0
            per_img_times.append(elapsed)
            total_time_s += elapsed

            # Normalize & save
            data_min, data_max = torch.min(data_Fuse), torch.max(data_Fuse)
            data_Fuse = (data_Fuse - data_min) / (data_max - data_min).clamp_min(1e-8)
            fi = np.squeeze((data_Fuse * 255).cpu().numpy()).astype(np.uint8)
            img_save(fi, img_name.split(".")[0], test_out_folder)

            # Compute metrics on uint8 arrays
            fi_eval = fi.astype(np.float32)
            ir_eval = ir.astype(np.float32)
            vi_eval = vi.astype(np.float32)
            m = compute_all_metrics(fi_eval, ir_eval, vi_eval)
            for k in ALL_METRICS:
                accum_metrics[k] += m[k]

    n = len(image_names)

    # — Averages —
    avg_metrics = {k: v / n for k, v in accum_metrics.items()}

    # — Speed stats —
    speed = {
        "n_images":    n,
        "total_s":     total_time_s,
        "avg_ms":      (total_time_s / n) * 1000,
        "fps":         n / total_time_s if total_time_s > 0 else 0.0,
        "first_ms":    per_img_times[0] * 1000,
        "best_ms":     min(per_img_times) * 1000,
        "worst_ms":    max(per_img_times) * 1000,
    }

    return avg_metrics, speed


# ================================================================
# Pretty-print & save
# ================================================================

def print_results(model_name, dataset_name, avg_metrics, speed):
    """Print a formatted table of metrics + speed."""
    header = (
        f"\n  {'Metric':>6}  " +
        " ".join(f"{m:>6}" for m in ALL_METRICS)
    )
    sep = "  " + "-" * (len(ALL_METRICS) * 7 + 5)
    values = (
        f"  {'Value':>6}  " +
        " ".join(f"{avg_metrics[m]:6.2f}" for m in ALL_METRICS)
    )
    print(f"\n  ── Quality Metrics ───────────────────────────────────────────────────────────────")
    print(sep)
    print(header)
    print(sep)
    print(values)
    print(sep)

    print(f"\n  ── Inference Speed ─────────────────────────────────────────────────────────────────")
    print(f"  Images: {speed['n_images']:4d}  |  Total: {speed['total_s']:6.2f} s  |  "
          f"Avg: {speed['avg_ms']:6.1f} ms  |  FPS: {speed['fps']:6.1f}  |  "
          f"First: {speed['first_ms']:5.0f} ms  |  Best: {speed['best_ms']:5.0f} ms  |  Worst: {speed['worst_ms']:5.0f} ms")
    print()


def append_to_file(model_name, dataset_name, avg_metrics, speed, filepath):
    """Append results to a text file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "a") as f:
        f.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}  Model={model_name}  Dataset={dataset_name}\n")
        f.write("Metrics:\n")
        for m in ALL_METRICS:
            f.write(f"  {m:>6}: {avg_metrics[m]:.4f}\n")
        f.write("Speed:\n")
        f.write(f"  images: {speed['n_images']}  total_s: {speed['total_s']:.3f}  "
                f"avg_ms: {speed['avg_ms']:.3f}  fps: {speed['fps']:.3f}  "
                f"first_ms: {speed['first_ms']:.3f}  best_ms: {speed['best_ms']:.3f}  worst_ms: {speed['worst_ms']:.3f}\n")
        f.write("-" * 80 + "\n")


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="CDDFuse One-click Evaluation & Inference")
    parser.add_argument("--model",   type=str, default=None, help="Model name (default: IVF)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (default: all discovered)")
    parser.add_argument("--gpu",     type=str, default="0",  help="CUDA_VISIBLE_DEVICES (default 0)")
    parser.add_argument("--no-save", action="store_true",    help="Don't write results.txt")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve models
    if args.model and args.model.upper() in MODELS:
        selected_models = {args.model.upper(): MODELS[args.model.upper()]}
    elif args.model and args.model.lower() == "all":
        selected_models = MODELS
    elif args.model is None:
        selected_models = MODELS  # default: all
    else:
        print(f"Unknown model '{args.model}'. Available: {list(MODELS.keys())}")
        sys.exit(1)

    # Resolve datasets
    all_datasets = discover_datasets(TEST_IMG_DIR)
    if not all_datasets:
        print(f"[ERROR] No datasets found under '{TEST_IMG_DIR}/'. "
              f"Expected structure: test_img/<dataset>/ir/*.png  +  test_img/<dataset>/vi/*.png")
        sys.exit(1)
    print(f"Discovered datasets: {all_datasets}")

    if args.dataset and args.dataset != "all":
        if args.dataset in all_datasets:
            selected_datasets = [args.dataset]
        else:
            print(f"Unknown dataset '{args.dataset}'. Available: {all_datasets}")
            sys.exit(1)
    else:
        selected_datasets = all_datasets

    print(f"Models:   {list(selected_models.keys())}")
    print(f"Datasets: {selected_datasets}")

    all_results = {}

    for model_name, ckpt_path in sorted(selected_models.items()):
        for ds in selected_datasets:
            try:
                avg_metrics, speed = evaluate_one_model(ckpt_path, model_name, ds, device)
                if avg_metrics is None:
                    continue
                print_results(model_name, ds, avg_metrics, speed)
                if not args.no_save:
                    append_to_file(model_name, ds, avg_metrics, speed, RESULTS_FILE)
                all_results[(model_name, ds)] = {"metrics": avg_metrics, "speed": speed}
            except Exception as e:
                print(f"  [ERROR] {model_name}/{ds}: {e}")
                import traceback
                traceback.print_exc()

    # — Summary table across models/datasets —
    if len(all_results) > 1:
        print(f"\n{'='*90}")
        print("  CROSS-MODEL SUMMARY")
        print(f"{'='*90}")
        # Header
        max_name = max(len(f"{m}/{d}") for (m, d) in all_results)
        hdr = f"  {'Model/Dataset':{max_name}s}  " + " ".join(f"{m:>6}" for m in ALL_METRICS) + "   FPS   Avg_ms"
        print(hdr)
        print("  " + "-" * len(hdr))
        for (mn, ds), data in sorted(all_results.items()):
            label = f"{mn}/{ds}"
            met  = data["metrics"]
            spd  = data["speed"]
            print(f"  {label:{max_name}s}  " +
                  " ".join(f"{met[m]:6.2f}" for m in ALL_METRICS) +
                  f"   {spd['fps']:5.1f}  {spd['avg_ms']:5.0f} ms")
        print("  " + "-" * len(hdr))

    if not args.no_save:
        print(f"\nResults appended to: {os.path.abspath(RESULTS_FILE)}")
    print("Done.\n")


if __name__ == "__main__":
    main()
