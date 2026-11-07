"""
PRADAS — Model Metrics Evaluator & Comparative Analysis.

Extracts training and validation metrics embedded in YOLO checkpoint files (.pt),
generates comparison tables, and plots performance graphs for project reports.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_ckpt_metrics(ckpt_path: str):
    """Load embedded training metrics and history from a YOLO checkpoint."""
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("train_metrics", {})
    results = ckpt.get("train_results", {})
    epochs = ckpt.get("epoch", 0)

    p = metrics.get("metrics/precision(B)", 0.0)
    r = metrics.get("metrics/recall(B)", 0.0)
    map50 = metrics.get("metrics/mAP50(B)", 0.0)
    map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    return {
        "epochs": epochs,
        "precision": p,
        "recall": r,
        "map50": map50,
        "map50_95": map50_95,
        "f1": f1,
        "val_box_loss": metrics.get("val/box_loss", 0.0),
        "val_cls_loss": metrics.get("val/cls_loss", 0.0),
        "history": results,
    }


def main():
    root = Path(__file__).resolve().parent.parent
    v1_path = root / "models" / "best.pt"
    v2_path = root / "models" / "best_v2.pt"
    out_dir = root / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    m1 = load_ckpt_metrics(str(v1_path))
    m2 = load_ckpt_metrics(str(v2_path))

    print("\n" + "=" * 70)
    print("      PRADAS — COMPARAISON DES PERFORMANCES DES MODÈLES")
    print("=" * 70)
    header = f"{'Métrique':<22} | {'v1 (YOLOv8s CrowdHuman)':<22} | {'v2 (YOLOv11s Joint)':<22} | {'Gain':<10}"
    print(header)
    print("-" * len(header))

    metrics_def = [
        ("mAP@0.5 (mAP50)", "map50", True),
        ("mAP@0.5:0.95", "map50_95", True),
        ("Précision (Precision)", "precision", True),
        ("Rappel (Recall)", "recall", True),
        ("F1-Score", "f1", True),
        ("Perte Val (Box Loss)", "val_box_loss", False),
        ("Perte Val (Cls Loss)", "val_cls_loss", False),
    ]

    for label, key, is_pct in metrics_def:
        val1 = m1[key] if m1 else 0.0
        val2 = m2[key] if m2 else 0.0
        diff = val2 - val1
        if is_pct:
            s1 = f"{val1 * 100:.2f} %"
            s2 = f"{val2 * 100:.2f} %"
            sd = f"+{diff * 100:.2f} %" if diff >= 0 else f"{diff * 100:.2f} %"
        else:
            s1 = f"{val1:.4f}"
            s2 = f"{val2:.4f}"
            sd = f"{diff:.4f}"
        print(f"{label:<22} | {s1:<22} | {s2:<22} | {sd:<10}")

    print("=" * 70)

    # 1. Bar Chart Comparison
    if m1 and m2:
        labels = ["mAP@0.5", "mAP@0.5:0.95", "Précision", "Rappel", "F1-Score"]
        vals1 = [m1["map50"] * 100, m1["map50_95"] * 100, m1["precision"] * 100, m1["recall"] * 100, m1["f1"] * 100]
        vals2 = [m2["map50"] * 100, m2["map50_95"] * 100, m2["precision"] * 100, m2["recall"] * 100, m2["f1"] * 100]

        x = np.arange(len(labels))
        width = 0.35

        plt.figure(figsize=(10, 6))
        rects1 = plt.bar(x - width / 2, vals1, width, label="v1: YOLOv8s (CrowdHuman)", color="#3498db")
        rects2 = plt.bar(x + width / 2, vals2, width, label="v2: YOLOv11s (CrowdHuman + MOT17)", color="#2ecc71")

        plt.ylabel("Score (%)", fontsize=12)
        plt.title("PRADAS: Comparaison des Métriques v1 vs v2", fontsize=14, fontweight="bold")
        plt.xticks(x, labels, fontsize=11)
        plt.ylim(0, 105)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.legend(fontsize=11)

        # Label values on bars
        for rect in rects1:
            h = rect.get_height()
            plt.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)
        for rect in rects2:
            h = rect.get_height()
            plt.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        chart_path = out_dir / "model_comparison.png"
        plt.savefig(str(chart_path), dpi=200)
        plt.close()
        print(f"\n[OK] Graphique comparatif généré : {chart_path}")

    # 2. Learning Curves v2
    if m2 and m2.get("history"):
        hist = m2["history"]
        epochs = hist.get("epoch", [])
        if epochs:
            plt.figure(figsize=(12, 5))

            # Subplot 1: Losses
            plt.subplot(1, 2, 1)
            plt.plot(epochs, hist.get("train/box_loss", []), label="Train Box Loss", color="#e74c3c")
            plt.plot(epochs, hist.get("val/box_loss", []), label="Val Box Loss", color="#e67e22", linestyle="--")
            plt.plot(epochs, hist.get("train/cls_loss", []), label="Train Cls Loss", color="#9b59b6")
            plt.plot(epochs, hist.get("val/cls_loss", []), label="Val Cls Loss", color="#8e44ad", linestyle="--")
            plt.xlabel("Époque")
            plt.ylabel("Loss")
            plt.title("Évolution des Pertes (Losses) — v2")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()

            # Subplot 2: mAP, Precision, Recall
            plt.subplot(1, 2, 2)
            plt.plot(epochs, [x * 100 for x in hist.get("metrics/mAP50(B)", [])], label="mAP@0.5", color="#2ecc71", linewidth=2)
            plt.plot(epochs, [x * 100 for x in hist.get("metrics/precision(B)", [])], label="Précision", color="#3498db")
            plt.plot(epochs, [x * 100 for x in hist.get("metrics/recall(B)", [])], label="Rappel", color="#f39c12")
            plt.xlabel("Époque")
            plt.ylabel("Score (%)")
            plt.title("Progression des Métriques (mAP, P, R) — v2")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()

            plt.tight_layout()
            curves_path = out_dir / "training_curves_v2.png"
            plt.savefig(str(curves_path), dpi=200)
            plt.close()
            print(f"[OK] Courbes d'apprentissage générées : {curves_path}")


if __name__ == "__main__":
    main()

