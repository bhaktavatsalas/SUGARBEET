import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE = "/Users/kalamanjunatha/Downloads/sugarbeet/Experiments"

OUTPUT_DIR = os.path.join(BASE, "Training_Progress_Graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# EXPERIMENT FILES
# ============================================================

experiments = {
    "YOLOv11_70_20_10": {
        "model": "YOLOv11 Nano",
        "split": "70:20:10",
        "path": os.path.join(
            BASE,
            "70_20_10",
            "YOLOv11_70_20_10_Detection",
            "results.csv"
        ),
        "type": "yolo"
    },

    "YOLOv12_70_20_10": {
        "model": "YOLOv12 Nano",
        "split": "70:20:10",
        "path": os.path.join(
            BASE,
            "70_20_10",
            "YOLOv12_70_20_10_Detection",
            "train",
            "results.csv"
        ),
        "type": "yolo"
    },

    "RT_DETR_70_20_10": {
        "model": "RT-DETR",
        "split": "70:20:10",
        "path": os.path.join(
            BASE,
            "70_20_10",
            "RT_DETR_70_20_10",
            "results.csv"
        ),
        "type": "rtdetr"
    },

    "YOLOv11_80_10_10": {
        "model": "YOLOv11 Nano",
        "split": "80:10:10",
        "path": os.path.join(
            BASE,
            "80_10_10",
            "YOLOv11",
            "detection",
            "results.csv"
        ),
        "type": "yolo"
    },

    "YOLOv12_80_10_10": {
        "model": "YOLOv12 Nano",
        "split": "80:10:10",
        "path": os.path.join(
            BASE,
            "80_10_10",
            "YOLOv12",
            "detection",
            "results.csv"
        ),
        "type": "yolo"
    },

    "RT_DETR_80_10_10": {
        "model": "RT-DETR",
        "split": "80:10:10",
        "path": os.path.join(
            BASE,
            "80_10_10",
            "RT_DETR4",
            "results.csv"
        ),
        "type": "rtdetr"
    }
}


# ============================================================
# COLUMN CLEANING
# ============================================================

def clean_columns(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


# ============================================================
# F1 CALCULATION
# ============================================================

def calculate_f1(precision, recall):
    denominator = precision + recall

    f1 = (2 * precision * recall) / denominator

    f1 = f1.where(denominator != 0, 0)

    return f1


# ============================================================
# CREATE GRAPH
# ============================================================

def create_graph(name, config):

    path = config["path"]
    model = config["model"]
    split = config["split"]
    experiment_type = config["type"]

    print("\n" + "=" * 70)
    print(f"Processing: {model} | Split: {split}")
    print(f"File: {path}")
    print("=" * 70)

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not os.path.exists(path):
        print("WARNING: results.csv NOT FOUND")
        print("Skipping this experiment.")
        return

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    df = pd.read_csv(path)
    df = clean_columns(df)

    print(f"Epochs found: {len(df)}")
    print("Columns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Epoch
    # --------------------------------------------------------

    epoch_col = find_column(
        df,
        ["epoch"]
    )

    if epoch_col is None:
        print("ERROR: Epoch column not found.")
        return

    epochs = df[epoch_col]

    # ========================================================
    # YOLO
    # ========================================================

    if experiment_type == "yolo":

        train_loss_col = find_column(
            df,
            [
                "train/box_loss",
                "train/box_loss"
            ]
        )

        val_loss_col = find_column(
            df,
            [
                "val/box_loss",
                "val/box_loss"
            ]
        )

        precision_col = find_column(
            df,
            [
                "metrics/precision(B)"
            ]
        )

        recall_col = find_column(
            df,
            [
                "metrics/recall(B)"
            ]
        )

        map50_col = find_column(
            df,
            [
                "metrics/mAP50(B)"
            ]
        )

        map5095_col = find_column(
            df,
            [
                "metrics/mAP50-95(B)"
            ]
        )

        loss_label = "Box Loss"

    # ========================================================
    # RT-DETR
    # ========================================================

    else:

        train_loss_col = find_column(
            df,
            [
                "train/giou_loss"
            ]
        )

        val_loss_col = find_column(
            df,
            [
                "val/giou_loss"
            ]
        )

        precision_col = find_column(
            df,
            [
                "metrics/precision(B)"
            ]
        )

        recall_col = find_column(
            df,
            [
                "metrics/recall(B)"
            ]
        )

        map50_col = find_column(
            df,
            [
                "metrics/mAP50(B)"
            ]
        )

        map5095_col = find_column(
            df,
            [
                "metrics/mAP50-95(B)"
            ]
        )

        loss_label = "GIoU Loss"

    # ========================================================
    # PRINT FOUND COLUMNS
    # ========================================================

    print("\nSelected columns:")

    print("Train Loss :", train_loss_col)
    print("Val Loss   :", val_loss_col)
    print("Precision  :", precision_col)
    print("Recall     :", recall_col)
    print("mAP50      :", map50_col)
    print("mAP50-95   :", map5095_col)

    # ========================================================
    # F1
    # ========================================================

    if precision_col and recall_col:

        precision = pd.to_numeric(
            df[precision_col],
            errors="coerce"
        )

        recall = pd.to_numeric(
            df[recall_col],
            errors="coerce"
        )

        f1 = calculate_f1(
            precision,
            recall
        )

    else:

        f1 = None

    # ========================================================
    # FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10)
    )

    fig.suptitle(
        f"{model} — {split} Training Progress",
        fontsize=20,
        fontweight="bold"
    )

    # ========================================================
    # 1. LOSS
    # ========================================================

    ax = axes[0, 0]

    if train_loss_col:
        ax.plot(
            epochs,
            df[train_loss_col],
            marker="o",
            markersize=3,
            linewidth=2,
            label="Train Loss"
        )

    if val_loss_col:
        ax.plot(
            epochs,
            df[val_loss_col],
            marker="s",
            markersize=3,
            linewidth=2,
            linestyle="--",
            label="Validation Loss"
        )

    ax.set_title(
        f"{model}: {loss_label} vs Epoch",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(loss_label)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ========================================================
    # 2. PRECISION & RECALL
    # ========================================================

    ax = axes[0, 1]

    if precision_col:
        ax.plot(
            epochs,
            df[precision_col],
            marker="o",
            markersize=3,
            linewidth=2,
            label="Precision"
        )

    if recall_col:
        ax.plot(
            epochs,
            df[recall_col],
            marker="s",
            markersize=3,
            linewidth=2,
            linestyle="--",
            label="Recall"
        )

    ax.set_title(
        f"{model}: Precision & Recall vs Epoch",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ========================================================
    # 3. mAP
    # ========================================================

    ax = axes[1, 0]

    if map50_col:
        ax.plot(
            epochs,
            df[map50_col],
            marker="o",
            markersize=3,
            linewidth=2,
            label="mAP@50"
        )

    if map5095_col:
        ax.plot(
            epochs,
            df[map5095_col],
            marker="s",
            markersize=3,
            linewidth=2,
            linestyle="--",
            label="mAP@50-95"
        )

    ax.set_title(
        f"{model}: mAP vs Epoch",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ========================================================
    # 4. F1
    # ========================================================

    ax = axes[1, 1]

    if f1 is not None:

        ax.plot(
            epochs,
            f1,
            marker="o",
            markersize=3,
            linewidth=2,
            label="Derived F1 Score"
        )

        ax.set_ylim(0, 1)

    else:

        ax.text(
            0.5,
            0.5,
            "F1 Score Not Available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14
        )

    ax.set_title(
        f"{model}: F1 Score vs Epoch",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ========================================================
    # LAYOUT
    # ========================================================

    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{name}_Training_Progress.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nSaved:")
    print(output_file)


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

print("\n")
print("=" * 70)
print("SUGARBEET MODEL TRAINING PROGRESS GRAPH GENERATOR")
print("=" * 70)

for name, config in experiments.items():

    create_graph(
        name,
        config
    )


print("\n")
print("=" * 70)
print("ALL GRAPH GENERATION COMPLETED")
print("=" * 70)
print(f"Output folder:")
print(OUTPUT_DIR)
print("=" * 70)