from pathlib import Path

# ============================================================
# SugarBeet Hybrid Detection - Common Configuration
# ============================================================

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

PROJECT_ROOT = Path.home() / "Downloads" / "sugarbeet"


# ------------------------------------------------------------
# Dataset YAML files
# ------------------------------------------------------------

DATASET_70 = PROJECT_ROOT / "dataset_70_20_10_detection"
DATASET_80 = PROJECT_ROOT / "dataset_80_10_10_detection"


# ------------------------------------------------------------
# Hybrid experiment root
# ------------------------------------------------------------

HYBRID_ROOT = PROJECT_ROOT / "Experiments" / "Hybrid_Approaches"

CAF_ROOT = HYBRID_ROOT / "CAF"
MGKD_ROOT = HYBRID_ROOT / "MGKD"
DSTR_ROOT = HYBRID_ROOT / "DSTR"
RESULTS_ROOT = HYBRID_ROOT / "results"


# ------------------------------------------------------------
# Trained SugarBeet Models
# ------------------------------------------------------------

# =========================
# 70:20:10 trained models
# =========================

YOLO11_70_MODEL = (
    PROJECT_ROOT
    / "Experiments"
    / "70_20_10"
    / "YOLOv11_70_20_10_Detection"
    / "weights"
    / "best.pt"
)

YOLO12_70_MODEL = (
    PROJECT_ROOT
    / "Experiments"
    / "70_20_10"
    / "YOLOv12_70_20_10_Detection"
    / "weights"
    / "best.pt"
)

RTDETR_70_MODEL = (
    PROJECT_ROOT
    / "Experiments"
    / "70_20_10"
    / "RT_DETR_70_20_10"
    / "weights"
    / "best.pt"
)


# =========================
# 80:10:10 trained models
# =========================

YOLO11_80_MODEL = (
    PROJECT_ROOT
    / "Experiments"
    / "80_10_10"
    / "YOLOv11"
    / "detection"
    / "weights"
    / "best.pt"
)

YOLO12_80_MODEL = (
    PROJECT_ROOT
    / "Experiments"
    / "80_10_10"
    / "YOLOv12"
    / "detection-2"
    / "weights"
    / "best.pt"
)

RTDETR_80_MODEL = (
    PROJECT_ROOT
    / "Experiments"
    / "80_10_10"
    / "RT_DETR4"
    / "weights"
    / "best.pt"
)


# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

IMG_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 50


# ------------------------------------------------------------
# Evaluation configuration
# ------------------------------------------------------------

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50


# ------------------------------------------------------------
# MGKD loss weights
# ------------------------------------------------------------

LAMBDA_LOW = 0.05
LAMBDA_MID = 0.05
LAMBDA_HIGH = 0.10
LAMBDA_PRED = 0.10


# ------------------------------------------------------------
# Feature alignment
# ------------------------------------------------------------

COMMON_FEATURE_SIZES = [
    (80, 80),
    (40, 40),
    (20, 20),
]


# ------------------------------------------------------------
# WBF configuration
# ------------------------------------------------------------

WBF_IOU_THRESHOLD = 0.55
WBF_SKIP_THRESHOLD = 0.05


# ------------------------------------------------------------
# Create required directories
# ------------------------------------------------------------

for directory in [
    HYBRID_ROOT,
    CAF_ROOT,
    MGKD_ROOT,
    DSTR_ROOT,
    RESULTS_ROOT,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Configuration verification
# ------------------------------------------------------------

print("=" * 70)
print("SUGARBEET HYBRID APPROACH - COMMON CONFIGURATION")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nDataset YAML files:")
print("70:20:10:", DATASET_70)
print("80:10:10:", DATASET_80)

print("\nTrained SugarBeet models:")

trained_models = {
    "YOLOv11 70:20:10": YOLO11_70_MODEL,
    "YOLOv12 70:20:10": YOLO12_70_MODEL,
    "RT-DETR 70:20:10": RTDETR_70_MODEL,
    "YOLOv11 80:10:10": YOLO11_80_MODEL,
    "YOLOv12 80:10:10": YOLO12_80_MODEL,
    "RT-DETR 80:10:10": RTDETR_80_MODEL,
}

for name, model_path in trained_models.items():
    status = "YES" if model_path.exists() else "NO"
    print(f"{name:<25}: {status}")

print("\nTraining configuration:")
print("Image size :", IMG_SIZE)
print("Batch size :", BATCH_SIZE)
print("Epochs     :", EPOCHS)

print("\nEvaluation configuration:")
print("Confidence threshold:", CONF_THRESHOLD)
print("IoU threshold       :", IOU_THRESHOLD)

print("\nMGKD loss weights:")
print("Lambda low :", LAMBDA_LOW)
print("Lambda mid :", LAMBDA_MID)
print("Lambda high:", LAMBDA_HIGH)
print("Lambda pred:", LAMBDA_PRED)

print("\nWBF configuration:")
print("WBF IoU threshold :", WBF_IOU_THRESHOLD)
print("WBF skip threshold:", WBF_SKIP_THRESHOLD)

print("\n" + "=" * 70)
print("COMMON CONFIGURATION LOADED")
print("=" * 70)