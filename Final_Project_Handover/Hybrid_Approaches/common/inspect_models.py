from pathlib import Path
import sys

# ============================================================
# SugarBeet Hybrid Detection - Model Inspection
# ============================================================

COMMON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(COMMON_DIR))

from ultralytics import YOLO

from config import (
    YOLO11_70_MODEL,
    YOLO12_70_MODEL,
    RTDETR_70_MODEL,
    YOLO11_80_MODEL,
    YOLO12_80_MODEL,
    RTDETR_80_MODEL,
)


# ------------------------------------------------------------
# Model inspection function
# ------------------------------------------------------------

def inspect_model(model_name, model_path):

    print("\n" + "-" * 70)
    print(model_name)
    print("-" * 70)

    print("Model path:")
    print(model_path)

    if not Path(model_path).exists():
        print("ERROR: Model file not found.")
        return False

    print("Model file: YES")

    try:
        model = YOLO(str(model_path))

        print("Model loaded: YES")

        # Task
        task = getattr(model, "task", "Unknown")
        print("Task   :", task)

        # Classes
        names = model.names
        print("Classes:", names)

        # Number of classes
        if isinstance(names, dict):
            print("Number of classes:", len(names))

        # Check SugarBeet class
        if isinstance(names, dict) and names == {0: "sugarbeet"}:
            print("SugarBeet class check: PASS")
        else:
            print("SugarBeet class check: WARNING")

        return True

    except Exception as e:
        print("ERROR while loading model:")
        print(e)
        return False


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("SUGARBEET HYBRID APPROACH - MODEL INSPECTION")
    print("=" * 70)

    print("\nChecking trained SugarBeet detection models...")

    models = [
        ("YOLOv11 - 70:20:10", YOLO11_70_MODEL),
        ("YOLOv12 - 70:20:10", YOLO12_70_MODEL),
        ("RT-DETR - 70:20:10", RTDETR_70_MODEL),

        ("YOLOv11 - 80:10:10", YOLO11_80_MODEL),
        ("YOLOv12 - 80:10:10", YOLO12_80_MODEL),
        ("RT-DETR - 80:10:10", RTDETR_80_MODEL),
    ]

    results = {}

    for model_name, model_path in models:
        results[model_name] = inspect_model(
            model_name,
            model_path
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL INSPECTION SUMMARY")
    print("=" * 70)

    for model_name, status in results.items():
        print(
            f"{model_name:<30}: "
            f"{'PASS' if status else 'FAIL'}"
        )

    print("\n" + "=" * 70)

    if all(results.values()):
        print("ALL TRAINED MODELS VERIFIED SUCCESSFULLY")
    else:
        print("WARNING: ONE OR MORE MODELS FAILED VERIFICATION")

    print("=" * 70)


if __name__ == "__main__":
    main()