import os
import glob
import shutil
from openpyxl import load_workbook
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE = "/Users/kalamanjunatha/Downloads/sugarbeet/Experiments"

# Find the Excel workbook automatically
excel_files = glob.glob(
    os.path.join(BASE, "**", "Sugarbeet_80_10_10_Model_Comparison*.xlsx"),
    recursive=True
)

if not excel_files:
    print("ERROR: Excel file not found.")
    print("Make sure the Excel file is inside the Experiments folder.")
    exit()

EXCEL = excel_files[0]

print("Excel file:")
print(EXCEL)


# ============================================================
# MODEL WEIGHTS
# ============================================================

weights = {
    "YOLOv11": os.path.join(
        BASE,
        "80_10_10",
        "YOLOv11",
        "detection",
        "weights",
        "best.pt"
    ),

    "YOLOv12": os.path.join(
        BASE,
        "80_10_10",
        "YOLOv12",
        "detection",
        "weights",
        "best.pt"
    ),

    "RT_DETR4": os.path.join(
        BASE,
        "80_10_10",
        "RT_DETR4",
        "weights",
        "best.pt"
    )
}


# ============================================================
# FIND WEIGHT IF PATH DIFFERENT
# ============================================================

def find_best_pt(model_name):

    if os.path.exists(weights[model_name]):
        return weights[model_name]

    print(f"\nDefault weight not found for {model_name}. Searching...")

    matches = glob.glob(
        os.path.join(
            BASE,
            "**",
            "best.pt"
        ),
        recursive=True
    )

    for m in matches:

        if model_name.lower() in m.lower():

            print("Found:")
            print(m)

            return m

    return None


# ============================================================
# LOAD WORKBOOK
# ============================================================

wb = load_workbook(EXCEL)

ws = wb["Model Comparison"]


# ============================================================
# HEADER MAP
# ============================================================

headers = {}

for col in range(1, ws.max_column + 1):

    value = ws.cell(
        row=1,
        column=col
    ).value

    if value:
        headers[value] = col


print("\nExcel columns:")
for h in headers:
    print(h)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

configs = {

    "YOLOv11": {
        "batch": 8,
        "imgsz": 640,
        "optimizer": "auto"
    },

    "YOLOv12": {
        "batch": 8,
        "imgsz": 640,
        "optimizer": "auto"
    },

    "RT_DETR4": {
        "batch": None,
        "imgsz": 640,
        "optimizer": None
    }
}


# ============================================================
# PROCESS DETECTION ROWS ONLY
# ============================================================

for row in range(2, ws.max_row + 1):

    model = ws.cell(
        row=row,
        column=headers["Model"]
    ).value

    task = ws.cell(
        row=row,
        column=headers["Task"]
    ).value

    split = ws.cell(
        row=row,
        column=headers["Split"]
    ).value

    # --------------------------------------------------------
    # ONLY 80:10:10 DETECTION
    # --------------------------------------------------------

    if split != "80:10:10":
        continue

    if task != "Detection":
        continue

    if model not in configs:
        continue

    print("\n" + "=" * 60)
    print(f"Processing {model} - {split}")
    print("=" * 60)

    config = configs[model]

    # --------------------------------------------------------
    # BATCH SIZE
    # --------------------------------------------------------

    if config["batch"] is not None:

        ws.cell(
            row=row,
            column=headers["Batch Size"]
        ).value = config["batch"]

    # --------------------------------------------------------
    # IMAGE SIZE
    # --------------------------------------------------------

    ws.cell(
        row=row,
        column=headers["Image Size"]
    ).value = config["imgsz"]

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    if config["optimizer"] is not None:

        ws.cell(
            row=row,
            column=headers["Optimizer"]
        ).value = config["optimizer"]

    # --------------------------------------------------------
    # TRAINING TIME
    # --------------------------------------------------------

    training_time = ws.cell(
        row=row,
        column=headers["Training Time"]
    ).value

    if training_time is not None:

        hours = float(training_time) / 3600

        print(
            f"Training time: "
            f"{training_time:.1f} sec "
            f"({hours:.2f} hours)"
        )

    # --------------------------------------------------------
    # BEST EPOCH
    # --------------------------------------------------------

    best_map = ws.cell(
        row=row,
        column=headers["Best mAP50-95"]
    ).value

    best_epoch = ws.cell(
        row=row,
        column=headers["Best Epoch"]
    ).value

    print(
        f"Best mAP50-95: {best_map}"
    )

    print(
        f"Best Epoch: {best_epoch}"
    )

    # --------------------------------------------------------
    # MODEL WEIGHT
    # --------------------------------------------------------

    weight_path = find_best_pt(model)

    if not weight_path:

        print(
            f"WARNING: best.pt not found for {model}"
        )

        print(
            "Parameters/GFLOPs/Model Size will remain "
            "Not Available."
        )

        continue

    print(
        f"Using weights:\n{weight_path}"
    )

    # --------------------------------------------------------
    # MODEL SIZE
    # --------------------------------------------------------

    size_bytes = os.path.getsize(
        weight_path
    )

    size_mb = size_bytes / (
        1024 * 1024
    )

    ws.cell(
        row=row,
        column=headers["Model Size (MB)"]
    ).value = round(
        size_mb,
        2
    )

    print(
        f"Model size: {size_mb:.2f} MB"
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    try:

        model_obj = YOLO(
            weight_path
        )

        # ----------------------------------------------------
        # MODEL INFO
        # ----------------------------------------------------

        info = model_obj.info(
            verbose=False
        )

        print(
            "Model information obtained."
        )

        # Ultralytics model.info() returns:
        # (layers, parameters, gradients, GFLOPs)

        if isinstance(info, tuple):

            if len(info) >= 4:

                layers = info[0]
                params = info[1]
                gradients = info[2]
                gflops = info[3]

                ws.cell(
                    row=row,
                    column=headers["Parameters"]
                ).value = int(
                    params
                )

                ws.cell(
                    row=row,
                    column=headers["GFLOPs"]
                ).value = round(
                    float(gflops),
                    2
                )

                print(
                    f"Parameters: {params:,}"
                )

                print(
                    f"GFLOPs: {gflops}"
                )

        # ----------------------------------------------------
        # INFERENCE SPEED
        # ----------------------------------------------------

        print(
            "\nRunning validation to obtain "
            "measured inference time..."
        )

        data_yaml = os.path.join(
            BASE,
            "80_10_10",
            "sugarbeet_80_10_10.yaml"
        )

        if not os.path.exists(data_yaml):

            # Search if YAML is elsewhere
            yaml_matches = glob.glob(
                os.path.join(
                    BASE,
                    "**",
                    "sugarbeet_80_10_10*.yaml"
                ),
                recursive=True
            )

            if yaml_matches:
                data_yaml = yaml_matches[0]

        if os.path.exists(data_yaml):

            results = model_obj.val(
                data=data_yaml,
                split="val",
                imgsz=640,
                batch=1,
                verbose=False
            )

            if hasattr(results, "speed"):

                inference_ms = results.speed.get(
                    "inference"
                )

                if inference_ms is not None:

                    ws.cell(
                        row=row,
                        column=headers[
                            "Inference Time (ms/image)"
                        ]
                    ).value = round(
                        float(inference_ms),
                        2
                    )

                    fps = (
                        1000 /
                        float(inference_ms)
                    )

                    ws.cell(
                        row=row,
                        column=headers["FPS"]
                    ).value = round(
                        fps,
                        2
                    )

                    print(
                        f"Inference: "
                        f"{inference_ms:.2f} ms/image"
                    )

                    print(
                        f"FPS: {fps:.2f}"
                    )

        else:

            print(
                "WARNING: 80:10:10 YAML not found."
            )

    except Exception as e:

        print(
            f"WARNING: Could not extract "
            f"model information for {model}"
        )

        print(
            "Reason:",
            e
        )


# ============================================================
# SAVE
# ============================================================

output = os.path.join(
    BASE,
    "Sugarbeet_80_10_10_Model_Comparison_FILLED.xlsx"
)

wb.save(output)

print("\n")
print("=" * 60)
print("COMPLETED")
print("=" * 60)
print("New Excel file:")
print(output)
print("=" * 60)