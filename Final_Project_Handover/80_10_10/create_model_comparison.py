import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.getcwd()

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "Sugarbeet_80_10_10_Model_Comparison.xlsx"
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, possible_names):
    """
    Finds the first matching column from a list of possible names.
    """
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def calculate_f1(precision, recall):
    """
    Calculate F1 score from precision and recall.
    """
    if precision is None or recall is None:
        return None

    if precision + recall == 0:
        return 0

    return 2 * precision * recall / (precision + recall)


def get_model_name(path):
    """
    Identify the model from the folder path.
    """

    path_lower = path.lower()

    if "yolov11" in path_lower:
        return "YOLOv11"

    if "yolov12" in path_lower:
        return "YOLOv12"

    if "rt-detr4" in path_lower:
        return "RT-DETR4"

    if "rtdetr4" in path_lower:
        return "RT-DETR4"

    if "rt-detr" in path_lower:
        return "RT-DETR"

    if "rtdetr" in path_lower:
        return "RT-DETR"

    return os.path.basename(os.path.dirname(path))


def get_task(path):
    """
    Identify whether the result belongs to detection or segmentation.
    """

    path_lower = path.lower()

    if "segment" in path_lower:
        return "Segmentation"

    # If mask metrics exist, it is segmentation.
    try:
        df = pd.read_csv(path)

        mask_columns = [
            c for c in df.columns
            if "seg" in c.lower() or "mask" in c.lower()
        ]

        if mask_columns:
            return "Segmentation"

    except:
        pass

    return "Detection"


# ============================================================
# FIND ALL RESULTS.CSV FILES
# ============================================================

print("\nSearching for results.csv files...\n")

results_files = []

for root, dirs, files in os.walk(BASE_DIR):

    # Ignore virtual environment
    if ".venv" in root or "venv" in root:
        continue

    if "results.csv" in files:
        full_path = os.path.join(root, "results.csv")
        results_files.append(full_path)


if not results_files:
    print("ERROR: No results.csv files were found.")
    print("Make sure you are inside the 80_10_10 folder.")
    exit()


print("Found results files:")

for file in results_files:
    print(" -", file)

# ============================================================
# EXTRACT METRICS
# ============================================================

rows = []

for csv_file in results_files:

    try:

        df = pd.read_csv(csv_file)

        if df.empty:
            continue

        # Remove accidental spaces from column names
        df.columns = [str(c).strip() for c in df.columns]

        # Last epoch
        last = df.iloc[-1]

        # Best epoch based on mAP50-95
        map5095_col = find_column(
            df,
            [
                "metrics/mAP50-95(B)",
                "metrics/mAP50-95(B) "
            ]
        )

        best_row = None

        if map5095_col:
            best_index = df[map5095_col].idxmax()
            best_row = df.loc[best_index]

        # ----------------------------------------------------
        # BOX METRICS
        # ----------------------------------------------------

        precision_col = find_column(
            df,
            ["metrics/precision(B)"]
        )

        recall_col = find_column(
            df,
            ["metrics/recall(B)"]
        )

        map50_col = find_column(
            df,
            ["metrics/mAP50(B)"]
        )

        map5095_col = find_column(
            df,
            ["metrics/mAP50-95(B)"]
        )

        precision = (
            float(last[precision_col])
            if precision_col else None
        )

        recall = (
            float(last[recall_col])
            if recall_col else None
        )

        map50 = (
            float(last[map50_col])
            if map50_col else None
        )

        map5095 = (
            float(last[map5095_col])
            if map5095_col else None
        )

        f1 = calculate_f1(precision, recall)

        # ----------------------------------------------------
        # MASK METRICS
        # ----------------------------------------------------

        mask_precision_col = find_column(
            df,
            ["metrics/precision(M)"]
        )

        mask_recall_col = find_column(
            df,
            ["metrics/recall(M)"]
        )

        mask_map50_col = find_column(
            df,
            ["metrics/mAP50(M)"]
        )

        mask_map5095_col = find_column(
            df,
            ["metrics/mAP50-95(M)"]
        )

        mask_precision = (
            float(last[mask_precision_col])
            if mask_precision_col else None
        )

        mask_recall = (
            float(last[mask_recall_col])
            if mask_recall_col else None
        )

        mask_map50 = (
            float(last[mask_map50_col])
            if mask_map50_col else None
        )

        mask_map5095 = (
            float(last[mask_map5095_col])
            if mask_map5095_col else None
        )

        mask_f1 = calculate_f1(
            mask_precision,
            mask_recall
        )

        # ----------------------------------------------------
        # TRAINING INFORMATION
        # ----------------------------------------------------

        epoch_col = find_column(
            df,
            ["epoch"]
        )

        time_col = find_column(
            df,
            ["time"]
        )

        train_box_col = find_column(
            df,
            ["train/box_loss"]
        )

        train_cls_col = find_column(
            df,
            ["train/cls_loss"]
        )

        train_dfl_col = find_column(
            df,
            ["train/dfl_loss"]
        )

        val_box_col = find_column(
            df,
            ["val/box_loss"]
        )

        val_cls_col = find_column(
            df,
            ["val/cls_loss"]
        )

        val_dfl_col = find_column(
            df,
            ["val/dfl_loss"]
        )

        epochs = (
            int(last[epoch_col]) + 1
            if epoch_col else None
        )

        training_time = (
            float(last[time_col])
            if time_col else None
        )

        # ----------------------------------------------------
        # BEST METRICS
        # ----------------------------------------------------

        if best_row is not None:

            best_map5095 = (
                float(best_row[map5095_col])
                if map5095_col else None
            )

            best_epoch = (
                int(best_row[epoch_col]) + 1
                if epoch_col else None
            )

        else:

            best_map5095 = None
            best_epoch = None

        # ----------------------------------------------------
        # MODEL INFORMATION FROM PATH
        # ----------------------------------------------------

        model = get_model_name(csv_file)

        task = get_task(csv_file)

        # Avoid accidentally including duplicate evaluation files
        relative_path = os.path.relpath(
            csv_file,
            BASE_DIR
        )

        # ----------------------------------------------------
        # CREATE ROW
        # ----------------------------------------------------

        row = {

            "Model": model,

            "Task": task,

            "Split": "80:10:10",

            "Results File": relative_path,

            "Precision (Box)": precision,

            "Recall (Box)": recall,

            "F1 (Box)": f1,

            "mAP50 (Box)": map50,

            "mAP50-95 (Box)": map5095,

            "Best mAP50-95": best_map5095,

            "Best Epoch": best_epoch,

            "Precision (Mask)": mask_precision,

            "Recall (Mask)": mask_recall,

            "F1 (Mask)": mask_f1,

            "mAP50 (Mask)": mask_map50,

            "mAP50-95 (Mask)": mask_map5095,

            "Final Epoch": epochs,

            "Training Time": training_time,

            "Train Box Loss": (
                float(last[train_box_col])
                if train_box_col else None
            ),

            "Train CLS Loss": (
                float(last[train_cls_col])
                if train_cls_col else None
            ),

            "Train DFL Loss": (
                float(last[train_dfl_col])
                if train_dfl_col else None
            ),

            "Validation Box Loss": (
                float(last[val_box_col])
                if val_box_col else None
            ),

            "Validation CLS Loss": (
                float(last[val_cls_col])
                if val_cls_col else None
            ),

            "Validation DFL Loss": (
                float(last[val_dfl_col])
                if val_dfl_col else None
            ),

            # These are intentionally blank unless actually recorded.
            "Parameters": None,

            "GFLOPs": None,

            "Model Size (MB)": None,

            "Inference Time (ms/image)": None,

            "FPS": None,

            "Batch Size": None,

            "Image Size": None,

            "Optimizer": None
        }

        rows.append(row)

    except Exception as e:

        print(
            f"\nCould not process {csv_file}"
        )

        print("Error:", e)


# ============================================================
# CREATE DATAFRAME
# ============================================================

comparison_df = pd.DataFrame(rows)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

comparison_df = comparison_df.drop_duplicates(
    subset=["Model", "Task"],
    keep="first"
)


# ============================================================
# CREATE EXCEL FILE
# ============================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    # Main comparison sheet
    comparison_df.to_excel(
        writer,
        sheet_name="Model Comparison",
        index=False
    )

    # --------------------------------------------------------
    # Metric Definitions
    # --------------------------------------------------------

    metric_guide = pd.DataFrame({

        "Metric": [

            "Precision",
            "Recall",
            "F1",
            "mAP50",
            "mAP50-95",
            "Parameters",
            "GFLOPs",
            "Inference Time",
            "FPS",
            "Training Loss"

        ],

        "Meaning": [

            "Proportion of predicted objects that are correct.",

            "Proportion of actual objects that are detected.",

            "Harmonic mean of Precision and Recall.",

            "Mean Average Precision at IoU = 0.50.",

            "Mean Average Precision averaged across IoU 0.50–0.95.",

            "Number of trainable model parameters.",

            "Approximate computational complexity.",

            "Time required to process one image.",

            "Frames/images processed per second.",

            "Loss value showing training error."
        ]
    })

    metric_guide.to_excel(
        writer,
        sheet_name="Metric Guide",
        index=False
    )


# ============================================================
# FORMAT EXCEL
# ============================================================

from openpyxl import load_workbook

wb = load_workbook(OUTPUT_FILE)

ws = wb["Model Comparison"]

# Header formatting
for cell in ws[1]:

    cell.font = Font(
        bold=True
    )

    cell.alignment = Alignment(
        horizontal="center"
    )


# Freeze header
ws.freeze_panes = "A2"

# Autofilter
ws.auto_filter.ref = ws.dimensions


# Adjust column widths
for column_cells in ws.columns:

    max_length = 0

    column_letter = get_column_letter(
        column_cells[0].column
    )

    for cell in column_cells:

        try:

            if cell.value is not None:

                max_length = max(
                    max_length,
                    len(str(cell.value))
                )

        except:
            pass

    ws.column_dimensions[
        column_letter
    ].width = min(max_length + 2, 35)


# Format metric values
for row in ws.iter_rows(
    min_row=2,
    max_row=ws.max_row
):

    for cell in row:

        if isinstance(cell.value, float):

            cell.number_format = "0.000"


# Format metric guide
guide_ws = wb["Metric Guide"]

for cell in guide_ws[1]:

    cell.font = Font(
        bold=True
    )

guide_ws.freeze_panes = "A2"


# Save
wb.save(OUTPUT_FILE)


# ============================================================
# FINISHED
# ============================================================

print("\n========================================")
print("EXCEL FILE CREATED SUCCESSFULLY")
print("========================================")

print("\nLocation:")
print(OUTPUT_FILE)

print("\nModels found:")

for model in comparison_df["Model"].unique():

    print(" -", model)

print("\nOpen the Excel file from the VS Code Explorer.")