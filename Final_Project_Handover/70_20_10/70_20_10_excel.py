from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.chart import BarChart, Reference
from openpyxl.utils import get_column_letter

# ============================================================
# 70:20:10 SUGARBEEТ DETECTION MODEL COMPARISON
# YOLOv11n vs YOLOv12n vs RT-DETR
# ============================================================

models = [
    {
        "Model": "YOLOv11n",
        "Task": "Detection",
        "Train Images": 345,
        "Val Images": 98,
        "Test Images": 51,
        "Test Instances": 164,
        "Epochs": 50,
        "Batch": 8,
        "Image Size": 640,

        "Val Precision": 0.68799,
        "Val Recall": 0.67143,
        "Val mAP50": 0.68483,
        "Val mAP50-95": 0.41054,

        "Test Precision": 0.723,
        "Test Recall": 0.591,
        "Test mAP50": 0.636,
        "Test mAP50-95": 0.365,

        "Inference ms/img": "Not recorded",

        "Train Box Loss": 1.14517,
        "Train Cls Loss": 0.74700,
        "Train DFL Loss": 0.81033,

        "Parameters": "Not recorded",
        "GFLOPs": "Not recorded",

        "Notes": "YOLOv11 Nano detection. Test metrics obtained using best.pt."
    },

    {
        "Model": "YOLOv12n",
        "Task": "Detection",
        "Train Images": 345,
        "Val Images": 98,
        "Test Images": 51,
        "Test Instances": 164,
        "Epochs": 50,
        "Batch": 8,
        "Image Size": 640,

        "Val Precision": 0.705,
        "Val Recall": 0.639,
        "Val mAP50": 0.674,
        "Val mAP50-95": 0.413,

        "Test Precision": 0.742,
        "Test Recall": 0.573,
        "Test mAP50": 0.626,
        "Test mAP50-95": 0.358,

        "Inference ms/img": 164.4,

        "Train Box Loss": "See results.csv",
        "Train Cls Loss": "See results.csv",
        "Train DFL Loss": "See results.csv",

        "Parameters": 2556923,
        "GFLOPs": 7.3,

        "Notes": "YOLOv12 Nano detection. Test metrics obtained using best.pt."
    },

    {
        "Model": "RT-DETR",
        "Task": "Detection",
        "Train Images": 345,
        "Val Images": 98,
        "Test Images": 51,
        "Test Instances": 164,
        "Epochs": 50,
        "Batch": "Not recorded",
        "Image Size": 640,

        "Val Precision": 0.80521,
        "Val Recall": 0.71324,
        "Val mAP50": 0.81403,
        "Val mAP50-95": 0.48988,

        "Test Precision": 0.841,
        "Test Recall": 0.743,
        "Test mAP50": 0.809,
        "Test mAP50-95": 0.509,

        "Inference ms/img": 1314.2,

        "Train Box Loss": "N/A",
        "Train Cls Loss": 0.469,
        "Train DFL Loss": "N/A",

        "Parameters": 31985795,
        "GFLOPs": 105.3,

        "Notes": "RT-DETR-L detection. Uses GIoU, classification and L1 losses."
    }
]


# ============================================================
# DERIVED F1 SCORE
# ============================================================

for model in models:

    precision = model["Test Precision"]
    recall = model["Test Recall"]

    model["Test F1"] = (
        2 * precision * recall / (precision + recall)
    )


# ============================================================
# CREATE WORKBOOK
# ============================================================

workbook = Workbook()

sheet = workbook.active
sheet.title = "70_20_10 Comparison"


# ============================================================
# MAIN COMPARISON TABLE
# ============================================================

headers = [
    "Model",
    "Task",
    "Train Images",
    "Val Images",
    "Test Images",
    "Test Instances",
    "Epochs",
    "Batch",
    "Image Size",
    "Parameters",
    "GFLOPs",

    "Val Precision",
    "Val Recall",
    "Val mAP50",
    "Val mAP50-95",

    "Test Precision",
    "Test Recall",
    "Test F1",
    "Test mAP50",
    "Test mAP50-95",

    "Inference ms/img",

    "Train Box Loss",
    "Train Cls Loss",
    "Train DFL Loss",

    "Notes"
]

sheet.append(headers)


for model in models:

    row = [
        model["Model"],
        model["Task"],
        model["Train Images"],
        model["Val Images"],
        model["Test Images"],
        model["Test Instances"],
        model["Epochs"],
        model["Batch"],
        model["Image Size"],
        model["Parameters"],
        model["GFLOPs"],

        model["Val Precision"],
        model["Val Recall"],
        model["Val mAP50"],
        model["Val mAP50-95"],

        model["Test Precision"],
        model["Test Recall"],
        model["Test F1"],
        model["Test mAP50"],
        model["Test mAP50-95"],

        model["Inference ms/img"],

        model["Train Box Loss"],
        model["Train Cls Loss"],
        model["Train DFL Loss"],

        model["Notes"]
    ]

    sheet.append(row)


# ============================================================
# TRAINING CONFIGURATION SHEET
# ============================================================

config = workbook.create_sheet("Training Configuration")

config_headers = [
    "Model",
    "Model File",
    "Task",
    "Dataset",
    "Epochs",
    "Batch Size",
    "Image Size",
    "Initial LR",
    "LRF",
    "Warmup Epochs",
    "Optimizer",
    "Augmentation"
]

config.append(config_headers)

config_data = [

    [
        "YOLOv11n",
        "yolo11n.pt",
        "Detection",
        "sugarbeet_70_20_10.yaml",
        50,
        8,
        640,
        0.01,
        0.01,
        3,
        "Auto",
        "HSV, translation, scale, horizontal flip, mosaic, RandAugment, erasing"
    ],

    [
        "YOLOv12n",
        "yolo12n.pt",
        "Detection",
        "sugarbeet_70_20_10.yaml",
        50,
        8,
        640,
        0.01,
        0.01,
        3,
        "Auto",
        "HSV, translation, scale, horizontal flip, mosaic, RandAugment, erasing"
    ],

    [
        "RT-DETR",
        "rtdetr-l.pt",
        "Detection",
        "sugarbeet_70_20_10.yaml",
        50,
        "Not recorded",
        640,
        "Not recorded",
        "Not recorded",
        "Not recorded",
        "Not recorded",
        "Detection model"
    ]
]

for row in config_data:
    config.append(row)


# ============================================================
# TEST CONFUSION MATRIX INFORMATION
# ============================================================

cm = workbook.create_sheet("Test Confusion Matrix")

cm.append([
    "Model",
    "TP",
    "FP",
    "FN",
    "TN",
    "Note"
])

cm.append([
    "YOLOv11n",
    "Not extracted",
    "Not extracted",
    "Not extracted",
    "N/A",
    "TN is not a standard meaningful metric for object detection."
])

cm.append([
    "YOLOv12n",
    193,
    81,
    87,
    "N/A",
    "Counts from displayed YOLOv12 test confusion matrix."
])

cm.append([
    "RT-DETR",
    117,
    110,
    19,
    "N/A",
    "Counts from displayed RT-DETR test confusion matrix."
])


# ============================================================
# MODEL RANKING
# ============================================================

ranking = workbook.create_sheet("Test Ranking")

ranking.append([
    "Rank",
    "Model",
    "Precision",
    "Recall",
    "F1",
    "mAP50",
    "mAP50-95",
    "Inference ms/img"
])


ranked_models = sorted(
    models,
    key=lambda x: x["Test mAP50-95"],
    reverse=True
)


for rank, model in enumerate(ranked_models, start=1):

    ranking.append([
        rank,
        model["Model"],
        model["Test Precision"],
        model["Test Recall"],
        model["Test F1"],
        model["Test mAP50"],
        model["Test mAP50-95"],
        model["Inference ms/img"]
    ])


# ============================================================
# CHART
# ============================================================

chart = BarChart()

chart.title = "70:20:10 Test mAP50-95 Comparison"
chart.y_axis.title = "mAP50-95"
chart.x_axis.title = "Model"

data = Reference(
    ranking,
    min_col=7,
    min_row=1,
    max_row=4
)

categories = Reference(
    ranking,
    min_col=2,
    min_row=2,
    max_row=4
)

chart.add_data(
    data,
    titles_from_data=True
)

chart.set_categories(categories)

ranking.add_chart(
    chart,
    "J2"
)


# ============================================================
# SOURCE / NOTES SHEET
# ============================================================

source = workbook.create_sheet("Notes")

notes = [

    ["Item", "Details"],

    [
        "Dataset Split",
        "70:20:10 detection dataset: 345 train, 98 validation, 51 test."
    ],

    [
        "YOLOv11 Validation",
        "Values from the YOLOv11 70:20:10 detection results.csv."
    ],

    [
        "YOLOv11 Test",
        "Values from separate split=test evaluation using best.pt."
    ],

    [
        "YOLOv12 Validation",
        "Values from completed YOLOv12 70:20:10 detection validation output."
    ],

    [
        "YOLOv12 Test",
        "Values from separate split=test evaluation using best.pt."
    ],

    [
        "RT-DETR Validation",
        "Values from RT-DETR 70:20:10 detection results.csv."
    ],

    [
        "RT-DETR Test",
        "Values from separate split=test evaluation using best.pt."
    ],

    [
        "F1",
        "Test F1 is derived from Precision and Recall."
    ],

    [
        "TN",
        "Not included as a standard classification metric because true negatives are not directly meaningful for object detection."
    ],

    [
        "Best Epoch",
        "Should be determined from results.csv using the highest validation mAP50-95. Do not assume epoch 50 is the best epoch."
    ]
]

for row in notes:
    source.append(row)


# ============================================================
# FORMATTING
# ============================================================

for ws in workbook.worksheets:

    ws.freeze_panes = "A2"

    ws.auto_filter.ref = ws.dimensions

    for cell in ws[1]:

        cell.font = Font(
            bold=True
        )

        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True
        )

    for row in ws.iter_rows():

        for cell in row:

            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True
            )

    # Automatic column widths
    for column in range(
        1,
        ws.max_column + 1
    ):

        max_length = 0

        for cell in ws[get_column_letter(column)]:

            if cell.value is not None:

                max_length = max(
                    max_length,
                    len(str(cell.value))
                )

        ws.column_dimensions[
            get_column_letter(column)
        ].width = min(
            max(max_length + 2, 12),
            35
        )


# ============================================================
# NUMBER FORMATTING
# ============================================================

for row in sheet.iter_rows(min_row=2):

    for column in range(12, 21):

        cell = row[column - 1]

        if isinstance(cell.value, float):

            cell.number_format = "0.000"


for row in ranking.iter_rows(min_row=2):

    for column in range(3, 8):

        cell = row[column - 1]

        if isinstance(cell.value, float):

            cell.number_format = "0.000"


# ============================================================
# SAVE FILE
# ============================================================

output_file = "70_20_10_All_Models_Detection_Comparison.xlsx"

workbook.save(output_file)

print()
print("==============================================")
print("Excel file created successfully!")
print("==============================================")
print()
print(f"File: {output_file}")
print()