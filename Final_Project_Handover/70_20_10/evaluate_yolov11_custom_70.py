from pathlib import Path
import numpy as np
from ultralytics import YOLO
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/Users/kalamanjunatha/Downloads/sugarbeet/Experiments/70_20_10"
)

DATASET = Path(
    "/Users/kalamanjunatha/Downloads/sugarbeet/dataset_70_20_10_detection"
)

MODEL_PATH = BASE / \
    "YOLOv11_70_20_10_Detection/weights/best.pt"

TEST_IMAGES = DATASET / "images/test"
TEST_LABELS = DATASET / "labels/test"

OUTPUT_DIR = BASE / "Custom_Evaluation_70_20_10"


# ============================================================
# FIXED EVALUATION SETTINGS
# ============================================================

CONF_THRESHOLD = 0.25
IOU_MATCH_THRESHOLD = 0.50


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)

    intersection = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * \
            max(0, box1[3] - box1[1])

    area2 = max(0, box2[2] - box2[0]) * \
            max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth(label_path, width, height):

    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:

        for line in f:

            values = line.strip().split()

            if len(values) < 5:
                continue

            cls = int(values[0])

            xc = float(values[1])
            yc = float(values[2])
            w = float(values[3])
            h = float(values[4])

            x1 = (xc - w / 2) * width
            y1 = (yc - h / 2) * height
            x2 = (xc + w / 2) * width
            y2 = (yc + h / 2) * height

            boxes.append([
                x1, y1, x2, y2, cls
            ])

    return boxes


# ============================================================
# MODEL PREDICTIONS
# ============================================================

def get_predictions(model, image_path):

    result = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=CONF_THRESHOLD,
        verbose=False
    )[0]

    predictions = []

    if result.boxes is None:
        return predictions

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    for box, score, cls in zip(
        boxes,
        scores,
        classes
    ):

        predictions.append([
            float(box[0]),
            float(box[1]),
            float(box[2]),
            float(box[3]),
            float(score),
            int(cls)
        ])

    return predictions


# ============================================================
# PRECISION / RECALL / F1
# ============================================================

def evaluate_predictions(
    predictions,
    ground_truths,
    iou_threshold=0.50
):

    TP = 0
    FP = 0
    FN = 0

    for preds, gts in zip(
        predictions,
        ground_truths
    ):

        matched_gt = set()

        preds = sorted(
            preds,
            key=lambda x: x[4],
            reverse=True
        )

        for pred in preds:

            best_iou = 0.0
            best_gt = -1

            for gt_index, gt in enumerate(gts):

                if gt_index in matched_gt:
                    continue

                if int(pred[5]) != int(gt[4]):
                    continue

                iou = calculate_iou(pred, gt)

                if iou > best_iou:

                    best_iou = iou
                    best_gt = gt_index

            if (
                best_iou >= iou_threshold
                and best_gt >= 0
            ):

                TP += 1
                matched_gt.add(best_gt)

            else:

                FP += 1

        FN += len(gts) - len(matched_gt)

    precision = (
        TP / (TP + FP)
        if TP + FP > 0 else 0
    )

    recall = (
        TP / (TP + FN)
        if TP + FN > 0 else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0 else 0
    )

    return precision, recall, f1, TP, FP, FN


# ============================================================
# AP
# ============================================================

def calculate_ap(
    predictions,
    ground_truths,
    iou_threshold
):

    prediction_list = []

    total_gt = 0

    for image_index, (preds, gts) in enumerate(
        zip(predictions, ground_truths)
    ):

        total_gt += len(gts)

        for pred in preds:

            prediction_list.append(
                (
                    pred[4],
                    image_index,
                    pred
                )
            )

    if total_gt == 0:
        return 0.0

    prediction_list.sort(
        key=lambda x: x[0],
        reverse=True
    )

    matched = {
        i: set()
        for i in range(len(ground_truths))
    }

    tp = np.zeros(len(prediction_list))
    fp = np.zeros(len(prediction_list))

    for index, (_, image_index, pred) in enumerate(
        prediction_list
    ):

        gts = ground_truths[image_index]

        best_iou = 0.0
        best_gt = -1

        for gt_index, gt in enumerate(gts):

            if gt_index in matched[image_index]:
                continue

            if int(pred[5]) != int(gt[4]):
                continue

            iou = calculate_iou(pred, gt)

            if iou > best_iou:

                best_iou = iou
                best_gt = gt_index

        if (
            best_iou >= iou_threshold
            and best_gt >= 0
        ):

            tp[index] = 1
            matched[image_index].add(best_gt)

        else:

            fp[index] = 1

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    recalls = cumulative_tp / total_gt

    precisions = cumulative_tp / (
        cumulative_tp +
        cumulative_fp +
        1e-9
    )

    # Precision envelope
    for i in range(
        len(precisions) - 2,
        -1,
        -1
    ):

        precisions[i] = max(
            precisions[i],
            precisions[i + 1]
        )

    recall_points = np.concatenate(
        ([0.0], recalls, [1.0])
    )

    precision_points = np.concatenate(
        ([1.0], precisions, [0.0])
    )

    ap = 0.0

    for i in range(
        len(recall_points) - 1
    ):

        ap += (
            recall_points[i + 1] -
            recall_points[i]
        ) * precision_points[i + 1]

    return float(ap)


# ============================================================
# mAP@50 AND mAP@50-95
# ============================================================

def calculate_map(
    predictions,
    ground_truths
):

    map50 = calculate_ap(
        predictions,
        ground_truths,
        0.50
    )

    thresholds = np.arange(
        0.50,
        0.96,
        0.05
    )

    aps = []

    for threshold in thresholds:

        aps.append(
            calculate_ap(
                predictions,
                ground_truths,
                threshold
            )
        )

    map5095 = float(np.mean(aps))

    return map50, map5095


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("YOLOv11 CUSTOM STANDARDIZED EVALUATION")
print("70:20:10 TEST SET")
print("=" * 70)

print()
print("Model:")
print(MODEL_PATH)

print()
print("Dataset:")
print(DATASET)

print()
print("Evaluation settings:")
print(f"Confidence threshold : {CONF_THRESHOLD}")
print(f"IoU matching threshold: {IOU_MATCH_THRESHOLD}")
print("Image size            : 640")


# ============================================================
# VERIFY PATHS
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not TEST_IMAGES.exists():

    raise FileNotFoundError(
        f"Test images not found:\n{TEST_IMAGES}"
    )

if not TEST_LABELS.exists():

    raise FileNotFoundError(
        f"Test labels not found:\n{TEST_LABELS}"
    )

print()
print("All paths verified.")


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading YOLOv11...")

model = YOLO(
    str(MODEL_PATH)
)

print("Model loaded successfully.")


# ============================================================
# TEST IMAGES
# ============================================================

image_files = sorted(
    list(TEST_IMAGES.glob("*.jpg")) +
    list(TEST_IMAGES.glob("*.jpeg")) +
    list(TEST_IMAGES.glob("*.png"))
)

print()
print(f"Test images: {len(image_files)}")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

all_predictions = []
all_ground_truths = []

print()
print("=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

for index, image_path in enumerate(
    image_files,
    start=1
):

    print(
        f"Processing {index}/{len(image_files)}",
        end="\r"
    )

    predictions = get_predictions(
        model,
        image_path
    )

    with Image.open(image_path) as img:

        width, height = img.size

    label_path = TEST_LABELS / (
        image_path.stem + ".txt"
    )

    ground_truth = load_ground_truth(
        label_path,
        width,
        height
    )

    all_predictions.append(predictions)
    all_ground_truths.append(ground_truth)

print()
print()
print("Predictions generated.")


# ============================================================
# EVALUATE
# ============================================================

precision, recall, f1, TP, FP, FN = \
    evaluate_predictions(
        all_predictions,
        all_ground_truths,
        IOU_MATCH_THRESHOLD
    )

map50, map5095 = calculate_map(
    all_predictions,
    all_ground_truths
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL YOLOv11 CUSTOM TEST RESULTS")
print("=" * 70)

print()
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"mAP50     : {map50:.4f}")
print(f"mAP50-95  : {map5095:.4f}")

print()
print(f"TP        : {TP}")
print(f"FP        : {FP}")
print(f"FN        : {FN}")


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = OUTPUT_DIR / \
    "yolov11_custom_test_results.txt"

with open(OUTPUT_FILE, "w") as f:

    f.write(
        "YOLOv11 CUSTOM STANDARDIZED EVALUATION\n"
    )

    f.write(
        "70:20:10 TEST SET\n\n"
    )

    f.write(
        f"Confidence threshold: "
        f"{CONF_THRESHOLD}\n"
    )

    f.write(
        f"IoU matching threshold: "
        f"{IOU_MATCH_THRESHOLD}\n"
    )

    f.write(
        "Image size: 640\n\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall: {recall:.4f}\n"
    )

    f.write(
        f"F1: {f1:.4f}\n"
    )

    f.write(
        f"mAP50: {map50:.4f}\n"
    )

    f.write(
        f"mAP50-95: {map5095:.4f}\n\n"
    )

    f.write(
        f"TP: {TP}\n"
    )

    f.write(
        f"FP: {FP}\n"
    )

    f.write(
        f"FN: {FN}\n"
    )

print()
print("=" * 70)
print("CUSTOM EVALUATION COMPLETED")
print("=" * 70)

print()
print("Results saved to:")
print(OUTPUT_FILE)