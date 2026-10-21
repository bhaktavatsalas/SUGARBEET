#!/usr/bin/env python3
"""
LIGHTWEIGHT MGKD FOR SUGERBEET DETECTION
========================================

Teacher:
    YOLOv11 or YOLOv12

Student:
    RT-DETR

Verified feature mapping:
    Low  : YOLO 16 -> RT-DETR 21   (80x80)
    Mid  : YOLO 19 -> RT-DETR 24   (40x40)
    High : YOLO 22 -> RT-DETR 27   (20x20)
    Pred : YOLO Detect -> RT-DETR decoder

Main design goals:
    - One teacher forward pass per cached batch.
    - Teacher knowledge cached to CPU/disk once.
    - Teacher released before student training.
    - Small MPS batch size.
    - Only RT-DETR late layers 21..28 are trainable.
    - Native RT-DETR detection loss is retained.
    - Low/mid/high spatial-attention KD + prediction KD.
    - No new detector head is created.
    - No 20x20 custom grid detector is created.
    - Final paper metrics must be produced with the project's
      unified standardized evaluator.

FIRST TEST:
    python Experiments/Hybrid_Approaches/MGKD/train_mgkd.py \
        --experiment y11_rt_70 \
        --epochs 3 \
        --batch-size 1

FULL RUN:
    python Experiments/Hybrid_Approaches/MGKD/train_mgkd.py \
        --experiment y11_rt_70 \
        --epochs 50 \
        --batch-size 1
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms

from ultralytics import YOLO, RTDETR
from ultralytics.utils.nms import non_max_suppression


# ============================================================
# GLOBAL CONFIG
# ============================================================

BASE = Path(
    "/Users/kalamanjunatha/Downloads/sugarbeet"
)

DATASET_70 = (
    BASE / "dataset_70_20_10_detection"
)

DATASET_80 = (
    BASE / "dataset_80_10_10_detection"
)

MODELS = {
    "y11_70": (
        BASE /
        "Experiments/70_20_10/"
        "YOLOv11_70_20_10_Detection/weights/best.pt"
    ),
    "y12_70": (
        BASE /
        "Experiments/70_20_10/"
        "YOLOv12_70_20_10_Detection/weights/best.pt"
    ),
    "rt_70": (
        BASE /
        "Experiments/70_20_10/"
        "RT_DETR_70_20_10/weights/best.pt"
    ),
    "y11_80": (
        BASE /
        "Experiments/80_10_10/"
        "YOLOv11/detection/weights/best.pt"
    ),
    "y12_80": (
        BASE /
        "Experiments/80_10_10/"
        "YOLOv12/detection-2/weights/best.pt"
    ),
    "rt_80": (
        BASE /
        "Experiments/80_10_10/"
        "RT_DETR4/weights/best.pt"
    ),
}

OUTPUT_ROOT = (
    BASE /
    "Experiments/Hybrid_Approaches/MGKD/results"
)

EXPERIMENTS = {
    "y11_rt_70": {
        "split": "70_20_10",
        "dataset": DATASET_70,
        "teacher": MODELS["y11_70"],
        "student": MODELS["rt_70"],
        "teacher_name": "YOLOv11",
        "name": "MGKD_YOLOv11_RTDETR_70_20_10",
    },
    "y12_rt_70": {
        "split": "70_20_10",
        "dataset": DATASET_70,
        "teacher": MODELS["y12_70"],
        "student": MODELS["rt_70"],
        "teacher_name": "YOLOv12",
        "name": "MGKD_YOLOv12_RTDETR_70_20_10",
    },
    "y11_rt_80": {
        "split": "80_10_10",
        "dataset": DATASET_80,
        "teacher": MODELS["y11_80"],
        "student": MODELS["rt_80"],
        "teacher_name": "YOLOv11",
        "name": "MGKD_YOLOv11_RTDETR_80_10_10",
    },
    "y12_rt_80": {
        "split": "80_10_10",
        "dataset": DATASET_80,
        "teacher": MODELS["y12_80"],
        "student": MODELS["rt_80"],
        "teacher_name": "YOLOv12",
        "name": "MGKD_YOLOv12_RTDETR_80_10_10",
    },
}

TEACHER_LAYERS = {
    "low": 16,
    "mid": 19,
    "high": 22,
}

STUDENT_LAYERS = {
    "low": 21,
    "mid": 24,
    "high": 27,
}

STUDENT_DECODER_LAYER = 28

# Manager's lambda terms.
LAMBDA_LOW = 0.05
LAMBDA_MID = 0.05
LAMBDA_HIGH = 0.10
LAMBDA_PRED = 0.10

# Training.
IMG_SIZE = 640
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0

# Teacher prediction KD.
TEACHER_CONF = 0.25
TEACHER_MAX_DET = 50

# Final standardized evaluation settings.
EVAL_CONF = 0.25
EVAL_IOU = 0.50

# Memory safety.
CACHE_BATCH_SIZE = 1
EMPTY_CACHE_EVERY = 5

# Freeze early student backbone/neck.
# Layers 0..20 frozen; 21..28 trainable.
TRAIN_FROM_LAYER = 21


# ============================================================
# DEVICE / MEMORY
# ============================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_memory(device: torch.device) -> None:
    gc.collect()

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# DATASET
# ============================================================

class SugarBeetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        img_size: int = IMG_SIZE,
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = int(img_size)

        self.image_dir = (
            self.root / "images" / split
        )
        self.label_dir = (
            self.root / "labels" / split
        )

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found:\n"
                f"{self.image_dir}"
            )

        if not self.label_dir.exists():
            raise FileNotFoundError(
                f"Label directory not found:\n"
                f"{self.label_dir}"
            )

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }

        self.images = sorted(
            [
                p
                for p in self.image_dir.iterdir()
                if p.suffix.lower() in extensions
            ]
        )

        if not self.images:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

        missing = [
            p.name
            for p in self.images
            if not (
                self.label_dir /
                f"{p.stem}.txt"
            ).exists()
        ]

        if missing:
            raise RuntimeError(
                "Missing label files. Examples: "
                f"{missing[:10]}"
            )

        print(
            f"{split}: {len(self.images)} images"
        )

    def __len__(self):
        return len(self.images)

    def read_labels(
        self,
        image_path: Path,
    ) -> torch.Tensor:
        label_path = (
            self.label_dir /
            f"{image_path.stem}.txt"
        )

        rows = []

        with open(
            label_path,
            "r"
        ) as f:

            for line in f:
                parts = (
                    line.strip()
                    .split()
                )

                if len(parts) < 5:
                    continue

                try:
                    cls = int(
                        float(parts[0])
                    )

                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                except ValueError:
                    continue

                # SugarBeet project is single-class.
                if cls != 0:
                    continue

                rows.append(
                    [
                        0.0,
                        min(max(x, 0.0), 1.0),
                        min(max(y, 0.0), 1.0),
                        min(max(w, 1e-6), 1.0),
                        min(max(h, 1e-6), 1.0),
                    ]
                )

        if not rows:
            return torch.empty(
                (0, 5),
                dtype=torch.float32
            )

        return torch.tensor(
            rows,
            dtype=torch.float32
        ).reshape(-1, 5)

    def letterbox(
        self,
        image: Image.Image,
        labels: torch.Tensor,
    ):
        original_w, original_h = (
            image.size
        )

        scale = min(
            self.img_size / original_w,
            self.img_size / original_h,
        )

        new_w = max(
            1,
            int(round(original_w * scale))
        )

        new_h = max(
            1,
            int(round(original_h * scale))
        )

        resized = image.resize(
            (new_w, new_h),
            Image.Resampling.BILINEAR
        )

        canvas = Image.new(
            "RGB",
            (self.img_size, self.img_size),
            (114, 114, 114),
        )

        pad_x = (
            self.img_size - new_w
        ) // 2

        pad_y = (
            self.img_size - new_h
        ) // 2

        canvas.paste(
            resized,
            (pad_x, pad_y)
        )

        array = np.asarray(
            canvas,
            dtype=np.float32
        ) / 255.0

        image_tensor = (
            torch.from_numpy(array)
            .permute(2, 0, 1)
            .contiguous()
        )

        transformed = []

        for row in labels.reshape(-1, 5):

            cls = float(
                row[0]
            )

            cx = (
                float(row[1])
                * original_w
            )

            cy = (
                float(row[2])
                * original_h
            )

            bw = (
                float(row[3])
                * original_w
            )

            bh = (
                float(row[4])
                * original_h
            )

            cx = (
                cx * scale
                +
                pad_x
            ) / self.img_size

            cy = (
                cy * scale
                +
                pad_y
            ) / self.img_size

            bw = (
                bw * scale
            ) / self.img_size

            bh = (
                bh * scale
            ) / self.img_size

            transformed.append(
                [
                    cls,
                    min(max(cx, 0.0), 1.0),
                    min(max(cy, 0.0), 1.0),
                    min(max(bw, 1e-6), 1.0),
                    min(max(bh, 1e-6), 1.0),
                ]
            )

        if transformed:
            transformed_labels = (
                torch.tensor(
                    transformed,
                    dtype=torch.float32
                ).reshape(-1, 5)
            )
        else:
            transformed_labels = (
                torch.empty(
                    (0, 5),
                    dtype=torch.float32
                )
            )

        return (
            image_tensor,
            transformed_labels
        )

    def __getitem__(
        self,
        index
    ):
        image_path = (
            self.images[index]
        )

        image = (
            Image.open(
                image_path
            )
            .convert("RGB")
        )

        labels = self.read_labels(
            image_path
        )

        image_tensor, labels = (
            self.letterbox(
                image,
                labels
            )
        )

        return (
            image_tensor,
            labels,
            image_path.name
        )


def collate_raw(batch):
    return (
        torch.stack(
            [x[0] for x in batch]
        ),
        [x[1] for x in batch],
        [x[2] for x in batch],
    )


# ============================================================
# FEATURE HOOKS
# ============================================================

class HookStore:
    def __init__(
        self,
        model,
        layer_indices,
    ):
        self.outputs = {}
        self.handles = []

        for index in layer_indices:

            handle = (
                model.model[index]
                .register_forward_hook(
                    self._make_hook(index)
                )
            )

            self.handles.append(
                handle
            )

    def _make_hook(
        self,
        index,
    ):
        def hook(
            module,
            inputs,
            output
        ):
            if isinstance(
                output,
                (tuple, list)
            ):
                output = output[0]

            if not torch.is_tensor(
                output
            ):
                raise RuntimeError(
                    f"Layer {index} returned "
                    "a non-tensor output."
                )

            self.outputs[index] = output

        return hook

    def clear(self):
        self.outputs.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()

        self.handles.clear()


class DecoderHook:
    def __init__(
        self,
        model,
        layer_index,
    ):
        self.output = None

        self.handle = (
            model.model[layer_index]
            .register_forward_hook(
                self._hook
            )
        )

    def _hook(
        self,
        module,
        inputs,
        output,
    ):
        self.output = output

    def clear(self):
        self.output = None

    def close(self):
        self.handle.remove()


# ============================================================
# SPATIAL ATTENTION
# ============================================================

def spatial_attention(
    feature: torch.Tensor
) -> torch.Tensor:
    """
    Convert [B,C,H,W] into a normalized spatial attention map
    [B,1,H,W]. This allows teacher/student feature maps with
    different channel counts to be compared.
    """

    if feature.ndim != 4:
        raise RuntimeError(
            "Expected [B,C,H,W], got "
            f"{tuple(feature.shape)}"
        )

    attention = (
        feature.float()
        .pow(2)
        .mean(
            dim=1,
            keepdim=True
        )
    )

    flat = attention.flatten(1)

    norm = (
        torch.linalg.vector_norm(
            flat,
            ord=2,
            dim=1,
            keepdim=True
        )
        + 1e-6
    )

    return (
        flat / norm
    ).reshape(
        attention.shape
    )


def feature_kd_loss(
    student_feature,
    teacher_attention,
):
    student_attention = spatial_attention(
        student_feature
    )

    teacher_attention = teacher_attention.to(
        device=student_feature.device,
        dtype=student_feature.dtype
    )

    if (
        student_attention.shape[-2:]
        !=
        teacher_attention.shape[-2:]
    ):

        teacher_attention = F.interpolate(
            teacher_attention,
            size=student_attention.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    return F.mse_loss(
        student_attention,
        teacher_attention
    )


# ============================================================
# BOX OPERATIONS
# ============================================================

def cxcywh_to_xyxy(
    boxes
):
    cx = boxes[..., 0]
    cy = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]

    return torch.stack(
        [
            cx - w / 2.0,
            cy - h / 2.0,
            cx + w / 2.0,
            cy + h / 2.0,
        ],
        dim=-1,
    )


def iou_matrix(
    boxes1,
    boxes2,
):
    if (
        len(boxes1) == 0
        or
        len(boxes2) == 0
    ):
        return torch.zeros(
            (
                len(boxes1),
                len(boxes2)
            ),
            device=boxes1.device,
            dtype=boxes1.dtype
        )

    a = boxes1[:, None, :]
    b = boxes2[None, :, :]

    ix1 = torch.maximum(
        a[..., 0],
        b[..., 0]
    )

    iy1 = torch.maximum(
        a[..., 1],
        b[..., 1]
    )

    ix2 = torch.minimum(
        a[..., 2],
        b[..., 2]
    )

    iy2 = torch.minimum(
        a[..., 3],
        b[..., 3]
    )

    inter = (
        (ix2 - ix1).clamp(min=0)
        *
        (iy2 - iy1).clamp(min=0)
    )

    area1 = (
        (a[..., 2] - a[..., 0]).clamp(min=0)
        *
        (a[..., 3] - a[..., 1]).clamp(min=0)
    )

    area2 = (
        (b[..., 2] - b[..., 0]).clamp(min=0)
        *
        (b[..., 3] - b[..., 1]).clamp(min=0)
    )

    union = (
        area1
        +
        area2
        -
        inter
        +
        1e-7
    )

    return inter / union


# ============================================================
# TEACHER PREDICTIONS
# ============================================================

def teacher_predictions_from_raw(
    raw_output,
):
    """
    Convert YOLO inference output into a list of normalized
    xywh boxes and confidence scores.

    Since the training input is already a 640x640 letterboxed
    tensor, pixel coordinates are normalized by IMG_SIZE.
    """

    if isinstance(
        raw_output,
        tuple
    ):
        predictions = raw_output[0]
    else:
        predictions = raw_output

    if isinstance(
        predictions,
        (tuple, list)
    ):
        predictions = predictions[0]

    if not torch.is_tensor(
        predictions
    ):
        raise RuntimeError(
            "YOLO output is not a tensor."
        )

    if (
        predictions.ndim == 3
        and
        predictions.shape[1] <= 10
        and
        predictions.shape[2] > predictions.shape[1]
    ):
        predictions = (
            predictions
            .permute(
                0,
                2,
                1
            )
            .contiguous()
        )

    nms_outputs = non_max_suppression(
        predictions,
        conf_thres=TEACHER_CONF,
        iou_thres=0.70,
        max_det=TEACHER_MAX_DET,
        classes=[0],
    )

    output = []

    for boxes in nms_outputs:

        if (
            boxes is None
            or
            len(boxes) == 0
        ):

            output.append(
                {
                    "boxes": torch.empty(
                        (0, 4),
                        dtype=torch.float32
                    ),
                    "scores": torch.empty(
                        (0,),
                        dtype=torch.float32
                    ),
                }
            )

            continue

        xyxy = (
            boxes[:, :4]
            /
            float(IMG_SIZE)
        ).clamp(
            0.0,
            1.0
        )

        scores = (
            boxes[:, 4]
            .clamp(
                0.0,
                1.0
            )
        )

        x1 = xyxy[:, 0]
        y1 = xyxy[:, 1]
        x2 = xyxy[:, 2]
        y2 = xyxy[:, 3]

        cx = (
            x1 + x2
        ) / 2.0

        cy = (
            y1 + y2
        ) / 2.0

        w = (
            x2 - x1
        ).clamp(
            min=1e-6
        )

        h = (
            y2 - y1
        ).clamp(
            min=1e-6
        )

        xywh = torch.stack(
            [
                cx,
                cy,
                w,
                h
            ],
            dim=1
        )

        output.append(
            {
                "boxes": xywh.detach().cpu(),
                "scores": scores.detach().cpu(),
            }
        )

    return output


# ============================================================
# TEACHER CACHE
# ============================================================

def cache_teacher_knowledge(
    teacher_path: Path,
    dataset: SugarBeetDataset,
    cache_path: Path,
    device: torch.device,
):
    """
    Create a compact teacher cache.

    Stored per image:
        low spatial attention
        mid spatial attention
        high spatial attention
        teacher pseudo boxes
        teacher confidence scores

    The teacher is executed only once.
    """

    if cache_path.exists():

        print(
            "\nLoading teacher cache:"
        )

        print(
            cache_path
        )

        cache = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True
        )

        names = [
            item["name"]
            for item in cache["items"]
        ]

        current_names = [
            p.name
            for p in dataset.images
        ]

        if (
            names
            !=
            current_names
        ):
            raise RuntimeError(
                "Existing teacher cache does not match "
                "current dataset ordering."
            )

        print(
            f"Teacher cache contains "
            f"{len(names)} images."
        )

        return cache

    print("\n" + "=" * 70)
    print("PHASE 1: ONE-TIME TEACHER CACHE")
    print("=" * 70)
    print(
        f"Teacher: {teacher_path}"
    )
    print(
        f"Device : {device}"
    )
    print(
        f"Images : {len(dataset)}"
    )
    print(
        f"Batch  : {CACHE_BATCH_SIZE}"
    )

    teacher = YOLO(
        str(teacher_path)
    )

    teacher.model.to(
        device
    )

    teacher.model.eval()

    for parameter in (
        teacher.model.parameters()
    ):
        parameter.requires_grad = False

    hooks = HookStore(
        teacher.model,
        list(
            TEACHER_LAYERS.values()
        )
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CACHE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_raw
    )

    items = []

    start = time.time()

    try:

        with torch.no_grad():

            for batch_index, (
                images,
                targets,
                names
            ) in enumerate(loader):

                images = images.to(
                    device
                )

                hooks.clear()

                raw_output = teacher.model(
                    images
                )

                required = (
                    list(
                        TEACHER_LAYERS.values()
                    )
                )

                if not all(
                    idx in hooks.outputs
                    for idx in required
                ):
                    raise RuntimeError(
                        "One or more teacher feature "
                        "layers were not captured."
                    )

                predictions = (
                    teacher_predictions_from_raw(
                        raw_output
                    )
                )

                low_att = (
                    spatial_attention(
                        hooks.outputs[
                            TEACHER_LAYERS["low"]
                        ]
                    )
                    .detach()
                    .cpu()
                )

                mid_att = (
                    spatial_attention(
                        hooks.outputs[
                            TEACHER_LAYERS["mid"]
                        ]
                    )
                    .detach()
                    .cpu()
                )

                high_att = (
                    spatial_attention(
                        hooks.outputs[
                            TEACHER_LAYERS["high"]
                        ]
                    )
                    .detach()
                    .cpu()
                )

                for i, name in enumerate(
                    names
                ):

                    items.append(
                        {
                            "name": name,
                            "low": low_att[i],
                            "mid": mid_att[i],
                            "high": high_att[i],
                            "boxes": predictions[i][
                                "boxes"
                            ],
                            "scores": predictions[i][
                                "scores"
                            ],
                        }
                    )

                del (
                    images,
                    targets,
                    raw_output,
                    predictions,
                    low_att,
                    mid_att,
                    high_att,
                )

                if (
                    batch_index + 1
                ) % EMPTY_CACHE_EVERY == 0:
                    clear_memory(device)

                if (
                    batch_index == 0
                    or
                    (
                        batch_index + 1
                    ) % 25 == 0
                ):

                    elapsed = (
                        time.time()
                        -
                        start
                    )

                    print(
                        f"Cached "
                        f"{len(items)}/{len(dataset)}"
                        f" | "
                        f"{elapsed/60:.2f} min"
                    )

    finally:

        hooks.close()

        del hooks
        del teacher

        clear_memory(
            device
        )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "version": "mgkd_light_v1",
            "teacher_layers": TEACHER_LAYERS,
            "items": items,
        },
        cache_path
    )

    print(
        "\nTeacher cache saved:"
    )
    print(
        cache_path
    )

    return {
        "version": "mgkd_light_v1",
        "teacher_layers": TEACHER_LAYERS,
        "items": items,
    }


# ============================================================
# CACHE DATASET
# ============================================================

class TeacherCacheDataset(
    torch.utils.data.Dataset
):
    def __init__(
        self,
        base_dataset,
        teacher_items
    ):
        if (
            len(base_dataset)
            !=
            len(teacher_items)
        ):
            raise RuntimeError(
                "Teacher cache and dataset lengths differ."
            )

        self.base_dataset = base_dataset
        self.teacher_items = teacher_items

    def __len__(self):
        return len(
            self.base_dataset
        )

    def __getitem__(
        self,
        index
    ):
        image, labels, name = (
            self.base_dataset[index]
        )

        teacher_item = (
            self.teacher_items[index]
        )

        if teacher_item["name"] != name:
            raise RuntimeError(
                "Dataset/cache ordering mismatch: "
                f"{teacher_item['name']} != {name}"
            )

        return (
            image,
            labels,
            teacher_item,
            name
        )


def collate_teacher_cache(
    batch
):
    return (
        torch.stack(
            [x[0] for x in batch]
        ),
        [x[1] for x in batch],
        [x[2] for x in batch],
        [x[3] for x in batch],
    )


# ============================================================
# STUDENT BATCH
# ============================================================

def make_student_batch(
    images,
    targets,
    device
):
    cls_list = []
    bbox_list = []
    batch_idx_list = []

    for batch_index, target in enumerate(
        targets
    ):

        if target is None:
            continue

        if target.numel() == 0:
            continue

        target = target.reshape(
            -1,
            5
        )

        cls_list.append(
            target[:, :1]
        )

        bbox_list.append(
            target[:, 1:5]
        )

        batch_idx_list.append(
            torch.full(
                (
                    target.shape[0],
                ),
                batch_index,
                dtype=torch.long
            )
        )

    if cls_list:

        cls = torch.cat(
            cls_list,
            dim=0
        )

        bboxes = torch.cat(
            bbox_list,
            dim=0
        )

        batch_idx = torch.cat(
            batch_idx_list,
            dim=0
        )

    else:

        cls = torch.empty(
            (0, 1),
            dtype=torch.float32
        )

        bboxes = torch.empty(
            (0, 4),
            dtype=torch.float32
        )

        batch_idx = torch.empty(
            (0,),
            dtype=torch.long
        )

    return {
        "img": images.to(device),
        "cls": cls.to(device),
        "bboxes": bboxes.to(device),
        "batch_idx": batch_idx.to(device),
    }


# ============================================================
# STUDENT DECODER
# ============================================================

def parse_student_decoder(
    decoder_output
):
    if not isinstance(
        decoder_output,
        (tuple, list)
    ):
        raise RuntimeError(
            "RT-DETR decoder output is not tuple/list."
        )

    if len(decoder_output) < 2:
        raise RuntimeError(
            "RT-DETR decoder output is too short."
        )

    boxes = decoder_output[0]
    scores = decoder_output[1]

    if not torch.is_tensor(
        boxes
    ):
        raise RuntimeError(
            "RT-DETR decoder boxes are not a tensor."
        )

    if not torch.is_tensor(
        scores
    ):
        raise RuntimeError(
            "RT-DETR decoder scores are not a tensor."
        )

    if boxes.ndim == 4:
        boxes = boxes[-1]

    if scores.ndim == 4:
        scores = scores[-1]

    if boxes.ndim != 3:
        raise RuntimeError(
            f"Unexpected decoder box shape: "
            f"{tuple(boxes.shape)}"
        )

    if scores.ndim != 3:
        raise RuntimeError(
            f"Unexpected decoder score shape: "
            f"{tuple(scores.shape)}"
        )

    # One class = first score channel.
    scores = scores[..., 0]

    # Normally RT-DETR boxes are normalized xywh.
    # Apply sigmoid only if they are clearly raw logits.
    min_value = (
        boxes.detach().min().item()
    )

    max_value = (
        boxes.detach().max().item()
    )

    if (
        min_value < -1e-3
        or
        max_value > 1.01
    ):
        boxes = boxes.sigmoid()

    boxes = boxes.clamp(
        0.0,
        1.0
    )

    return (
        boxes,
        scores
    )


# ============================================================
# PREDICTION KD
# ============================================================

def prediction_kd_loss(
    student_boxes,
    student_scores,
    teacher_boxes,
    teacher_scores
):
    if (
        len(teacher_boxes) == 0
        or
        len(student_boxes) == 0
    ):
        return (
            student_boxes.sum()
            * 0.0
        )

    teacher_boxes = (
        teacher_boxes.to(
            device=student_boxes.device,
            dtype=student_boxes.dtype
        )
    )

    teacher_scores = (
        teacher_scores.to(
            device=student_scores.device,
            dtype=student_scores.dtype
        )
    )

    # Limit matching workload.
    if len(teacher_boxes) > TEACHER_MAX_DET:
        teacher_boxes = (
            teacher_boxes[
                :TEACHER_MAX_DET
            ]
        )

        teacher_scores = (
            teacher_scores[
                :TEACHER_MAX_DET
            ]
        )

    # Assignment cost is detached.
    l1 = torch.cdist(
        student_boxes,
        teacher_boxes,
        p=1
    )

    iou = iou_matrix(
        cxcywh_to_xyxy(
            student_boxes
        ),
        cxcywh_to_xyxy(
            teacher_boxes
        )
    )

    score_prob = (
        student_scores.sigmoid()
        if (
            student_scores.detach().min().item() < 0.0
            or
            student_scores.detach().max().item() > 1.0
        )
        else
        student_scores
    )

    score_prob = score_prob.clamp(
        1e-6,
        1.0 - 1e-6
    )

    score_cost = (
        -score_prob.detach().log()[:, None]
        *
        teacher_scores.detach()[None, :]
    )

    cost = (
        l1.detach()
        +
        (1.0 - iou.detach())
        +
        score_cost
    )

    rows, cols = (
        linear_sum_assignment(
            cost.cpu().numpy()
        )
    )

    if len(rows) == 0:
        return (
            student_boxes.sum()
            * 0.0
        )

    rows_t = torch.tensor(
        rows,
        dtype=torch.long,
        device=student_boxes.device
    )

    cols_t = torch.tensor(
        cols,
        dtype=torch.long,
        device=student_boxes.device
    )

    s_boxes = (
        student_boxes[
            rows_t
        ]
    )

    t_boxes = (
        teacher_boxes[
            cols_t
        ]
    )

    s_scores = (
        student_scores[
            rows_t
        ]
    )

    t_scores = (
        teacher_scores[
            cols_t
        ]
    )

    box_loss = F.smooth_l1_loss(
        s_boxes,
        t_boxes,
        reduction="mean"
    )

    s_prob = (
        s_scores.sigmoid()
        if (
            s_scores.detach().min().item() < 0.0
            or
            s_scores.detach().max().item() > 1.0
        )
        else
        s_scores
    )

    score_loss = F.mse_loss(
        s_prob,
        t_scores
    )

    return (
        box_loss
        +
        score_loss
    )


# ============================================================
# TRAINING
# ============================================================

def train_experiment(
    experiment_name: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    lr: float,
):
    seed_everything()

    if experiment_name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment: {experiment_name}"
        )

    # Keep global image size synchronized with CLI.
    global IMG_SIZE
    IMG_SIZE = int(img_size)

    exp = EXPERIMENTS[
        experiment_name
    ]

    device = get_device()

    output_dir = (
        OUTPUT_ROOT /
        exp["name"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    teacher_cache_dir = (
        output_dir /
        "teacher_cache"
    )

    teacher_cache_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions_dir = (
        output_dir /
        "predictions"
    )

    predictions_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 80)
    print("LIGHTWEIGHT MGKD TRAINING")
    print("=" * 80)
    print(
        f"Experiment        : {experiment_name}"
    )
    print(
        f"Teacher           : {exp['teacher_name']}"
    )
    print(
        "Student           : RT-DETR"
    )
    print(
        f"Split             : {exp['split']}"
    )
    print(
        f"Device            : {device}"
    )
    print(
        f"Epochs            : {epochs}"
    )
    print(
        f"Batch size        : {batch_size}"
    )
    print(
        f"Image size        : {IMG_SIZE}"
    )
    print(
        f"Learning rate     : {lr}"
    )
    print(
        f"Train from layer  : {TRAIN_FROM_LAYER}"
    )
    print(
        "Frozen layers     : 0..20"
    )
    print(
        "Trainable layers : 21..28"
    )
    print(
        f"Lambdas           : "
        f"{LAMBDA_LOW}, "
        f"{LAMBDA_MID}, "
        f"{LAMBDA_HIGH}, "
        f"{LAMBDA_PRED}"
    )
    print("=" * 80)

    teacher_path = Path(
        exp["teacher"]
    )

    student_path = Path(
        exp["student"]
    )

    if not teacher_path.exists():
        raise FileNotFoundError(
            f"Teacher not found:\n{teacher_path}"
        )

    if not student_path.exists():
        raise FileNotFoundError(
            f"Student not found:\n{student_path}"
        )

    dataset_root = Path(
        exp["dataset"]
    )

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{dataset_root}"
        )

    train_dataset = SugarBeetDataset(
        dataset_root,
        "train",
        IMG_SIZE
    )

    # --------------------------------------------------------
    # Cache teacher ONCE.
    # --------------------------------------------------------

    teacher_cache_path = (
        teacher_cache_dir /
        "knowledge_v2.pt"
    )

    cache = cache_teacher_knowledge(
        teacher_path,
        train_dataset,
        teacher_cache_path,
        device,
    )

    # --------------------------------------------------------
    # Load student.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 2: STUDENT")
    print("=" * 70)

    student = RTDETR(
        str(student_path)
    )

    student.model.to(
        device
    )
    student.model.criterion = student.model.init_criterion()

    # Freeze everything first.
    for parameter in (
        student.model.parameters()
    ):
        parameter.requires_grad = False

    # Unfreeze only layers 21..28.
    for index in range(
        TRAIN_FROM_LAYER,
        len(student.model.model)
    ):
        for parameter in (
            student.model
            .model[index]
            .parameters()
        ):
            parameter.requires_grad = True

    trainable_parameters = [
        parameter
        for parameter
        in student.model.parameters()
        if parameter.requires_grad
    ]

    total_parameters = sum(
        parameter.numel()
        for parameter
        in student.model.parameters()
    )

    trainable_count = sum(
        parameter.numel()
        for parameter
        in trainable_parameters
    )

    if trainable_count == 0:
        raise RuntimeError(
            "Student has zero trainable parameters."
        )

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_count:,}"
    )

    # --------------------------------------------------------
    # Hooks.
    # --------------------------------------------------------

    student_features = HookStore(
        student.model,
        list(
            STUDENT_LAYERS.values()
        )
    )

    student_decoder = DecoderHook(
        student.model,
        STUDENT_DECODER_LAYER
    )

    # --------------------------------------------------------
    # Dataset loader.
    # --------------------------------------------------------

    kd_dataset = TeacherCacheDataset(
        train_dataset,
        cache["items"]
    )

    loader = torch.utils.data.DataLoader(
        kd_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_teacher_cache
    )

    # --------------------------------------------------------
    # Optimizer.
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.1
    )

    # --------------------------------------------------------
    # Save configuration.
    # --------------------------------------------------------

    config = {
        "experiment": experiment_name,
        "teacher_name": exp["teacher_name"],
        "teacher_model": str(teacher_path),
        "student_model": str(student_path),
        "dataset": str(dataset_root),
        "split": exp["split"],
        "device": str(device),
        "image_size": IMG_SIZE,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "weight_decay": WEIGHT_DECAY,
        "teacher_layers": TEACHER_LAYERS,
        "student_layers": STUDENT_LAYERS,
        "student_decoder_layer": STUDENT_DECODER_LAYER,
        "train_from_layer": TRAIN_FROM_LAYER,
        "lambda_low": LAMBDA_LOW,
        "lambda_mid": LAMBDA_MID,
        "lambda_high": LAMBDA_HIGH,
        "lambda_pred": LAMBDA_PRED,
        "teacher_conf": TEACHER_CONF,
        "evaluation_conf": EVAL_CONF,
        "evaluation_iou": EVAL_IOU,
        "architecture": (
            "Frozen early RT-DETR + trainable late "
            "feature/decoder layers with native RT-DETR "
            "detection loss and multi-granularity teacher "
            "distillation."
        ),
    }

    with open(
        output_dir /
        "config.json",
        "w"
    ) as f:
        json.dump(
            config,
            f,
            indent=2
        )

    history = []

    best_loss = float("inf")
    best_epoch = 0

    # --------------------------------------------------------
    # Training.
    # --------------------------------------------------------

    try:

        for epoch in range(
            1,
            epochs + 1
        ):

            student.model.train()

            epoch_start = time.time()

            sum_total = 0.0
            sum_det = 0.0
            sum_low = 0.0
            sum_mid = 0.0
            sum_high = 0.0
            sum_pred = 0.0

            for batch_index, (
                images,
                targets,
                teacher_batch,
                names
            ) in enumerate(loader):

                images = images.to(
                    device
                )

                student_features.clear()
                student_decoder.clear()

                batch_dict = make_student_batch(
                    images,
                    targets,
                    device
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                # Native RT-DETR training path.
                native_output = student.model(
                    batch_dict
                )

                if isinstance(
                    native_output,
                    tuple
                ):
                    detection_loss = (
                        native_output[0]
                    )
                else:
                    detection_loss = (
                        native_output
                    )

                if not torch.is_tensor(
                    detection_loss
                ):
                    raise RuntimeError(
                        "RT-DETR native loss is not a tensor."
                    )

                if not torch.isfinite(
                    detection_loss
                ):
                    raise RuntimeError(
                        "RT-DETR native detection loss "
                        "became NaN/Inf."
                    )

                # Ensure all three levels were captured.
                required = (
                    list(
                        STUDENT_LAYERS.values()
                    )
                )

                if not all(
                    index in student_features.outputs
                    for index in required
                ):
                    raise RuntimeError(
                        "Missing one or more student "
                        "MGKD feature maps."
                    )

                s_low = (
                    student_features.outputs[
                        STUDENT_LAYERS["low"]
                    ]
                )

                s_mid = (
                    student_features.outputs[
                        STUDENT_LAYERS["mid"]
                    ]
                )

                s_high = (
                    student_features.outputs[
                        STUDENT_LAYERS["high"]
                    ]
                )

                # Teacher attention maps come from disk cache.
                t_low = torch.stack(
                    [
                        item["low"]
                        for item in teacher_batch
                    ]
                ).to(
                    device=device,
                    dtype=s_low.dtype
                )

                t_mid = torch.stack(
                    [
                        item["mid"]
                        for item in teacher_batch
                    ]
                ).to(
                    device=device,
                    dtype=s_mid.dtype
                )

                t_high = torch.stack(
                    [
                        item["high"]
                        for item in teacher_batch
                    ]
                ).to(
                    device=device,
                    dtype=s_high.dtype
                )

                low_loss = feature_kd_loss(
                    s_low,
                    t_low
                )

                mid_loss = feature_kd_loss(
                    s_mid,
                    t_mid
                )

                high_loss = feature_kd_loss(
                    s_high,
                    t_high
                )

                # Prediction KD.
                if student_decoder.output is None:
                    raise RuntimeError(
                        "Student RT-DETR decoder output "
                        "was not captured."
                    )

                student_boxes, student_scores = (
                    parse_student_decoder(
                        student_decoder.output
                    )
                )

                pred_losses = []

                for i in range(
                    len(teacher_batch)
                ):

                    loss_i = prediction_kd_loss(
                        student_boxes[i],
                        student_scores[i],
                        teacher_batch[i]["boxes"],
                        teacher_batch[i]["scores"]
                    )

                    pred_losses.append(
                        loss_i
                    )

                if pred_losses:

                    pred_loss = torch.stack(
                        pred_losses
                    ).mean()

                else:

                    pred_loss = (
                        detection_loss
                        * 0.0
                    )

                total_loss = (
                    detection_loss
                    +
                    LAMBDA_LOW * low_loss
                    +
                    LAMBDA_MID * mid_loss
                    +
                    LAMBDA_HIGH * high_loss
                    +
                    LAMBDA_PRED * pred_loss
                )

                if not torch.isfinite(
                    total_loss
                ):
                    raise RuntimeError(
                        "MGKD total loss became NaN/Inf."
                    )

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    GRAD_CLIP
                )

                optimizer.step()

                sum_total += float(
                    total_loss.detach().item()
                )

                sum_det += float(
                    detection_loss.detach().item()
                )

                sum_low += float(
                    low_loss.detach().item()
                )

                sum_mid += float(
                    mid_loss.detach().item()
                )

                sum_high += float(
                    high_loss.detach().item()
                )

                sum_pred += float(
                    pred_loss.detach().item()
                )

                del (
                    images,
                    targets,
                    teacher_batch,
                    batch_dict,
                    native_output,
                    detection_loss,
                    s_low,
                    s_mid,
                    s_high,
                    t_low,
                    t_mid,
                    t_high,
                    low_loss,
                    mid_loss,
                    high_loss,
                    pred_loss,
                    total_loss,
                    student_boxes,
                    student_scores,
                )

                if (
                    batch_index + 1
                ) % EMPTY_CACHE_EVERY == 0:

                    clear_memory(
                        device
                    )

                if (
                    batch_index == 0
                    or
                    (
                        batch_index + 1
                    ) % 50 == 0
                ):

                    print(
                        f"  Epoch {epoch} | "
                        f"batch "
                        f"{batch_index + 1}/"
                        f"{len(loader)}"
                    )

            scheduler.step()

            num_batches = max(
                len(loader),
                1
            )

            epoch_total = (
                sum_total /
                num_batches
            )

            epoch_det = (
                sum_det /
                num_batches
            )

            epoch_low = (
                sum_low /
                num_batches
            )

            epoch_mid = (
                sum_mid /
                num_batches
            )

            epoch_high = (
                sum_high /
                num_batches
            )

            epoch_pred = (
                sum_pred /
                num_batches
            )

            elapsed = (
                time.time()
                -
                epoch_start
            )

            row = {
                "epoch": epoch,
                "total_loss": epoch_total,
                "student_detection_loss": epoch_det,
                "low_kd_loss": epoch_low,
                "mid_kd_loss": epoch_mid,
                "high_kd_loss": epoch_high,
                "prediction_kd_loss": epoch_pred,
                "learning_rate":
                    optimizer.param_groups[0]["lr"],
                "epoch_time_min":
                    elapsed / 60.0,
            }

            history.append(
                row
            )

            pd.DataFrame(
                history
            ).to_csv(
                output_dir /
                "history.csv",
                index=False
            )

            # Training-loss checkpoint.
            # This is not a paper performance ranking.
            if epoch_total < best_loss:

                best_loss = epoch_total
                best_epoch = epoch

                try:
                    student.save(
                        str(
                            output_dir /
                            "best.pt"
                        )
                    )
                except Exception:

                    torch.save(
                        {
                            "model_state_dict": {
                                k: v.detach().cpu()
                                for k, v
                                in student.model.state_dict().items()
                            },
                            "config": config,
                            "epoch": epoch,
                            "training_loss": epoch_total,
                        },
                        output_dir /
                        "best_state.pt"
                    )

            print("\n" + "-" * 75)
            print(
                f"Epoch {epoch}/{epochs}"
            )
            print(
                f"Total loss : {epoch_total:.5f}"
            )
            print(
                f"Det loss   : {epoch_det:.5f}"
            )
            print(
                f"Low KD     : {epoch_low:.5f}"
            )
            print(
                f"Mid KD     : {epoch_mid:.5f}"
            )
            print(
                f"High KD    : {epoch_high:.5f}"
            )
            print(
                f"Pred KD    : {epoch_pred:.5f}"
            )
            print(
                f"Best loss  : {best_loss:.5f}"
            )
            print(
                f"Best epoch : {best_epoch}"
            )
            print(
                f"Time       : {elapsed/60.0:.2f} min"
            )

    finally:

        student_features.close()
        student_decoder.close()

        try:
            student.save(
                str(
                    output_dir /
                    "last.pt"
                )
            )
        except Exception:

            torch.save(
                {
                    "model_state_dict": {
                        k: v.detach().cpu()
                        for k, v
                        in student.model.state_dict().items()
                    },
                    "config": config,
                    "epoch": epochs,
                },
                output_dir /
                "last_state.pt"
            )

        del student

        clear_memory(
            device
        )

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    summary = {
        "status": "completed",
        "experiment": experiment_name,
        "teacher": exp["teacher_name"],
        "student": "RT-DETR",
        "split": exp["split"],
        "epochs": epochs,
        "batch_size": batch_size,
        "best_training_loss": best_loss,
        "best_training_loss_epoch": best_epoch,
        "standardized_evaluation": {
            "confidence": EVAL_CONF,
            "iou": EVAL_IOU,
            "note": (
                "Run the existing unified standardized "
                "evaluator on the saved best RT-DETR "
                "checkpoint before reporting final metrics."
            ),
        },
    }

    with open(
        output_dir /
        "summary.json",
        "w"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    print("\n" + "=" * 80)
    print("MGKD TRAINING COMPLETE")
    print("=" * 80)
    print(
        f"Experiment : {experiment_name}"
    )
    print(
        f"Best epoch : {best_epoch}"
    )
    print(
        f"Best loss  : {best_loss:.5f}"
    )
    print(
        f"Output     : {output_dir}"
    )
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Lightweight SugarBeet "
            "Multi-Granularity Knowledge Distillation"
        )
    )

    parser.add_argument(
        "--experiment",
        required=True,
        choices=list(
            EXPERIMENTS.keys()
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=IMG_SIZE
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError(
            "--epochs must be >= 1"
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    # Keep CPU thread count small on laptop.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    train_experiment(
        experiment_name=args.experiment,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
