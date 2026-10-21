#!/usr/bin/env python3
"""
MGKD (Multi-Granularity Knowledge Distillation) for SugarBeet detection
=======================================================================

Teacher : YOLOv11 / YOLOv12   (frozen, run once, knowledge cached to disk)
Student : RT-DETR             (late layers fine-tuned)

Knowledge transferred
    Low  : teacher P3 feature  -> student P3 feature  (80x80 @ 640)
    Mid  : teacher P4 feature  -> student P4 feature  (40x40 @ 640)
    High : teacher P5 feature  -> student P5 feature  (20x20 @ 640)
    Pred : teacher detections  -> student decoder (real queries only)

The layer indices are read from the models themselves (inputs of the
Detect head / RT-DETR decoder), so YOLOv11 and YOLOv12 are both handled.
For YOLOv11 + RT-DETR-L this resolves to YOLO 16/19/22 -> RT-DETR 21/24/27.

Two run modes share *identical* data, augmentation, optimiser, schedule,
BatchNorm handling and checkpoint selection:

    --mode mgkd     : detection loss + feature KD + prediction KD
    --mode control  : detection loss only (all lambdas = 0, no teacher)

The difference between the two runs is therefore attributable to
distillation alone.

Changes relative to the previous version (v1 / knowledge_v2 cache)
    1. Teacher cache is built with the SAME 640x640 square letterbox the
       student sees, so feature maps and pseudo-boxes share one coordinate
       frame. The cache stores metadata (version, image size, teacher hash,
       layers, shapes, thresholds) and is rejected if anything differs.
       Feature maps must match exactly - no silent interpolation.
    2. YOLO output is passed to NMS in its native [B, 4+nc, N] layout
       (the old permute produced garbage pseudo-labels).
    3. BatchNorm running statistics are frozen (and frozen layers stay in
       eval mode), which is required for batch sizes of 1-4.
    4. Feature KD uses the attention-transfer loss (squared L2 distance
       between unit-normalised attention maps, summed over positions),
       giving losses of order 0.1-1 instead of ~1e-5.
    5. Prediction KD strips RT-DETR denoising queries, matches only real
       object queries, keeps matches with IoU >= PRED_MATCH_IOU, and uses
       L1 + GIoU box loss and BCE-with-logits score loss.
    6. Hooks are detached before saving, so checkpoints are real
       Ultralytics .pt files loadable with RTDETR(path).
    7. best.pt is selected by validation fitness (0.1*mAP50 + 0.9*mAP50-95
       at conf=0.001 on the val split), never by training loss.
    8. Horizontal/vertical flip augmentation, applied consistently to the
       image, labels, teacher attention maps and teacher boxes.

Usage
    # Smoke test
    python train_mgkd.py --experiment y11_rt_70 --epochs 1 --max-batches 20

    # Distillation run
    python train_mgkd.py --experiment y11_rt_70 --mode mgkd --epochs 50

    # Matched control run (same settings, no distillation)
    python train_mgkd.py --experiment y11_rt_70 --mode control --epochs 50

Report final paper numbers by running the project's standardized
evaluator on <output>/best.pt for BOTH runs and for the RT-DETR baseline,
on the test split. Compute P/R/F1 at the chosen operating confidence
(e.g. 0.25) but mAP50 / mAP50-95 at a low confidence (0.001), identically
for every model.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import random
import shutil
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou, generalized_box_iou_loss

from ultralytics import RTDETR, YOLO
from ultralytics.utils.nms import non_max_suppression


# ============================================================
# GLOBAL CONFIG
# ============================================================

BASE = Path("/Users/kalamanjunatha/Downloads/sugarbeet")


def build_experiments(base: Path) -> dict:
    d70 = base / "dataset_70_20_10_detection"
    d80 = base / "dataset_80_10_10_detection"
    m = {
        "y11_70": base / "Experiments/70_20_10/YOLOv11_70_20_10_Detection/weights/best.pt",
        "y12_70": base / "Experiments/70_20_10/YOLOv12_70_20_10_Detection/weights/best.pt",
        "rt_70": base / "Experiments/70_20_10/RT_DETR_70_20_10/weights/best.pt",
        "y11_80": base / "Experiments/80_10_10/YOLOv11/detection/weights/best.pt",
        "y12_80": base / "Experiments/80_10_10/YOLOv12/detection-2/weights/best.pt",
        "rt_80": base / "Experiments/80_10_10/RT_DETR4/weights/best.pt",
    }
    return {
        "y11_rt_70": dict(split="70_20_10", dataset=d70, teacher=m["y11_70"], student=m["rt_70"],
                          teacher_name="YOLOv11", name="MGKD_YOLOv11_RTDETR_70_20_10"),
        "y12_rt_70": dict(split="70_20_10", dataset=d70, teacher=m["y12_70"], student=m["rt_70"],
                          teacher_name="YOLOv12", name="MGKD_YOLOv12_RTDETR_70_20_10"),
        "y11_rt_80": dict(split="80_10_10", dataset=d80, teacher=m["y11_80"], student=m["rt_80"],
                          teacher_name="YOLOv11", name="MGKD_YOLOv11_RTDETR_80_10_10"),
        "y12_rt_80": dict(split="80_10_10", dataset=d80, teacher=m["y12_80"], student=m["rt_80"],
                          teacher_name="YOLOv12", name="MGKD_YOLOv12_RTDETR_80_10_10"),
    }


EXPERIMENT_KEYS = ["y11_rt_70", "y12_rt_70", "y11_rt_80", "y12_rt_80"]

# The class being detected (SugarBeet project is single-class).
TARGET_CLASS = 0

# Loss weights (mgkd mode). In control mode all are forced to 0.
LAMBDA_LOW = 0.05
LAMBDA_MID = 0.05
LAMBDA_HIGH = 0.10
LAMBDA_PRED = 0.10

# Prediction KD.
TEACHER_CONF = 0.25
TEACHER_NMS_IOU = 0.70
TEACHER_MAX_DET = 50
PRED_MATCH_IOU = 0.50       # discard Hungarian matches below this IoU
PRED_SCORE_WEIGHT = 0.50    # weight of score term inside prediction KD

# Training.
IMG_SIZE = 640
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0
FLIP_PROB = 0.5

# Freeze early student backbone/neck: layers < TRAIN_FROM_LAYER frozen.
TRAIN_FROM_LAYER = 21

# Validation used ONLY for checkpoint selection.
VAL_CONF = 0.001
VAL_IOU = 0.70

CACHE_VERSION = "mgkd_v3_square_letterbox"
CACHE_FILENAME = "knowledge_v3.pt"
EMPTY_CACHE_EVERY = 5
LETTERBOX_FILL = 114


# ============================================================
# DEVICE / MEMORY / SEED
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ultralytics_device(device: torch.device) -> str:
    if device.type == "cuda":
        return str(device.index or 0)
    return device.type


def clear_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# DATASET
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SugarBeetDataset(torch.utils.data.Dataset):
    """Square, centred 640x640 letterbox. Labels -> normalised canvas xywh."""

    def __init__(self, root: Path, split: str, img_size: int = IMG_SIZE):
        self.root = Path(root)
        self.split = split
        self.img_size = int(img_size)
        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split

        for d in (self.image_dir, self.label_dir):
            if not d.exists():
                raise FileNotFoundError(f"Directory not found:\n{d}")

        self.images = sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.images:
            raise RuntimeError(f"No images found in {self.image_dir}")

        missing = [p.name for p in self.images if not (self.label_dir / f"{p.stem}.txt").exists()]
        if missing:
            raise RuntimeError(f"Missing label files. Examples: {missing[:10]}")

        self.skipped_other_class = 0
        print(f"{split}: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def read_labels(self, image_path: Path) -> torch.Tensor:
        rows = []
        with open(self.label_dir / f"{image_path.stem}.txt") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    x, y, w, h = map(float, parts[1:5])
                except ValueError:
                    continue
                if cls != TARGET_CLASS:
                    self.skipped_other_class += 1
                    continue
                rows.append([0.0, min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0),
                             min(max(w, 1e-6), 1.0), min(max(h, 1e-6), 1.0)])
        if not rows:
            return torch.empty((0, 5), dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)

    def letterbox(self, image: Image.Image, labels: torch.Tensor):
        ow, oh = image.size
        s = self.img_size
        scale = min(s / ow, s / oh)
        nw, nh = max(1, int(round(ow * scale))), max(1, int(round(oh * scale)))
        resized = image.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (s, s), (LETTERBOX_FILL,) * 3)
        pad_x, pad_y = (s - nw) // 2, (s - nh) // 2
        canvas.paste(resized, (pad_x, pad_y))

        image_tensor = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()

        if len(labels):
            out = labels.clone()
            out[:, 1] = (labels[:, 1] * ow * scale + pad_x) / s
            out[:, 2] = (labels[:, 2] * oh * scale + pad_y) / s
            out[:, 3] = labels[:, 3] * ow * scale / s
            out[:, 4] = labels[:, 4] * oh * scale / s
            out[:, 1:3] = out[:, 1:3].clamp(0.0, 1.0)
            out[:, 3:5] = out[:, 3:5].clamp(1e-6, 1.0)
        else:
            out = labels
        return image_tensor, out

    def __getitem__(self, index):
        path = self.images[index]
        image = Image.open(path).convert("RGB")
        image_tensor, labels = self.letterbox(image, self.read_labels(path))
        return image_tensor, labels, path.name


def collate_raw(batch):
    return torch.stack([x[0] for x in batch]), [x[1] for x in batch], [x[2] for x in batch]


# ============================================================
# FEATURE HOOKS (detachable, so checkpoints stay picklable)
# ============================================================

class HookStore:
    """Captures outputs of selected layers. attach()/detach() around saving."""

    def __init__(self, model, layer_indices, keep_raw: bool = False):
        self.model = model
        self.layer_indices = list(layer_indices)
        self.keep_raw = keep_raw
        self.outputs: dict = {}
        self.handles: list = []
        self.attach()

    def _hook(self, index):
        def hook(module, inputs, output):
            if not self.keep_raw and isinstance(output, (tuple, list)):
                output = output[0]
            self.outputs[index] = output
        return hook

    def attach(self):
        if self.handles:
            return
        for idx in self.layer_indices:
            self.handles.append(self.model.model[idx].register_forward_hook(self._hook(idx)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.outputs.clear()

    def clear(self):
        self.outputs.clear()


@contextlib.contextmanager
def hooks_detached(*stores):
    stores = [s for s in stores if s is not None]
    for s in stores:
        s.detach()
    try:
        yield
    finally:
        for s in stores:
            s.attach()


def head_input_layers(model) -> list[int]:
    """Indices of the layers feeding the detection head (P3, P4, P5)."""
    head = model.model[-1]
    f = head.f if isinstance(head.f, (list, tuple)) else [head.f]
    f = [int(i) if i >= 0 else len(model.model) - 1 + i for i in f]
    if len(f) != 3:
        raise RuntimeError(f"Expected 3 head inputs (P3,P4,P5), got {f} for {type(head).__name__}")
    return f


# ============================================================
# ATTENTION-TRANSFER FEATURE KD
# ============================================================

def spatial_attention(feature: torch.Tensor) -> torch.Tensor:
    """[B,C,H,W] -> unit-L2-normalised spatial attention [B,1,H,W]."""
    if feature.ndim != 4:
        raise RuntimeError(f"Expected [B,C,H,W], got {tuple(feature.shape)}")
    att = feature.float().pow(2).mean(dim=1, keepdim=True)
    flat = att.flatten(1)
    flat = flat / (flat.norm(p=2, dim=1, keepdim=True) + 1e-6)
    return flat.reshape(att.shape)


def feature_kd_loss(student_feature: torch.Tensor, teacher_attention: torch.Tensor) -> torch.Tensor:
    """Attention transfer: ||a_s - a_t||^2 summed over positions, mean over batch.

    Shapes must match exactly; a mismatch means the teacher cache was built
    with a different letterbox and is an error, not something to interpolate.
    """
    s = spatial_attention(student_feature)
    t = teacher_attention.to(device=s.device, dtype=s.dtype)
    if s.shape != t.shape:
        raise RuntimeError(f"Teacher/student attention shape mismatch: {tuple(t.shape)} vs {tuple(s.shape)}")
    return (s - t).pow(2).flatten(1).sum(dim=1).mean()


# ============================================================
# BOX HELPERS
# ============================================================

def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


# ============================================================
# TEACHER CACHE
# ============================================================

def teacher_predictions(raw_output, img_size: int, conf: float) -> list[dict]:
    """YOLO eval output -> per-image normalised cxcywh boxes + scores.

    Ultralytics NMS expects the native [B, 4+nc, N] layout, so the tensor is
    passed through unchanged.
    """
    pred = raw_output[0] if isinstance(raw_output, (tuple, list)) else raw_output
    if not torch.is_tensor(pred) or pred.ndim != 3:
        raise RuntimeError(f"Unexpected YOLO output: {type(pred)}")
    if pred.shape[1] >= pred.shape[2]:
        raise RuntimeError(f"YOLO output not in [B, 4+nc, N] layout: {tuple(pred.shape)}")

    dets = non_max_suppression(pred, conf_thres=conf, iou_thres=TEACHER_NMS_IOU,
                               max_det=TEACHER_MAX_DET, classes=[TARGET_CLASS])
    out = []
    for d in dets:
        if d is None or len(d) == 0:
            out.append({"boxes": torch.empty((0, 4)), "scores": torch.empty((0,))})
            continue
        xyxy = (d[:, :4] / float(img_size)).clamp(0.0, 1.0)
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
        w = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=1e-6)
        h = (xyxy[:, 3] - xyxy[:, 1]).clamp(min=1e-6)
        out.append({"boxes": torch.stack([cx, cy, w, h], 1).float().cpu(),
                    "scores": d[:, 4].clamp(0, 1).float().cpu()})
    return out


def expected_cache_meta(teacher_path: Path, dataset: SugarBeetDataset, teacher_conf: float) -> dict:
    s = dataset.img_size
    return {
        "version": CACHE_VERSION,
        "img_size": s,
        "letterbox": f"square_centered_fill{LETTERBOX_FILL}",
        "teacher_path": str(teacher_path),
        "teacher_sha256": file_sha256(teacher_path),
        "teacher_conf": teacher_conf,
        "teacher_nms_iou": TEACHER_NMS_IOU,
        "teacher_max_det": TEACHER_MAX_DET,
        "target_class": TARGET_CLASS,
        "shapes": {"low": [1, s // 8, s // 8], "mid": [1, s // 16, s // 16], "high": [1, s // 32, s // 32]},
        "names": [p.name for p in dataset.images],
    }


def validate_cache(cache: dict, expected: dict, path: Path) -> None:
    meta = cache.get("meta")
    if meta is None:
        raise RuntimeError(f"{path} has no metadata (old v1/v2 cache). Re-run with --rebuild-cache.")
    problems = [k for k in expected if meta.get(k) != expected[k]]
    if problems:
        raise RuntimeError(f"Teacher cache {path} does not match this run ({', '.join(problems)}). "
                           "Re-run with --rebuild-cache.")
    for item in cache["items"]:
        for k, shp in expected["shapes"].items():
            if list(item[k].shape) != shp:
                raise RuntimeError(f"Cache item {item['name']} {k} shape {list(item[k].shape)} != {shp}")


def build_or_load_teacher_cache(teacher_path, dataset, cache_path, device, teacher_conf, rebuild):
    expected = expected_cache_meta(teacher_path, dataset, teacher_conf)

    if cache_path.exists() and not rebuild:
        print(f"\nLoading teacher cache: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        validate_cache(cache, expected, cache_path)
        print(f"Teacher cache OK: {len(cache['items'])} images, version {CACHE_VERSION}")
        return cache

    print("\n" + "=" * 70)
    print("PHASE 1: ONE-TIME TEACHER CACHE")
    print("=" * 70)
    print(f"Teacher: {teacher_path}\nDevice : {device}\nImages : {len(dataset)}")

    teacher = YOLO(str(teacher_path))
    tmodel = teacher.model.to(device).float().eval()
    for p in tmodel.parameters():
        p.requires_grad = False

    layers = head_input_layers(tmodel)
    print(f"Teacher feature layers (Detect inputs): {layers}")
    hooks = HookStore(tmodel, layers)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                         collate_fn=collate_raw)
    items, start = [], time.time()
    try:
        with torch.no_grad():
            for bi, (images, _, names) in enumerate(loader):
                hooks.clear()
                raw = tmodel(images.to(device))
                preds = teacher_predictions(raw, dataset.img_size, teacher_conf)
                att = {lvl: spatial_attention(hooks.outputs[idx]).cpu()
                       for lvl, idx in zip(("low", "mid", "high"), layers)}
                for i, name in enumerate(names):
                    item = {"name": name, "boxes": preds[i]["boxes"], "scores": preds[i]["scores"]}
                    for lvl in ("low", "mid", "high"):
                        item[lvl] = att[lvl][i].contiguous()
                        if list(item[lvl].shape) != expected["shapes"][lvl]:
                            raise RuntimeError(f"Teacher {lvl} attention shape {list(item[lvl].shape)} "
                                               f"!= expected {expected['shapes'][lvl]}")
                    items.append(item)
                if (bi + 1) % EMPTY_CACHE_EVERY == 0:
                    clear_memory(device)
                if bi == 0 or (bi + 1) % 25 == 0:
                    print(f"Cached {len(items)}/{len(dataset)} | {(time.time() - start) / 60:.2f} min")
    finally:
        hooks.detach()
        del hooks, tmodel, teacher
        clear_memory(device)

    n_boxes = [len(it["boxes"]) for it in items]
    print(f"Teacher pseudo-boxes/image: mean {np.mean(n_boxes):.2f}, "
          f"images with none: {sum(n == 0 for n in n_boxes)}")

    cache = {"meta": {**expected, "teacher_layers": layers}, "items": items}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"Teacher cache saved: {cache_path}")
    return cache


# ============================================================
# TRAINING DATASET (+ consistent flip augmentation)
# ============================================================

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, base: SugarBeetDataset, teacher_items: list | None, flip_prob: float):
        if teacher_items is not None:
            if len(teacher_items) != len(base):
                raise RuntimeError("Teacher cache and dataset lengths differ.")
        self.base = base
        self.teacher_items = teacher_items
        self.flip_prob = flip_prob

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, labels, name = self.base[index]
        t = None
        if self.teacher_items is not None:
            src = self.teacher_items[index]
            if src["name"] != name:
                raise RuntimeError(f"Dataset/cache ordering mismatch: {src['name']} != {name}")
            t = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in src.items()}

        labels = labels.clone()
        # dim -1 = width (x, column 1), dim -2 = height (y, column 2)
        for dim, col in ((-1, 1), (-2, 2)):
            if self.flip_prob > 0 and torch.rand(()) < self.flip_prob:
                image = image.flip(dim)
                if len(labels):
                    labels[:, col] = 1.0 - labels[:, col]
                if t is not None:
                    for lvl in ("low", "mid", "high"):
                        t[lvl] = t[lvl].flip(dim)
                    if len(t["boxes"]):
                        t["boxes"][:, col - 1] = 1.0 - t["boxes"][:, col - 1]
        return image, labels, t, name


def collate_train(batch):
    return (torch.stack([x[0] for x in batch]), [x[1] for x in batch],
            [x[2] for x in batch], [x[3] for x in batch])


def make_student_batch(images, targets, device):
    cls, boxes, bidx = [], [], []
    for i, t in enumerate(targets):
        if t is None or t.numel() == 0:
            continue
        t = t.reshape(-1, 5)
        cls.append(t[:, :1])
        boxes.append(t[:, 1:5])
        bidx.append(torch.full((len(t),), i, dtype=torch.long))
    if cls:
        cls, boxes, bidx = torch.cat(cls), torch.cat(boxes), torch.cat(bidx)
    else:
        cls, boxes, bidx = torch.empty((0, 1)), torch.empty((0, 4)), torch.empty((0,), dtype=torch.long)
    return {"img": images.to(device), "cls": cls.to(device), "bboxes": boxes.to(device),
            "batch_idx": bidx.to(device)}


# ============================================================
# PREDICTION KD
# ============================================================

def parse_student_decoder(decoder_output):
    """Last decoder layer, real object queries only (denoising queries removed).

    Returns boxes [B,Q,4] (normalised cxcywh, already sigmoid) and class
    logits [B,Q] for TARGET_CLASS.
    """
    if not isinstance(decoder_output, (tuple, list)) or len(decoder_output) != 5:
        raise RuntimeError("Unexpected RT-DETR decoder output; expected training-mode 5-tuple.")
    dec_bboxes, dec_scores, _, _, dn_meta = decoder_output
    boxes, logits = dec_bboxes[-1], dec_scores[-1]
    if dn_meta is not None:
        num_dn = int(dn_meta["dn_num_split"][0])
        boxes, logits = boxes[:, num_dn:], logits[:, num_dn:]
    return boxes, logits[..., TARGET_CLASS]


def prediction_kd_loss(s_boxes, s_logits, t_boxes, t_scores):
    """Hungarian-matched teacher->student box/score distillation for one image."""
    zero = s_boxes.sum() * 0.0
    if len(t_boxes) == 0 or len(s_boxes) == 0:
        return zero, 0

    t_boxes = t_boxes.to(device=s_boxes.device, dtype=s_boxes.dtype)
    t_scores = t_scores.to(device=s_logits.device, dtype=s_logits.dtype)

    with torch.no_grad():
        s_xyxy, t_xyxy = cxcywh_to_xyxy(s_boxes), cxcywh_to_xyxy(t_boxes)
        iou = box_iou(s_xyxy, t_xyxy)
        cost = torch.cdist(s_boxes, t_boxes, p=1) + (1.0 - iou) - s_logits.sigmoid()[:, None] * t_scores[None, :]
        rows, cols = linear_sum_assignment(cost.float().cpu().numpy())
        rows = torch.as_tensor(rows, dtype=torch.long, device=s_boxes.device)
        cols = torch.as_tensor(cols, dtype=torch.long, device=s_boxes.device)
        keep = iou[rows, cols] >= PRED_MATCH_IOU
        rows, cols = rows[keep], cols[keep]

    if len(rows) == 0:
        return zero, 0

    sb, tb = s_boxes[rows], t_boxes[cols]
    box_loss = F.l1_loss(sb, tb) + generalized_box_iou_loss(cxcywh_to_xyxy(sb), cxcywh_to_xyxy(tb),
                                                            reduction="mean")
    score_loss = F.binary_cross_entropy_with_logits(s_logits[rows], t_scores[cols])
    return box_loss + PRED_SCORE_WEIGHT * score_loss, int(len(rows))


# ============================================================
# STUDENT HELPERS
# ============================================================

def set_student_train_mode(model: nn.Module, freeze_bn: bool) -> None:
    model.train()
    for idx in range(min(TRAIN_FROM_LAYER, len(model.model))):
        model.model[idx].eval()
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def save_student(model: nn.Module, base_ckpt: dict, path: Path, extra: dict) -> None:
    """Write an Ultralytics-compatible checkpoint (hooks must be detached)."""
    m = deepcopy(model).cpu().half()
    if hasattr(m, "criterion"):
        m.criterion = None
    for p in m.parameters():
        p.requires_grad = False
    meta = {k: v for k, v in (base_ckpt or {}).items() if k not in ("model", "ema", "optimizer")}
    torch.save({**meta, "model": m, "ema": None, "optimizer": None, "mgkd": extra,
                "date": datetime.now().isoformat()}, path)


def write_data_yaml(dataset_root: Path, names: dict, out_path: Path) -> Path:
    lines = [f"path: {dataset_root}", "train: images/train", "val: images/val"]
    if (dataset_root / "images" / "test").exists():
        lines.append("test: images/test")
    lines.append("names:")
    for k in sorted(names):
        lines.append(f"  {int(k)}: {json.dumps(str(names[k]))}")
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def validate_checkpoint(ckpt_path: Path, data_yaml: Path, img_size: int, device: torch.device,
                        out_dir: Path) -> dict:
    model = RTDETR(str(ckpt_path))
    r = model.val(data=str(data_yaml), split="val", imgsz=img_size, batch=1, conf=VAL_CONF, iou=VAL_IOU,
                  device=ultralytics_device(device), plots=False, verbose=False, save_json=False,
                  project=str(out_dir), name="val", exist_ok=True)
    b = r.box
    metrics = {"val_precision": float(b.mp), "val_recall": float(b.mr),
               "val_mAP50": float(b.map50), "val_mAP50_95": float(b.map)}
    metrics["val_fitness"] = 0.1 * metrics["val_mAP50"] + 0.9 * metrics["val_mAP50_95"]
    del model
    return metrics


# ============================================================
# TRAINING
# ============================================================

def train(args) -> None:
    seed_everything(args.seed)
    experiments = build_experiments(Path(args.base))
    exp = dict(experiments[args.experiment])
    for key in ("teacher", "student", "dataset"):
        if getattr(args, key):
            exp[key] = Path(getattr(args, key))

    control = args.mode == "control"
    lam = dict(low=0.0, mid=0.0, high=0.0, pred=0.0) if control else \
        dict(low=LAMBDA_LOW, mid=LAMBDA_MID, high=LAMBDA_HIGH, pred=LAMBDA_PRED)

    device = get_device(args.device)
    if device.type == "mps":
        torch.set_num_threads(1)

    run_name = exp["name"] + ("_CONTROL" if control else "")
    output_root = Path(args.output_root) if args.output_root else Path(args.base) / "Experiments/Hybrid_Approaches/MGKD/results"
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_path, student_path, dataset_root = Path(exp["teacher"]), Path(exp["student"]), Path(exp["dataset"])
    for label, p in (("Student", student_path), ("Dataset", dataset_root)) + (() if control else (("Teacher", teacher_path),)):
        if not p.exists():
            raise FileNotFoundError(f"{label} not found:\n{p}")

    print("\n" + "=" * 80)
    print(f"MGKD TRAINING  [{args.mode.upper()}]")
    print("=" * 80)
    print(f"Experiment : {args.experiment}   Split: {exp['split']}   Device: {device}")
    print(f"Teacher    : {'(none - control run)' if control else exp['teacher_name']}")
    print(f"Epochs     : {args.epochs}   Batch: {args.batch_size}   Img: {args.img_size}   LR: {args.lr}")
    print(f"Lambdas    : low {lam['low']}, mid {lam['mid']}, high {lam['high']}, pred {lam['pred']}")
    print(f"BN frozen  : {not args.train_bn}   Flip prob: {args.flip_prob}   Seed: {args.seed}")
    print(f"Output     : {output_dir}")
    print("=" * 80)

    train_dataset = SugarBeetDataset(dataset_root, "train", args.img_size)

    # ---------------- Teacher cache (mgkd only) ----------------
    teacher_items = None
    if not control:
        cache = build_or_load_teacher_cache(teacher_path, train_dataset,
                                            output_dir / "teacher_cache" / CACHE_FILENAME,
                                            device, args.teacher_conf, args.rebuild_cache)
        teacher_items = cache["items"]
        del cache

    # ---------------- Student ----------------
    print("\n" + "=" * 70 + "\nPHASE 2: STUDENT\n" + "=" * 70)
    student = RTDETR(str(student_path))
    model = student.model.to(device).float()
    base_ckpt = student.ckpt if isinstance(student.ckpt, dict) else {}
    names = dict(model.names) if hasattr(model, "names") else {0: "sugarbeet"}
    if not hasattr(model, "nc"):
        model.nc = model.model[-1].nc
    model.criterion = model.init_criterion()

    for p in model.parameters():
        p.requires_grad = False
    for idx in range(TRAIN_FROM_LAYER, len(model.model)):
        for p in model.model[idx].parameters():
            p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("Student has zero trainable parameters.")
    print(f"Total parameters     : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters : {sum(p.numel() for p in trainable):,}  (layers {TRAIN_FROM_LAYER}..{len(model.model) - 1})")

    student_layers = head_input_layers(model)
    decoder_index = len(model.model) - 1
    print(f"Student feature layers (decoder inputs): {student_layers}   decoder: {decoder_index}")

    feat_hooks = dec_hook = None
    if not control:
        feat_hooks = HookStore(model, student_layers)
        dec_hook = HookStore(model, [decoder_index], keep_raw=True)

    # ---------------- Data / optimiser ----------------
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(TrainDataset(train_dataset, teacher_items, args.flip_prob),
                                         batch_size=args.batch_size, shuffle=True, num_workers=0,
                                         collate_fn=collate_train, generator=g)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    data_yaml = write_data_yaml(dataset_root, names, output_dir / "data_mgkd.yaml")
    config = {
        "mode": args.mode, "experiment": args.experiment, "teacher_name": exp["teacher_name"],
        "teacher_model": str(teacher_path), "student_model": str(student_path), "dataset": str(dataset_root),
        "split": exp["split"], "device": str(device), "seed": args.seed, "image_size": args.img_size,
        "batch_size": args.batch_size, "epochs": args.epochs, "learning_rate": args.lr,
        "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "flip_prob": args.flip_prob,
        "bn_frozen": not args.train_bn, "train_from_layer": TRAIN_FROM_LAYER,
        "student_feature_layers": student_layers, "student_decoder_layer": decoder_index,
        "lambdas": lam, "feature_kd": "attention_transfer_sum_sq_l2",
        "teacher_conf": args.teacher_conf, "pred_match_iou": PRED_MATCH_IOU,
        "pred_score_weight": PRED_SCORE_WEIGHT, "cache_version": None if control else CACHE_VERSION,
        "selection": f"val fitness (0.1*mAP50 + 0.9*mAP50-95), conf={VAL_CONF}, iou={VAL_IOU}",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    if train_dataset.skipped_other_class:
        print(f"NOTE: {train_dataset.skipped_other_class} labels with class != {TARGET_CLASS} ignored so far.")

    history, best = [], {"fitness": -1.0, "epoch": 0, "metrics": None}
    selection_note = "validation fitness"
    last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"

    try:
        for epoch in range(1, args.epochs + 1):
            set_student_train_mode(model, freeze_bn=not args.train_bn)
            t0 = time.time()
            sums = dict(total=0.0, det=0.0, low=0.0, mid=0.0, high=0.0, pred=0.0, kd_weighted=0.0, matches=0)
            n_batches = 0

            for bi, (images, targets, teacher_batch, _) in enumerate(loader):
                if args.max_batches and bi >= args.max_batches:
                    break
                if feat_hooks:
                    feat_hooks.clear()
                    dec_hook.clear()

                batch = make_student_batch(images, targets, device)
                optimizer.zero_grad(set_to_none=True)
                out = model(batch)
                det_loss = out[0] if isinstance(out, (tuple, list)) else out
                det_loss = det_loss.sum()
                if not torch.isfinite(det_loss):
                    raise RuntimeError("RT-DETR detection loss became NaN/Inf.")

                total = det_loss
                if not control:
                    feats = feat_hooks.outputs
                    kd = {}
                    for lvl, idx in zip(("low", "mid", "high"), student_layers):
                        t_att = torch.stack([t[lvl] for t in teacher_batch])
                        kd[lvl] = feature_kd_loss(feats[idx], t_att)

                    s_boxes, s_logits = parse_student_decoder(dec_hook.outputs[decoder_index])
                    pl, nm = [], 0
                    for i, t in enumerate(teacher_batch):
                        li, mi = prediction_kd_loss(s_boxes[i], s_logits[i], t["boxes"], t["scores"])
                        pl.append(li)
                        nm += mi
                    kd["pred"] = torch.stack(pl).mean()

                    kd_weighted = sum(lam[k] * kd[k] for k in ("low", "mid", "high", "pred"))
                    total = det_loss + kd_weighted
                    for k in ("low", "mid", "high", "pred"):
                        sums[k] += float(kd[k].detach())
                    sums["kd_weighted"] += float(kd_weighted.detach())
                    sums["matches"] += nm

                if not torch.isfinite(total):
                    raise RuntimeError("MGKD total loss became NaN/Inf.")
                total.backward()
                torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                optimizer.step()

                sums["total"] += float(total.detach())
                sums["det"] += float(det_loss.detach())
                n_batches += 1

                if (bi + 1) % EMPTY_CACHE_EVERY == 0:
                    clear_memory(device)
                if bi == 0 or (bi + 1) % 50 == 0:
                    print(f"  Epoch {epoch} | batch {bi + 1}/{len(loader)} | total {float(total.detach()):.4f}")

            epoch_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            n = max(n_batches, 1)
            row = {"epoch": epoch, "total_loss": sums["total"] / n, "detection_loss": sums["det"] / n,
                   "low_kd": sums["low"] / n, "mid_kd": sums["mid"] / n, "high_kd": sums["high"] / n,
                   "pred_kd": sums["pred"] / n, "weighted_kd": sums["kd_weighted"] / n,
                   "kd_share_of_total": (sums["kd_weighted"] / sums["total"]) if sums["total"] else 0.0,
                   "pred_matches_per_batch": sums["matches"] / n,
                   "learning_rate": epoch_lr,
                   "epoch_time_min": (time.time() - t0) / 60.0}

            # ---- save + validate (hooks detached so the model pickles) ----
            model.eval()
            with hooks_detached(feat_hooks, dec_hook):
                save_student(model, base_ckpt, last_path, {"epoch": epoch, "mode": args.mode, "config": config})
                do_val = args.val_every > 0 and (epoch % args.val_every == 0 or epoch == args.epochs)
                if do_val:
                    try:
                        metrics = validate_checkpoint(last_path, data_yaml, args.img_size, device, output_dir)
                        row.update(metrics)
                        if metrics["val_fitness"] > best["fitness"]:
                            best = {"fitness": metrics["val_fitness"], "epoch": epoch, "metrics": metrics}
                            shutil.copy2(last_path, best_path)
                    except Exception as e:  # noqa: BLE001 - reported loudly, never swallowed
                        print(f"\n!!! VALIDATION FAILED at epoch {epoch}: {type(e).__name__}: {e}")
                        selection_note = f"last epoch (validation failed: {type(e).__name__}: {e})"
                        args.val_every = 0
            clear_memory(device)

            history.append(row)
            pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

            print("\n" + "-" * 75)
            print(f"Epoch {epoch}/{args.epochs}  ({row['epoch_time_min']:.2f} min)")
            print(f"  Det loss {row['detection_loss']:.4f} | total {row['total_loss']:.4f}")
            if not control:
                print(f"  KD low {row['low_kd']:.4f} mid {row['mid_kd']:.4f} high {row['high_kd']:.4f} "
                      f"pred {row['pred_kd']:.4f} | weighted KD share {100 * row['kd_share_of_total']:.1f}% "
                      f"| matches/batch {row['pred_matches_per_batch']:.2f}")
            if "val_fitness" in row:
                print(f"  Val P {row['val_precision']:.4f} R {row['val_recall']:.4f} "
                      f"mAP50 {row['val_mAP50']:.4f} mAP50-95 {row['val_mAP50_95']:.4f} "
                      f"| best epoch {best['epoch']}")
    finally:
        for h in (feat_hooks, dec_hook):
            if h is not None:
                h.detach()

    if args.val_every == 0 or best["epoch"] == 0:
        shutil.copy2(last_path, best_path)
        best = {"fitness": None, "epoch": args.epochs, "metrics": None}
        if selection_note == "validation fitness":
            selection_note = "last epoch (validation disabled)"

    summary = {
        "status": "completed", "mode": args.mode, "experiment": args.experiment, "run_name": run_name,
        "teacher": None if control else exp["teacher_name"], "student": "RT-DETR", "split": exp["split"],
        "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
        "checkpoint_selection": selection_note, "best_epoch": best["epoch"], "best_val_metrics": best["metrics"],
        "best_checkpoint": str(best_path), "last_checkpoint": str(last_path),
        "reporting_note": ("Evaluate best.pt of the MGKD run, the CONTROL run and the RT-DETR baseline on the "
                           "TEST split with the standardized evaluator. P/R/F1 at the operating confidence "
                           "(e.g. 0.25); mAP50 and mAP50-95 at conf=0.001, identically for all models."),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 80)
    print(f"MGKD TRAINING COMPLETE [{args.mode.upper()}]")
    print(f"Best epoch : {best['epoch']}  ({selection_note})")
    print(f"Best ckpt  : {best_path}")
    print("=" * 80)
    del student, model
    clear_memory(device)


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser(description="SugarBeet Multi-Granularity Knowledge Distillation (YOLO -> RT-DETR)")
    p.add_argument("--experiment", required=True, choices=EXPERIMENT_KEYS)
    p.add_argument("--mode", choices=["mgkd", "control"], default="mgkd",
                   help="mgkd = detection + distillation; control = identical fine-tuning, no distillation")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--flip-prob", type=float, default=FLIP_PROB, help="per-axis flip probability (0 disables)")
    p.add_argument("--train-bn", action="store_true", help="update BatchNorm stats (not recommended for batch < 8)")
    p.add_argument("--teacher-conf", type=float, default=TEACHER_CONF)
    p.add_argument("--val-every", type=int, default=1, help="validate every N epochs (0 = keep last epoch)")
    p.add_argument("--rebuild-cache", action="store_true", help="rebuild the teacher cache even if one exists")
    p.add_argument("--max-batches", type=int, default=0, help="limit batches per epoch (smoke tests)")
    p.add_argument("--device", default=None, help="override device, e.g. mps, cuda:0, cpu")
    p.add_argument("--base", default=str(BASE), help="project root")
    p.add_argument("--output-root", default=None, help="override results directory")
    p.add_argument("--teacher", default=None, help="override teacher weights path")
    p.add_argument("--student", default=None, help="override student weights path")
    p.add_argument("--dataset", default=None, help="override dataset root")
    args = p.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be >= 1")
    if args.img_size % 32:
        raise ValueError("--img-size must be a multiple of 32")
    train(args)


if __name__ == "__main__":
    main()
