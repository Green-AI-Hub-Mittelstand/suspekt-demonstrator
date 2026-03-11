"""Inference helpers for YOLO/TensorRT pipelines."""

from __future__ import annotations

import numpy as np


def class_wise_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_thresh: float,
) -> list[int]:
    """Run non-maximum suppression per class and return kept indices."""
    keep_indices: list[int] = []
    unique_classes = np.unique(classes)
    for cls in unique_classes:
        inds = np.where(classes == cls)[0]
        cls_boxes = boxes[inds]
        cls_scores = scores[inds]
        order = cls_scores.argsort()[::-1]
        suppressed = np.zeros(len(inds), dtype=bool)
        for i_idx in range(len(order)):
            if suppressed[order[i_idx]]:
                continue
            keep_idx = inds[order[i_idx]]
            keep_indices.append(keep_idx)
            box_i = cls_boxes[order[i_idx]]
            for j_idx in range(i_idx + 1, len(order)):
                if suppressed[order[j_idx]]:
                    continue
                box_j = cls_boxes[order[j_idx]]
                xx1 = max(box_i[0], box_j[0])
                yy1 = max(box_i[1], box_j[1])
                xx2 = min(box_i[2], box_j[2])
                yy2 = min(box_i[3], box_j[3])
                w = max(0.0, xx2 - xx1)
                h = max(0.0, yy2 - yy1)
                inter = w * h
                area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
                area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
                union = area_i + area_j - inter
                iou = inter / union if union > 0 else 0.0
                if iou > iou_thresh:
                    suppressed[order[j_idx]] = True
    return keep_indices
