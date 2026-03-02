"""
Compare embeddings between:
- Image A normal face
- Image A masked face (lower-right quadrant zeroed)
and
- Image B face (pipeline: detect + align)

Usage:
    python3 apps/face/debug_compare_a_b.py \
        --image-a data/face/face-import-pics/tungdtmask.jpg \
        --image-b data/face/face-import-pics/tungdt2.jpg \
        --out-dir data/tmp/face-debug-v2
"""

import argparse
import os
from typing import Tuple

import cv2
import numpy as np

from scr_onnx import SCRFD
from apps.face.arcface_onnx import IRES
from utils import align_face


ARCFACE_PATH = os.getenv(
    "ARCFACE_MODEL",
    "/home/jetson/tungdt/vms/data/face/models/arcface/arcface_r100.onnx",
)
SCRFD_PATH = os.getenv(
    "SCRFD_MODEL",
    "/home/jetson/tungdt/vms/data/face/models/scrfd640/scrfd_2.5g_bnkps_dynamic.onnx",
)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _cosine_l2(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    cosine = float(np.dot(a, b))
    l2 = float(np.linalg.norm(a - b))
    return cosine, l2


def _detect_align_best(det: SCRFD, image: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    bboxes, kpss = det.detect(image, 0.5, input_size=(640, 640))
    if len(bboxes) == 0:
        raise RuntimeError(f"No face detected in {label}")

    best = int(np.argmax([b[4] for b in bboxes]))
    x1, y1, x2, y2 = bboxes[best][:4]
    lm = kpss[best]

    aligned = align_face(image.copy(), [x1, y1, x2, y2], lm)

    vis = image.copy()
    score = float(bboxes[best][4])
    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(
        vis,
        f"score={score:.3f}",
        (int(x1), max(0, int(y1) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    return aligned, vis


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare A(normal/mask) embeddings against B embedding."
    )
    parser.add_argument("--image-a", required=True, help="Input image A path")
    parser.add_argument("--image-b", required=True, help="Input image B path")
    parser.add_argument(
        "--out-dir", default="data/tmp/face-debug", help="Output directory for debug images"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    image_a = cv2.imread(args.image_a)
    if image_a is None:
        print(f"[ERROR] Cannot read image A: {args.image_a}")
        return 1
    image_b = cv2.imread(args.image_b)
    if image_b is None:
        print(f"[ERROR] Cannot read image B: {args.image_b}")
        return 1

    print("[debug] Loading models...")
    det = SCRFD(model_file=SCRFD_PATH)
    det.prepare(-1)
    rec = IRES(ARCFACE_PATH)

    try:
        print("[debug] Detect + align A...")
        a_normal, a_vis = _detect_align_best(det, image_a, "image A")
        print("[debug] Detect + align B...")
        b_aligned, b_vis = _detect_align_best(det, image_b, "image B")
    except RuntimeError as err:
        print(f"[ERROR] {err}")
        return 1

    a_mask = a_normal.copy()
    a_mask[65:, 65:, :] = 0

    # Save debug images
    path_a_normal = os.path.join(args.out_dir, "A_aligned_normal.jpg")
    path_a_mask = os.path.join(args.out_dir, "A_aligned_mask.jpg")
    path_b_aligned = os.path.join(args.out_dir, "B_aligned.jpg")
    path_a_bbox = os.path.join(args.out_dir, "A_input_with_bbox.jpg")
    path_b_bbox = os.path.join(args.out_dir, "B_input_with_bbox.jpg")
    cv2.imwrite(path_a_normal, a_normal)
    cv2.imwrite(path_a_mask, a_mask)
    cv2.imwrite(path_b_aligned, b_aligned)
    cv2.imwrite(path_a_bbox, a_vis)
    cv2.imwrite(path_b_bbox, b_vis)

    feat_a_normal = _normalize(rec.predict(a_normal.copy()).flatten())
    feat_a_mask = _normalize(rec.predict(a_mask.copy()).flatten())
    feat_b = _normalize(rec.predict(b_aligned.copy()).flatten())

    cos_ab_normal, l2_ab_normal = _cosine_l2(feat_a_normal, feat_b)
    cos_ab_mask, l2_ab_mask = _cosine_l2(feat_a_mask, feat_b)
    cos_a_normal_mask, l2_a_normal_mask = _cosine_l2(feat_a_normal, feat_a_mask)

    print("[debug] Saved files:")
    print(f"  - {path_a_normal}")
    print(f"  - {path_a_mask}")
    print(f"  - {path_b_aligned}")
    print(f"  - {path_a_bbox}")
    print(f"  - {path_b_bbox}")

    print("[debug] Embedding comparison:")
    print(f"  - A_normal vs B: cosine={cos_ab_normal:.6f}, l2={l2_ab_normal:.6f}")
    print(f"  - A_mask   vs B: cosine={cos_ab_mask:.6f}, l2={l2_ab_mask:.6f}")
    print(
        f"  - A_normal vs A_mask: cosine={cos_a_normal_mask:.6f}, l2={l2_a_normal_mask:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
