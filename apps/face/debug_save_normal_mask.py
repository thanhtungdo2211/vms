"""
Save aligned normal/masked face images and compare embeddings.

Usage:
    python3 apps/face/debug_save_normal_mask.py \
        --image data/face/face-import-pics/tungdt.jpg \
        --out-dir /tmp/face-debug
"""

import argparse
import os

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SCRFD + ArcFace and save normal/mask aligned images."
    )
    parser.add_argument("--image", required=True, help="Input face image path")
    parser.add_argument(
        "--out-dir", default="data/tmp/face-debug", help="Output directory for saved images"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    image = cv2.imread(args.image)
    if image is None:
        print(f"[ERROR] Cannot read image: {args.image}")
        return 1

    print("[debug] Loading models...")
    det = SCRFD(model_file=SCRFD_PATH)
    det.prepare(-1)
    rec = IRES(ARCFACE_PATH)

    print("[debug] Detecting face...")
    bboxes, kpss = det.detect(image, 0.5, input_size=(640, 640))
    if len(bboxes) == 0:
        print("[debug] No face detected.")
        return 1

    best = int(np.argmax([b[4] for b in bboxes]))
    score = float(bboxes[best][4])
    x1, y1, x2, y2 = bboxes[best][:4]
    lm = kpss[best]

    aligned_normal = align_face(image.copy(), [x1, y1, x2, y2], lm)
    aligned_mask = aligned_normal.copy()
    aligned_mask[65:, 65:, :] = 0

    path_normal = os.path.join(args.out_dir, "aligned_normal.jpg")
    path_mask = os.path.join(args.out_dir, "aligned_mask.jpg")
    path_bbox = os.path.join(args.out_dir, "input_with_bbox.jpg")

    vis = image.copy()
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

    cv2.imwrite(path_normal, aligned_normal)
    cv2.imwrite(path_mask, aligned_mask)
    cv2.imwrite(path_bbox, vis)

    feat_normal = _normalize(rec.predict(aligned_normal.copy()).flatten())
    feat_mask = _normalize(rec.predict(aligned_mask).flatten())

    cosine = float(np.dot(feat_normal, feat_mask))
    l2 = float(np.linalg.norm(feat_normal - feat_mask))

    print("[debug] Saved files:")
    print(f"  - {path_normal}")
    print(f"  - {path_mask}")
    print(f"  - {path_bbox}")
    print("[debug] Embedding comparison:")
    print(f"  - cosine(normal, mask): {cosine:.6f}")
    print(f"  - l2(normal, mask): {l2:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
