"""
Face pipeline: SCRFD detection + ArcFace recognition
Usage:
    # SCRFD only (detect + save crops)
    python3 apps/face/run_face_pipeline.py scrfd \
        --engine data/face/models/scrfd/scrfd.engine \
        --image data/face/test.jpg \
        --save-crops

    # ArcFace only (pre-cropped face)
    python3 apps/face/run_face_pipeline.py arcface \
        --engine data/face/models/arcface/arcface_r100.engine \
        --image data/face/face_crop.jpg

    # Full pipeline on single image
    python3 apps/face/run_face_pipeline.py pipeline \
        --det-engine data/face/models/scrfd/scrfd.engine \
        --rec-engine data/face/models/arcface/arcface_r100.engine \
        --image data/face/test.jpg \
        --save-crops

    # Full pipeline: compare two images (cosine similarity)
    python3 apps/face/run_face_pipeline.py compare \
        --det-engine data/face/models/scrfd/scrfd.engine \
        --rec-engine data/face/models/arcface/arcface_r100.engine \
        --image1 data/face/person_a.jpg \
        --image2 data/face/person_b.jpg
"""

import argparse
import os
import sys
import cv2
import numpy as np

# Add apps/face to path for engine imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scrfd_engine import SCRFDTRT
from arcface_engine import IRES


# ─────────────────────────────────────────────────────────
# Face Alignment (5-point landmarks → 112x112)
# ─────────────────────────────────────────────────────────

ARCFACE_REF_POINTS = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose tip
    [41.5493, 92.3655],  # left mouth corner
    [70.7299, 92.2041],  # right mouth corner
], dtype=np.float32)


def estimate_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate 2x3 affine transform from src to dst (5 points)."""
    num = src.shape[0]
    A = np.zeros((num * 2, 4), dtype=np.float64)
    b = np.zeros((num * 2,), dtype=np.float64)
    for i in range(num):
        A[2 * i]     = [src[i, 0], -src[i, 1], 1, 0]
        A[2 * i + 1] = [src[i, 1],  src[i, 0], 0, 1]
        b[2 * i]     = dst[i, 0]
        b[2 * i + 1] = dst[i, 1]
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b_val, tx, ty = params
    return np.array([[a, -b_val, tx], [b_val, a, ty]], dtype=np.float64)


def align_face(img: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    """
    Align face to ArcFace standard 112x112 using 5 landmarks.
    Args:
        img: BGR full image
        kps: (5, 2) landmarks from SCRFD
        size: output size (default 112)
    Returns:
        aligned: (112, 112, 3) BGR
    """
    M = estimate_affine(kps.astype(np.float32), ARCFACE_REF_POINTS)
    return cv2.warpAffine(img, M, (size, size), borderValue=0)


# ─────────────────────────────────────────────────────────
# Cosine Similarity
# ─────────────────────────────────────────────────────────

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized embeddings."""
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    n1 = np.linalg.norm(emb1)
    n2 = np.linalg.norm(emb2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(emb1 / n1, emb2 / n2))


# ─────────────────────────────────────────────────────────
# Draw Detections
# ─────────────────────────────────────────────────────────

def draw_detections(img: np.ndarray, bboxes: np.ndarray,
                    kpss: np.ndarray = None) -> np.ndarray:
    """Draw bboxes and landmarks on image."""
    vis = img.copy()
    KPS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
                  (255, 255, 0), (0, 255, 255)]
    for i in range(len(bboxes)):
        x1, y1, x2, y2, score = bboxes[i]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{score:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if kpss is not None and i < len(kpss):
            for j, (px, py) in enumerate(kpss[i]):
                cv2.circle(vis, (int(px), int(py)), 3, KPS_COLORS[j], -1)
    return vis


# ─────────────────────────────────────────────────────────
# Mode: SCRFD only
# ─────────────────────────────────────────────────────────

def run_scrfd(args):
    """Detect faces, optionally save crop images."""
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {args.image}")
    print(f"[INFO] Image: {args.image} ({img.shape[1]}x{img.shape[0]})")

    detector = SCRFDTRT(engine_path=args.engine)

    bboxes, kpss = detector.predict(img)
    print(f"[INFO] Detected {len(bboxes)} face(s)")

    for i, bbox in enumerate(bboxes):
        print(f"  Face {i}: bbox=[{bbox[0]:.0f},{bbox[1]:.0f},"
              f"{bbox[2]:.0f},{bbox[3]:.0f}] score={bbox[4]:.3f}")

    # Save crops
    if args.save_crops:
        crop_dir = args.crop_dir or os.path.splitext(args.image)[0] + "_crops"
        os.makedirs(crop_dir, exist_ok=True)
        for i, (bbox, kps) in enumerate(zip(bboxes, kpss)):
            # Aligned crop using landmarks
            aligned = align_face(img, kps)
            crop_path = os.path.join(crop_dir, f"face_{i:03d}_aligned.jpg")
            cv2.imwrite(crop_path, aligned)
            print(f"  [CROP] Saved aligned: {crop_path}")

            # Also save raw bbox crop
            x1, y1, x2, y2 = (max(0, int(bbox[0])), max(0, int(bbox[1])),
                               min(img.shape[1], int(bbox[2])),
                               min(img.shape[0], int(bbox[3])))
            raw_crop = img[y1:y2, x1:x2]
            raw_path = os.path.join(crop_dir, f"face_{i:03d}_raw.jpg")
            cv2.imwrite(raw_path, raw_crop)
            print(f"  [CROP] Saved raw:     {raw_path}")
        print(f"[INFO] Crops saved to: {crop_dir}")

    # Save visualization
    vis = draw_detections(img, bboxes, kpss)
    out_path = args.output or "output_scrfd.jpg"
    cv2.imwrite(out_path, vis)
    print(f"[INFO] Saved detection result: {out_path}")

    detector.destroy()
    return bboxes, kpss


# ─────────────────────────────────────────────────────────
# Mode: ArcFace only
# ─────────────────────────────────────────────────────────

def run_arcface(args):
    """Extract embedding from a pre-cropped face image."""
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {args.image}")
    print(f"[INFO] Image: {args.image} ({img.shape[1]}x{img.shape[0]})")

    recognizer = IRES(onnx_path=args.engine)
    embedding = recognizer.predict(img)
    embedding = embedding.flatten()

    print(f"[INFO] Embedding shape : {embedding.shape}")
    print(f"[INFO] Embedding norm  : {np.linalg.norm(embedding):.4f}")
    print(f"[INFO] Embedding[:8]   : {embedding[:8]}")

    if args.save_embedding:
        np.save(args.save_embedding, embedding)
        print(f"[INFO] Saved embedding: {args.save_embedding}")

    recognizer.destroy()
    return embedding


# ─────────────────────────────────────────────────────────
# Mode: Full pipeline (detect → align → recognize)
# ─────────────────────────────────────────────────────────

def get_embeddings_from_image(img: np.ndarray, detector: SCRFDTRT,
                               recognizer: IRES,
                               save_crops: bool = False,
                               crop_dir: str = None,
                               image_name: str = "img") -> list:
    """
    Full pipeline on one image.
    Returns list of dicts: {bbox, kps, aligned, embedding}
    """
    bboxes, kpss = detector.predict(img)
    print(f"[INFO] [{image_name}] Detected {len(bboxes)} face(s)")

    results = []
    for i, (bbox, kps) in enumerate(zip(bboxes, kpss)):
        aligned = align_face(img, kps)
        embedding = recognizer.predict(aligned).flatten()

        print(f"  Face {i}: bbox=[{bbox[0]:.0f},{bbox[1]:.0f},"
              f"{bbox[2]:.0f},{bbox[3]:.0f}] score={bbox[4]:.3f} "
              f"emb_norm={np.linalg.norm(embedding):.4f}")

        if save_crops and crop_dir:
            os.makedirs(crop_dir, exist_ok=True)
            aligned_path = os.path.join(crop_dir, f"{image_name}_face{i:03d}_aligned.jpg")
            cv2.imwrite(aligned_path, aligned)
            print(f"  [CROP] Saved: {aligned_path}")

        results.append({
            "bbox": bbox,
            "kps": kps,
            "aligned": aligned,
            "embedding": embedding,
        })

    return results


def run_pipeline(args):
    """Full pipeline: detect → align → recognize on one image."""
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {args.image}")
    print(f"[INFO] Image: {args.image} ({img.shape[1]}x{img.shape[0]})")

    detector   = SCRFDTRT(engine_path=args.det_engine)
    recognizer = IRES(onnx_path=args.rec_engine)

    crop_dir = args.crop_dir if args.save_crops else None
    results = get_embeddings_from_image(
        img, detector, recognizer,
        save_crops=args.save_crops,
        crop_dir=crop_dir or "output_crops",
        image_name=os.path.splitext(os.path.basename(args.image))[0],
    )

    # Visualization
    bboxes = np.array([r["bbox"] for r in results])
    kpss   = np.array([r["kps"]  for r in results]) if results else None
    vis = draw_detections(img, bboxes, kpss)
    out_path = args.output or "output_pipeline.jpg"
    cv2.imwrite(out_path, vis)
    print(f"[INFO] Saved result: {out_path}")

    # Save embeddings
    if args.save_embedding:
        embeddings = np.array([r["embedding"] for r in results])
        np.save(args.save_embedding, embeddings)
        print(f"[INFO] Saved {len(embeddings)} embeddings: {args.save_embedding}")

    detector.destroy()
    return results


# ─────────────────────────────────────────────────────────
# Mode: Compare two images
# ─────────────────────────────────────────────────────────

def run_compare(args):
    """Compare two images: detect best face, extract embedding, compute cosine."""
    img1 = cv2.imread(args.image1)
    img2 = cv2.imread(args.image2)
    if img1 is None:
        raise FileNotFoundError(f"Cannot read: {args.image1}")
    if img2 is None:
        raise FileNotFoundError(f"Cannot read: {args.image2}")

    detector   = SCRFDTRT(engine_path=args.det_engine)
    recognizer = IRES(onnx_path=args.rec_engine)

    name1 = os.path.splitext(os.path.basename(args.image1))[0]
    name2 = os.path.splitext(os.path.basename(args.image2))[0]

    crop_dir = args.crop_dir if args.save_crops else None

    res1 = get_embeddings_from_image(img1, detector, recognizer,
                                      save_crops=args.save_crops,
                                      crop_dir=crop_dir or "output_crops",
                                      image_name=name1)
    res2 = get_embeddings_from_image(img2, detector, recognizer,
                                      save_crops=args.save_crops,
                                      crop_dir=crop_dir or "output_crops",
                                      image_name=name2)

    if not res1:
        print(f"[WARN] No face detected in: {args.image1}")
        detector.destroy()
        return
    if not res2:
        print(f"[WARN] No face detected in: {args.image2}")
        detector.destroy()
        return

    # Use highest-score face from each image
    best1 = max(res1, key=lambda r: r["bbox"][4])
    best2 = max(res2, key=lambda r: r["bbox"][4])

    sim  = cosine_similarity(best1["embedding"], best2["embedding"])
    dist = float(np.linalg.norm(best1["embedding"] - best2["embedding"]))
    threshold = args.threshold

    print("\n" + "=" * 60)
    print("  COMPARISON RESULT")
    print(f"  Image 1           : {args.image1}")
    print(f"  Image 2           : {args.image2}")
    print(f"  Cosine Similarity : {sim:.4f}")
    print(f"  L2 Distance       : {dist:.4f}")
    print(f"  Threshold         : {threshold}")
    print(f"  Same Person?      : {'YES ✓' if sim >= threshold else 'NO  ✗'}")
    print("=" * 60)

    # Build comparison visualization
    h = 112
    gap = 10
    canvas = np.full((h, h * 2 + gap, 3), 200, dtype=np.uint8)
    canvas[:, :h]       = best1["aligned"]
    canvas[:, h + gap:] = best2["aligned"]
    color = (0, 180, 0) if sim >= threshold else (0, 0, 200)
    label = f"sim={sim:.3f}  {'SAME' if sim >= threshold else 'DIFF'}"
    cv2.putText(canvas, label, (5, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    out_path = args.output or "output_compare.jpg"
    cv2.imwrite(out_path, canvas)
    print(f"[INFO] Saved comparison: {out_path}")

    detector.destroy()
    return sim


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Face Pipeline: SCRFD + ArcFace")
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── scrfd subcommand ──
    p_scrfd = sub.add_parser("scrfd", help="SCRFD detection only")
    p_scrfd.add_argument("--engine",      required=True, help="SCRFD TRT engine path")
    p_scrfd.add_argument("--image",       required=True, help="Input image path")
    p_scrfd.add_argument("--output",      default=None,  help="Output visualization path")
    p_scrfd.add_argument("--save-crops",  action="store_true", help="Save cropped faces")
    p_scrfd.add_argument("--crop-dir",    default=None,  help="Directory to save crops")

    # ── arcface subcommand ──
    p_arc = sub.add_parser("arcface", help="ArcFace embedding only (pre-cropped)")
    p_arc.add_argument("--engine",         required=True, help="ArcFace ONNX model path")
    p_arc.add_argument("--image",          required=True, help="Input face crop image path")
    p_arc.add_argument("--save-embedding", default=None,  help="Save embedding .npy path")

    # ── pipeline subcommand ──
    p_pipe = sub.add_parser("pipeline", help="Full pipeline: detect → align → recognize")
    p_pipe.add_argument("--det-engine",    required=True, help="SCRFD TRT engine path")
    p_pipe.add_argument("--rec-engine",    required=True, help="ArcFace ONNX model path")
    p_pipe.add_argument("--image",         required=True, help="Input image path")
    p_pipe.add_argument("--output",        default=None,  help="Output visualization path")
    p_pipe.add_argument("--save-crops",    action="store_true", help="Save aligned crops")
    p_pipe.add_argument("--crop-dir",      default=None,  help="Directory to save crops")
    p_pipe.add_argument("--save-embedding",default=None,  help="Save embeddings .npy path")

    # ── compare subcommand ──
    p_cmp = sub.add_parser("compare", help="Compare two images (cosine similarity)")
    p_cmp.add_argument("--det-engine",  required=True, help="SCRFD TRT engine path")
    p_cmp.add_argument("--rec-engine",  required=True, help="ArcFace ONNX model path")
    p_cmp.add_argument("--image1",      required=True, help="First image path")
    p_cmp.add_argument("--image2",      required=True, help="Second image path")
    p_cmp.add_argument("--output",      default=None,  help="Output comparison image path")
    p_cmp.add_argument("--threshold",   type=float, default=0.4, help="Cosine threshold (default: 0.4)")
    p_cmp.add_argument("--save-crops",  action="store_true", help="Save aligned crops")
    p_cmp.add_argument("--crop-dir",    default=None, help="Directory to save crops")

    args = parser.parse_args()

    if args.mode == "scrfd":
        run_scrfd(args)
    elif args.mode == "arcface":
        run_arcface(args)
    elif args.mode == "pipeline":
        run_pipeline(args)
    elif args.mode == "compare":
        run_compare(args)


if __name__ == "__main__":
    main()