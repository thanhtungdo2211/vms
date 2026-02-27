"""
ArcFace Feature Extraction inference with TensorRT engine.
Takes a face image (or full image with SCRFD detection), extracts 512-d embedding.

Usage:
    # Extract embedding from a pre-cropped face image (112x112)
    python3 apps/face/test_arcface_trt.py \
        --engine data/face/models/arcface/arcface_r100.onnx_b16_gpu0_fp16.engine \
        --image data/face/test_face_crop.jpg

    # Full pipeline: detect face with SCRFD then extract embedding
    python3 apps/face/test_arcface_trt.py \
        --engine data/face/models/arcface/arcface_r100.onnx_b16_gpu0_fp16.engine \
        --det-engine data/face/models/scrfd640/scrfd_2.5g_bnkps_dynamic.onnx_b16_gpu0_fp16.engine \
        --image data/face/test.jpg

    # Compare two face images
    python3 apps/face/test_arcface_trt.py \
        --engine data/face/models/arcface/arcface_r100.onnx_b16_gpu0_fp16.engine \
        --det-engine data/face/models/scrfd640/scrfd_2.5g_bnkps_dynamic.onnx_b16_gpu0_fp16.engine \
        --image data/face/person_a.jpg \
        --image2 data/face/person_b.jpg
"""

import argparse
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import List, Tuple, Optional

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ─────────────────────────────────────────────────────────
# Face Alignment (using 5 landmarks from SCRFD)
# ─────────────────────────────────────────────────────────

# ArcFace standard alignment reference points (for 112x112)
ARCFACE_REF_POINTS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


def estimate_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Estimate 2x3 affine transform matrix from src points to dst points.
    Uses least squares approach similar to skimage.transform.SimilarityTransform.
    """
    num = src.shape[0]
    # Build equation system: [x, y, 1, 0; y, -x, 0, 1] * [a, b, tx, ty]^T = [dx, dy]
    A = np.zeros((num * 2, 4), dtype=np.float64)
    b = np.zeros((num * 2,), dtype=np.float64)

    for i in range(num):
        A[2 * i] = [src[i, 0], -src[i, 1], 1, 0]
        A[2 * i + 1] = [src[i, 1], src[i, 0], 0, 1]
        b[2 * i] = dst[i, 0]
        b[2 * i + 1] = dst[i, 1]

    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b_val, tx, ty = params

    M = np.array([
        [a, -b_val, tx],
        [b_val, a, ty],
    ], dtype=np.float64)
    return M


def align_face(img: np.ndarray, landmarks: np.ndarray,
               image_size: int = 112) -> np.ndarray:
    """
    Align face using 5 landmarks to standard ArcFace reference.
    Args:
        img: BGR image
        landmarks: (5, 2) facial landmarks from SCRFD
        image_size: output size (default 112)
    Returns:
        Aligned face image (112x112x3)
    """
    dst = ARCFACE_REF_POINTS.copy()
    M = estimate_affine(landmarks.astype(np.float32), dst)
    aligned = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0)
    return aligned


# ─────────────────────────────────────────────────────────
# ArcFace TensorRT Inference
# ─────────────────────────────────────────────────────────

class ArcFaceTensorRT:
    """ArcFace Feature Extractor using TensorRT engine."""

    def __init__(self, engine_path: str, input_size: int = 112):
        self.input_size = input_size
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        self._print_bindings()

    def _load_engine(self, engine_path: str):
        """Load serialized TensorRT engine."""
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")
        print(f"[INFO] Loaded ArcFace TRT engine: {engine_path}")
        return engine

    def _allocate_buffers(self):
        """Allocate host and device buffers for all bindings."""
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)

            # Replace dynamic dims (-1) with actual values
            resolved = []
            for idx, s in enumerate(shape):
                if s > 0:
                    resolved.append(s)
                elif idx == 0:
                    resolved.append(1)  # batch_size = 1
                elif idx == 1:
                    resolved.append(3)  # channels
                else:
                    resolved.append(self.input_size)
            shape = tuple(resolved)

            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            buf = {
                "name": name,
                "host": host_mem,
                "device": device_mem,
                "shape": shape,
                "dtype": dtype,
            }
            if self.engine.binding_is_input(i):
                inputs.append(buf)
            else:
                outputs.append(buf)

        return inputs, outputs, bindings, stream

    def _print_bindings(self):
        """Print engine binding info for debugging."""
        print("\n[INFO] ArcFace Engine Bindings:")
        print(f"  {'Name':<30} {'Shape':<25} {'Dtype':<10} {'I/O'}")
        print("  " + "-" * 75)
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = self.engine.get_binding_dtype(i)
            io = "INPUT" if self.engine.binding_is_input(i) else "OUTPUT"
            print(f"  {name:<30} {str(tuple(shape)):<25} {str(dtype):<10} {io}")
        print()

    def preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocess aligned face image for ArcFace.
        Matches DeepStream nvinfer config:
            offsets=127.5;127.5;127.5
            net-scale-factor=0.00784313725 (= 1/127.5)
        So: pixel = (pixel - 127.5) * 0.00784313725 = (pixel - 127.5) / 127.5

        Args:
            face_img: BGR image, should be 112x112
        Returns:
            blob: (1, 3, 112, 112) float32
        """
        if face_img.shape[:2] != (self.input_size, self.input_size):
            face_img = cv2.resize(face_img, (self.input_size, self.input_size),
                                  interpolation=cv2.INTER_LINEAR)

        blob = face_img.astype(np.float32)
        # Match nvinfer: (pixel - offset) * net-scale-factor
        # offsets=127.5, net-scale-factor=1/127.5
        blob = (blob - 127.5) / 127.5  # normalize to [-1, 1]
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)  # NCHW
        blob = np.ascontiguousarray(blob)
        return blob

    def infer(self, blob: np.ndarray) -> np.ndarray:
        """
        Run TRT inference.
        Args:
            blob: (1, 3, 112, 112) float32
        Returns:
            embedding: (512,) float32 (L2-normalized)
        """
        np.copyto(self.inputs[0]["host"], blob.ravel())
        self.context.set_binding_shape(0, blob.shape)

        # H2D
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)

        # Execute
        self.context.execute_async_v2(bindings=self.bindings,
                                      stream_handle=self.stream.handle)

        # D2H
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

        self.stream.synchronize()

        # Get output
        results = []
        for out in self.outputs:
            shape = self.context.get_binding_shape(
                self.engine.get_binding_index(out["name"])
            )
            results.append(out["host"].reshape(shape))

        # Debug
        print("[DEBUG] ArcFace output shapes:")
        for i, r in enumerate(results):
            name = self.outputs[i]["name"] if i < len(self.outputs) else f"out_{i}"
            print(f"  [{i}] {name}: {r.shape}")

        # Usually output is (1, 512) or (1, 512, 1, 1)
        embedding = results[0].flatten()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def get_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """
        Full pipeline: preprocess -> infer -> L2-normalized embedding.
        Args:
            face_img: BGR face image (112x112 or will be resized)
        Returns:
            embedding: (512,) float32, L2-normalized
        """
        blob = self.preprocess(face_img)
        embedding = self.infer(blob)
        return embedding


# ─────────────────────────────────────────────────────────
# Utility: cosine similarity
# ─────────────────────────────────────────────────────────

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized embeddings."""
    return float(np.dot(emb1, emb2))


def l2_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute L2 distance between two embeddings."""
    return float(np.linalg.norm(emb1 - emb2))


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ArcFace TensorRT Feature Extraction")
    parser.add_argument("--engine", type=str, required=True,
                        help="Path to ArcFace TensorRT engine file")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image (face crop or full image)")
    parser.add_argument("--image2", type=str, default=None,
                        help="Path to second image for comparison")
    parser.add_argument("--det-engine", type=str, default=None,
                        help="Path to SCRFD TensorRT engine for face detection + alignment")
    parser.add_argument("--input-size", type=int, default=112,
                        help="ArcFace input size (default: 112)")
    parser.add_argument("--det-input-size", type=int, nargs=2, default=[640, 640],
                        help="SCRFD input size: width height (default: 640 640)")
    parser.add_argument("--conf-thresh", type=float, default=0.5,
                        help="SCRFD confidence threshold (default: 0.5)")
    parser.add_argument("--output", type=str, default="output_arcface.jpg",
                        help="Path to output visualization (default: output_arcface.jpg)")
    parser.add_argument("--save-embedding", type=str, default=None,
                        help="Save embedding to .npy file")
    args = parser.parse_args()

    # ── Load ArcFace ──
    arcface = ArcFaceTensorRT(
        engine_path=args.engine,
        input_size=args.input_size,
    )

    # ── Optionally load SCRFD detector ──
    detector = None
    if args.det_engine:
        from test_det_trt import SCRFDTensorRT
        detector = SCRFDTensorRT(
            engine_path=args.det_engine,
            input_size=tuple(args.det_input_size),
            conf_thresh=args.conf_thresh,
        )
        print("[INFO] SCRFD detector loaded for face detection + alignment")

    def process_image(image_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Process an image: detect face (if detector available), align, extract embedding.
        Returns: (embedding, aligned_face, vis_image)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        print(f"\n[INFO] Image: {image_path} ({img.shape[1]}x{img.shape[0]})")

        if detector is not None:
            # Detect + align
            bboxes, kpss = detector.detect(img)
            print(f"[INFO] Detected {len(bboxes)} face(s)")

            if len(bboxes) == 0:
                raise RuntimeError(f"No face detected in: {image_path}")

            # Use the face with highest score
            best_idx = np.argmax(bboxes[:, 4])
            bbox = bboxes[best_idx]
            kps = kpss[best_idx]
            print(f"[INFO] Best face: bbox=[{bbox[0]:.0f},{bbox[1]:.0f},"
                  f"{bbox[2]:.0f},{bbox[3]:.0f}] score={bbox[4]:.3f}")

            # Align face using landmarks
            aligned = align_face(img, kps, image_size=args.input_size)

            # Draw detection on image for visualization
            vis = img.copy()
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{bbox[4]:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
                      (255, 255, 0), (0, 255, 255)]
            for j in range(5):
                px, py = int(kps[j][0]), int(kps[j][1])
                cv2.circle(vis, (px, py), 3, colors[j], -1)
        else:
            # No detector: treat input as pre-cropped face
            aligned = cv2.resize(img, (args.input_size, args.input_size))
            vis = None
            print("[INFO] No detector — treating image as pre-cropped face")

        # Extract embedding
        embedding = arcface.get_embedding(aligned)
        print(f"[INFO] Embedding shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.4f}")
        print(f"[INFO] Embedding[:8]: {embedding[:8]}")

        return embedding, aligned, vis

    # ── Process image 1 ──
    emb1, aligned1, vis1 = process_image(args.image)

    # ── Process image 2 (comparison mode) ──
    if args.image2:
        emb2, aligned2, vis2 = process_image(args.image2)

        sim = cosine_similarity(emb1, emb2)
        dist = l2_distance(emb1, emb2)

        print("\n" + "=" * 60)
        print(f"  COMPARISON RESULT")
        print(f"  Cosine Similarity : {sim:.4f}")
        print(f"  L2 Distance       : {dist:.4f}")
        print(f"  Same Person?      : {'YES ✓' if sim > 0.4 else 'NO ✗'} (threshold=0.4)")
        print("=" * 60)

        # Build comparison visualization
        h = args.input_size
        gap = 20
        canvas = np.full((h, h * 2 + gap, 3), 255, dtype=np.uint8)
        canvas[:, :h] = aligned1
        canvas[:, h + gap:] = aligned2

        color = (0, 200, 0) if sim > 0.4 else (0, 0, 200)
        label = f"sim={sim:.3f} {'SAME' if sim > 0.4 else 'DIFF'}"
        cv2.putText(canvas, label, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imwrite(args.output, canvas)
        print(f"[INFO] Saved comparison to: {args.output}")

    else:
        # Single image mode
        # Save aligned face
        aligned_path = args.output.replace(".jpg", "_aligned.jpg")
        cv2.imwrite(aligned_path, aligned1)
        print(f"[INFO] Saved aligned face to: {aligned_path}")

        if vis1 is not None:
            cv2.imwrite(args.output, vis1)
            print(f"[INFO] Saved detection visualization to: {args.output}")

    # ── Save embedding ──
    if args.save_embedding:
        np.save(args.save_embedding, emb1)
        print(f"[INFO] Saved embedding to: {args.save_embedding}")


if __name__ == "__main__":
    main()