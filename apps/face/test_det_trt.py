"""
SCRFD Face Detection inference with TensorRT engine.
Usage:
    python3 apps/face/test_det_trt.py --engine <path_to_engine> --image <path_to_image>
"""

import argparse
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from itertools import product
from typing import List, Tuple

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# TensorRT version compatibility
TRT_VERSION = tuple(int(x) for x in trt.__version__.split(".")[:2])
TRT_NEW_API = TRT_VERSION >= (10, 0)


class SCRFDTensorRT:
    """SCRFD Face Detector using TensorRT engine."""

    # SCRFD typical strides and feature map config
    _FEAT_STRIDE_FPN = [8, 16, 32]
    _NUM_ANCHORS = 2  # SCRFD uses 2 anchors per location

    def __init__(self, engine_path: str, input_size: Tuple[int, int] = (640, 640),
                 conf_thresh: float = 0.5, nms_thresh: float = 0.4):
        self.input_size = input_size  # (width, height)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        # Load TRT engine
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

        # Print binding info
        self._print_bindings()

    def _load_engine(self, engine_path: str):
        """Load serialized TensorRT engine."""
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")
        print(f"[INFO] Loaded TRT engine: {engine_path}")
        print(f"[INFO] TensorRT version: {trt.__version__} (new API: {TRT_NEW_API})")
        return engine

    def _get_num_bindings(self):
        """Get number of I/O tensors (compatible with TRT 8 and 10)."""
        if TRT_NEW_API:
            return self.engine.num_io_tensors
        else:
            return self.engine.num_bindings

    def _get_binding_name(self, i):
        if TRT_NEW_API:
            return self.engine.get_tensor_name(i)
        else:
            return self.engine.get_binding_name(i)

    def _get_binding_dtype(self, i):
        if TRT_NEW_API:
            name = self.engine.get_tensor_name(i)
            return trt.nptype(self.engine.get_tensor_dtype(name))
        else:
            return trt.nptype(self.engine.get_binding_dtype(i))

    def _get_binding_shape(self, i):
        if TRT_NEW_API:
            name = self.engine.get_tensor_name(i)
            return tuple(self.engine.get_tensor_shape(name))
        else:
            return tuple(self.engine.get_binding_shape(i))

    def _is_input(self, i):
        if TRT_NEW_API:
            name = self.engine.get_tensor_name(i)
            return self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        else:
            return self.engine.binding_is_input(i)

    def _allocate_buffers(self):
        """Allocate host and device buffers for all bindings."""
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        num_bindings = self._get_num_bindings()

        for i in range(num_bindings):
            name = self._get_binding_name(i)
            dtype = self._get_binding_dtype(i)
            shape = self._get_binding_shape(i)

            # Replace dynamic dims (-1) with actual input size
            shape = tuple(
                s if s > 0 else (1 if idx == 0 else self.input_size[1] if idx == 2
                                 else self.input_size[0] if idx == 3 else 3)
                for idx, s in enumerate(shape)
            )

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
            if self._is_input(i):
                inputs.append(buf)
            else:
                outputs.append(buf)

        return inputs, outputs, bindings, stream

    def _print_bindings(self):
        """Print engine binding info for debugging."""
        print("\n[INFO] Engine Bindings:")
        print(f"  {'Name':<30} {'Shape':<25} {'Dtype':<10} {'I/O'}")
        print("  " + "-" * 75)
        num_bindings = self._get_num_bindings()
        for i in range(num_bindings):
            name = self._get_binding_name(i)
            shape = self._get_binding_shape(i)
            io = "INPUT" if self._is_input(i) else "OUTPUT"
            if TRT_NEW_API:
                dtype = self.engine.get_tensor_dtype(name)
            else:
                dtype = self.engine.get_binding_dtype(i)
            print(f"  {name:<30} {str(shape):<25} {str(dtype):<10} {io}")
        print()

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image: resize with letterbox padding, normalize.
        Returns: (blob, scale, padding)
        """
        ih, iw = img.shape[:2]
        w, h = self.input_size

        # Calculate scale (keep aspect ratio)
        scale = min(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        pad_w, pad_h = (w - nw) // 2, (h - nh) // 2

        # Resize and pad
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        padded = np.full((h, w, 3), 0, dtype=np.uint8)
        padded[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized

        # HWC -> CHW, BGR->RGB, float32, normalize to [0, 1] or as required
        blob = padded.astype(np.float32)
        # SCRFD typically uses mean subtraction = 127.5, scale = 1/128
        blob = (blob - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)  # CHW
        blob = np.expand_dims(blob, axis=0)  # NCHW
        blob = np.ascontiguousarray(blob)

        return blob, scale, (pad_w, pad_h)

    def infer(self, blob: np.ndarray) -> List[np.ndarray]:
        """Run TRT inference."""
        # Copy input to host buffer
        np.copyto(self.inputs[0]["host"], blob.ravel())

        if TRT_NEW_API:
            # TensorRT 10+ API
            input_name = self.inputs[0]["name"]
            self.context.set_input_shape(input_name, blob.shape)

            # Set tensor addresses
            for inp in self.inputs:
                cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)
                self.context.set_tensor_address(inp["name"], int(inp["device"]))
            for out in self.outputs:
                self.context.set_tensor_address(out["name"], int(out["device"]))

            # Execute
            self.context.execute_async_v3(stream_handle=self.stream.handle)

            # D2H
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        else:
            # TensorRT 8 API
            self.context.set_binding_shape(0, blob.shape)

            for inp in self.inputs:
                cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)

            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

            for out in self.outputs:
                cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

        self.stream.synchronize()

        # Collect outputs
        results = []
        for out in self.outputs:
            if TRT_NEW_API:
                shape = tuple(self.context.get_tensor_shape(out["name"]))
            else:
                shape = self.context.get_binding_shape(
                    self.engine.get_binding_index(out["name"])
                )
            results.append(out["host"].reshape(shape))

        return results

    # ...existing code...

    def _distance2bbox(self, points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """Convert distance predictions to bounding boxes."""
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    def _distance2kps(self, points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """Convert distance predictions to keypoints (5 landmarks)."""
        kps = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, 0] + distance[:, i]
            py = points[:, 1] + distance[:, i + 1]
            kps.extend([px, py])
        return np.stack(kps, axis=-1)

    def _nms(self, dets: np.ndarray, thresh: float) -> List[int]:
        """Non-maximum suppression."""
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= thresh)[0]
            order = order[inds + 1]
        return keep

    def postprocess(self, outputs: List[np.ndarray], scale: float,
                    pad: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode SCRFD outputs into bboxes and keypoints.
        Supports both stride-interleaved and type-grouped output orderings
        by matching outputs to strides via name lookup or shape matching.
        """
        pad_w, pad_h = pad

        # Build name->array map from self.outputs metadata
        output_map = {self.outputs[i]["name"]: outputs[i] for i in range(len(outputs))}

        # Determine if keypoints are available
        has_kps = any("kps" in name for name in output_map)

        scores_list, bboxes_list, kpss_list = [], [], []

        for idx, stride in enumerate(self._FEAT_STRIDE_FPN):
            # Try name-based lookup first (robust against output ordering)
            score_out = output_map.get(f"score_{stride}")
            bbox_out = output_map.get(f"bbox_{stride}")
            kps_out = output_map.get(f"kps_{stride}") if has_kps else None

            if score_out is None or bbox_out is None:
                # Fallback: assume stride-interleaved order
                outputs_per_stride = 3 if has_kps else 2
                score_out = outputs[idx * outputs_per_stride]
                bbox_out = outputs[idx * outputs_per_stride + 1]
                if has_kps:
                    kps_out = outputs[idx * outputs_per_stride + 2]

            # Remove batch dimension (take first batch item)
            score_out = score_out[0] if score_out.ndim == 3 else score_out
            bbox_out = bbox_out[0] if bbox_out.ndim == 3 else bbox_out
            if kps_out is not None:
                kps_out = kps_out[0] if kps_out.ndim == 3 else kps_out

            scores = score_out.reshape(-1)
            bbox_preds = bbox_out.reshape(-1, 4)
            if has_kps and kps_out is not None:
                kps_preds = kps_out.reshape(-1, 10)

            h = self.input_size[1] // stride
            w = self.input_size[0] // stride
            anchor_centers = np.array(
                [[x * stride, y * stride] for y, x in product(range(h), range(w))],
                dtype=np.float32,
            )
            if self._NUM_ANCHORS > 1:
                anchor_centers = np.repeat(anchor_centers, self._NUM_ANCHORS, axis=0)

            pos_inds = np.where(scores >= self.conf_thresh)[0]
            if len(pos_inds) == 0:
                continue

            scores = scores[pos_inds]
            bbox_preds = bbox_preds[pos_inds] * stride
            anchor_pts = anchor_centers[pos_inds]

            bboxes = self._distance2bbox(anchor_pts, bbox_preds)
            scores_list.append(scores)
            bboxes_list.append(bboxes)

            if has_kps and kps_out is not None:
                kps_preds = kps_preds[pos_inds] * stride
                kpss = self._distance2kps(anchor_pts, kps_preds)
                kpss_list.append(kpss)

        if len(scores_list) == 0:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)

        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list)
        if has_kps and kpss_list:
            kpss = np.concatenate(kpss_list)
        else:
            kpss = np.empty((len(scores), 10), dtype=np.float32)

        dets = np.hstack([bboxes, scores[:, None]])
        keep = self._nms(dets, self.nms_thresh)
        dets = dets[keep]
        kpss = kpss[keep]

        dets[:, 0] = (dets[:, 0] - pad_w) / scale
        dets[:, 1] = (dets[:, 1] - pad_h) / scale
        dets[:, 2] = (dets[:, 2] - pad_w) / scale
        dets[:, 3] = (dets[:, 3] - pad_h) / scale

        kpss = kpss.reshape(-1, 5, 2)
        kpss[:, :, 0] = (kpss[:, :, 0] - pad_w) / scale
        kpss[:, :, 1] = (kpss[:, :, 1] - pad_h) / scale

        return dets, kpss

    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Full detection pipeline: preprocess -> infer -> postprocess."""
        blob, scale, pad = self.preprocess(img)
        outputs = self.infer(blob)

        print("[DEBUG] Output shapes:")
        for i, o in enumerate(outputs):
            name = self.outputs[i]["name"] if i < len(self.outputs) else f"out_{i}"
            print(f"  [{i}] {name}: {o.shape}")

        bboxes, kpss = self.postprocess(outputs, scale, pad)
        return bboxes, kpss


def draw_detections(img: np.ndarray, bboxes: np.ndarray, kpss: np.ndarray) -> np.ndarray:
    """Draw bounding boxes and landmarks on image."""
    vis = img.copy()
    for i in range(len(bboxes)):
        x1, y1, x2, y2, score = bboxes[i]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{score:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if kpss is not None and len(kpss) > i:
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
                      (255, 255, 0), (0, 255, 255)]
            for j in range(5):
                px, py = int(kpss[i][j][0]), int(kpss[i][j][1])
                cv2.circle(vis, (px, py), 3, colors[j], -1)

    return vis


def main():
    parser = argparse.ArgumentParser(description="SCRFD TensorRT Inference")
    parser.add_argument("--engine", type=str, required=True,
                        help="Path to SCRFD TensorRT engine file")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--output", type=str, default="output_det.jpg",
                        help="Path to output image (default: output_det.jpg)")
    parser.add_argument("--input-size", type=int, nargs=2, default=[640, 640],
                        help="Model input size: width height (default: 640 640)")
    parser.add_argument("--conf-thresh", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--nms-thresh", type=float, default=0.4,
                        help="NMS IoU threshold (default: 0.4)")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    print(f"[INFO] Image: {args.image} ({img.shape[1]}x{img.shape[0]})")

    detector = SCRFDTensorRT(
        engine_path=args.engine,
        input_size=tuple(args.input_size),
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
    )

    bboxes, kpss = detector.detect(img)
    print(f"[INFO] Detected {len(bboxes)} face(s)")

    for i, (bbox, kps) in enumerate(zip(bboxes, kpss)):
        print(f"  Face {i}: bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}] "
              f"score={bbox[4]:.3f}")

    vis = draw_detections(img, bboxes, kpss)
    cv2.imwrite(args.output, vis)
    print(f"[INFO] Saved result to: {args.output}")


if __name__ == "__main__":
    main()