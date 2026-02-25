# --- START OF FILE face_infer.py ---

import ctypes
from datetime import datetime, timedelta
import json
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import numpy as np
import cv2
from tracking.sort import Sort

import sys
import os
import threading
import queue
import traceback
import io
import math
import uuid
import requests
from sqlalchemy.orm import joinedload
from sqlalchemy import func

# Import search module
sys.path.append('/app')
from face_recognize_ds_search.search_module.search import search, check_and_save_feature, upsert
from face_recognize_ds_search.search_module.qdrant_client_service import QdrantFeatureStorage
from sql.database import SessionLocal
from models.models_sql import *
from qdrant_client.models import Filter, FieldCondition, MatchValue
import gc
# --- CONFIG ---
CONFIG_PATH = "/app/config.json"
try:
    with open(CONFIG_PATH, 'r') as f:
        loaded_config = json.load(f)
        
    # FIX: Kiểm tra nếu config là list (do json bắt đầu bằng []) thì lấy phần tử đầu hoặc chuyển về dict rỗng
    if isinstance(loaded_config, list):
        print("[WARN] Config is a LIST, taking first element or default.")
        if len(loaded_config) > 0 and isinstance(loaded_config[0], dict):
            config = loaded_config[0]
        else:
            config = {}
    elif isinstance(loaded_config, dict):
        config = loaded_config
    else:
        config = {}

except Exception as e:
    print(f"[WARN] Cannot load config: {e}, using defaults")
    config = {}

# Đảm bảo các key quan trọng tồn tại để tránh lỗi về sau
if "url_event" not in config: config["url_event"] = "http://localhost/event"
if "is_front_face" not in config: config["is_front_face"] = 1

# --- EVENT WORKER ---
EVENT_QUEUE = queue.Queue(maxsize=500)

def event_worker():
    while True:
        evt = EVENT_QUEUE.get()
        if evt is None:
            continue
        try:
            _send_event_worker(evt)
        except Exception as e:
            print(f"[EVENT_WORKER] ERROR: {e}")
            # traceback.print_exc() # Debug only
        finally:
            EVENT_QUEUE.task_done()

threading.Thread(target=event_worker, daemon=True).start()

def _send_event_worker(evt):
    try:
        pad_index  = evt["pad_index"]
        person_id  = evt["person_id"]
        timestamp  = evt["timestamp"]
        frame_full = evt["frame_full"]
        face_img   = evt["face_img"]
        clothes_info = evt["clothes_info"]
        face_region = evt["face_region"]
        label = evt.get("label", "?")

        url = config.get("url_event")
        if not url: 
            return

        timestamp_utc7 = timestamp + timedelta(hours=7)
        time_access_str = timestamp_utc7.isoformat()

        data = {
            "person_id": person_id,
            "stream_id": pad_index,
            "uniform_warning": evt.get("uniform_warning", 0),
            "time_access": time_access_str,
        }

        files = {}

        # 1. Encode Face
        if isinstance(face_img, np.ndarray) and face_img.size > 0:
            ok, buf = cv2.imencode(".jpg", face_img)
            if ok:
                files["image_face"] = ("face.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")

        # 2. Encode Full Frame
        full_bytes = None
        if isinstance(frame_full, np.ndarray) and frame_full.size > 0:
            # Vẽ box debug lên ảnh full trước khi gửi (nếu cần)
            # frame_safe = frame_full.copy() # Copy nếu không muốn vẽ lên frame gốc (đã copy ở ngoài rồi)
            fr = face_region or {}
            x1, y1, x2, y2 = int(fr.get("x1", 0)), int(fr.get("y1", 0)), int(fr.get("x2", 0)), int(fr.get("y2", 0))
            h, w = frame_full.shape[:2]
            
            if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
                 cv2.rectangle(frame_full, (x1, y1), (x2, y2), (0,255,0), 2)
            
            try:
                day_folder = datetime.now().strftime("%Y-%m-%d")
                debug_dir = f"./debug_faces/{day_folder}"
                os.makedirs(debug_dir, exist_ok=True)

                filename = f"{datetime.now().strftime('%H-%M-%S')}_{label}.jpg"
                print(f"Recognizer person : {label}")
                cv2.imwrite(os.path.join(debug_dir, filename), frame_full)
            except Exception as e:
                print(f"[EVENT_WORKER] debug save fail: {e}")

            ok2, buf2 = cv2.imencode(".jpg", frame_full)
            if ok2:
                full_bytes = io.BytesIO(buf2.tobytes())
                files["image_full"] = ("image_full.jpg", full_bytes, "image/jpeg")

        # 3. Uniform
        uniform_bytes = None
        if clothes_info and isinstance(clothes_info.get("image_uniform"), np.ndarray):
             uni = clothes_info["image_uniform"]
             if uni.size > 0:
                oku, buf3 = cv2.imencode(".jpg", uni)
                if oku:
                    uniform_bytes = io.BytesIO(buf3.tobytes())

        if uniform_bytes is None:
            # Dummy image to prevent server reject
            dummy = np.zeros((10,10,3), np.uint8)
            _, buf4 = cv2.imencode(".jpg", dummy)
            uniform_bytes = io.BytesIO(buf4.tobytes())

        files["image_uniform"] = ("uniform.jpg", uniform_bytes, "image/jpeg")

        # 4. Send
        r = requests.post(url, data=data, files=files, timeout=5)
        # print(f"[EVENT_WORKER] Sent {person_id} - Status: {r.status_code}")
        r.close()

    except Exception as e:
        print(f"[EVENT_WORKER] Exception sending event: {e}")

class FaceRecognizer:
    def __init__(self, perf_data, clothes_recognizer):
        self.pipeline = None
        self.perf_data = perf_data
        self.clothes_recognizer = clothes_recognizer
        
        # Data Structures
        self.data_face = {}
        self.tracks_his = {}
        self.lock = threading.Lock() # Lock cho tracks_his và data_face

        # Recognition Config
        self.skip_reid = 10
        self.similarity_threshold = 1.15
        
        # Database Caches
        self.features = []
        self.users_id = []
        self.user_id_name_cache = {}
        self.features_matrix = np.empty((0,512), dtype=np.float32)
        self.features_owner_pid = []

        # Logic Control
        self.alert_cooldown_sec = 20
        self.last_alert_at = {}
        self.pending_new = {} 
        self.new_person_cache = {}
        self.flush_interval_sec = 5

        # Thresholds
        self.pg_sim_high  = 0.58
        self.pg_sim_low   = 0.45
        self.pg_sim_floor = 0.40
        self.qdrant_threshold = 0.40
        self.qdrant_low_threshold = 0.40
        self.min_unknown_frames_before_create = 20
        self.recheck_cooldown_sec = 3.0
        self.stable_frames = 15
        self.frame_counter = 0

        self.stopping_streams = set()
        # Start Timers
        GLib.timeout_add_seconds(5, self.perf_data.perf_print_callback)
        GLib.timeout_add_seconds(30, self.refresh_users_id_and_features)
        GLib.timeout_add_seconds(self.flush_interval_sec, self.flush_new_persons)
        
        # Initial Load
        self.refresh_users_id_and_features()

    def mark_stream_stopping(self, pad_index):
        """Đánh dấu stream sắp bị xóa để Probe không chạm vào nữa"""
        with self.lock:
            self.stopping_streams.add(pad_index)

    def clear_stream_data(self, pad_index):
        with self.lock:
            if pad_index in self.tracks_his: del self.tracks_his[pad_index]
            if pad_index in self.data_face: del self.data_face[pad_index]
            if pad_index in self.stopping_streams: self.stopping_streams.remove(pad_index)

    def _ci_to_dict(self, ci):
        if ci is None: return {}
        if isinstance(ci, dict): return ci
        if isinstance(ci, list) and len(ci) > 0:
            if isinstance(ci[0], dict): return ci[0]
            if isinstance(ci[0], (list, tuple)):
                d = {}
                if len(ci[0]) > 0: d['shirt'] = ci[0][0]
                if len(ci[0]) > 1: d['pant'] = ci[0][1]
                return d
        return {}

    def _ci_has_colors(self, ci_dict):
        """Có đủ shirt/pant không?"""
        # --- FIX: Kiểm tra kiểu dữ liệu ---
        if not isinstance(ci_dict, dict):
            return False
        # ----------------------------------
        return bool(ci_dict.get('shirt') or ci_dict.get('pant'))

    def _ci_merge(self, base_dict, extra_dict):
        out = dict(base_dict or {})
        for k in ('shirt', 'pant', 'image_uniform'):
            v = (extra_dict or {}).get(k, None)
            if v is not None:
                out[k] = v
        return out

    # ... (Các hàm _get_most_used_person_id, _pick_person_id_from_qdrant, _seed_qdrant_for_user giữ nguyên) ...
    def _get_most_used_person_id(self, session, candidate_ids, scores=None):
        if not candidate_ids: return None
        try:
            normalized_ids = [pid.strip().lower() for pid in candidate_ids if pid]
            counts = (
                session.query(AccessEvent.person_id, func.count(AccessEvent.id))
                .filter(func.lower(func.trim(AccessEvent.person_id)).in_(normalized_ids))
                .group_by(AccessEvent.person_id)
                .all()
            )
            count_map = {pid.lower(): cnt for pid, cnt in counts}
            
            best_pid = None
            best_weight = -1
            for pid in normalized_ids:
                cnt = count_map.get(pid.lower(), 0)
                score = scores.get(pid, 0.0) if scores else 0.0
                weight = cnt * 2 + score * 10
                if weight > best_weight:
                    best_weight = weight
                    best_pid = pid
            
            if best_pid: return best_pid
            if scores: return max(scores, key=lambda x: scores[x])
        except Exception as e:
            print(f"[WARN] _get_most_used_person_id error: {e}")
        
        return sorted(candidate_ids)[0]

    def _pick_person_id_from_qdrant(self, feats_np: np.ndarray, threshold=0.35):
        candidate_ids = []
        scores = {}
        try:
            if feats_np.shape[0] == 0: return [], {}
            # Lấy mẫu vài vector để search, không cần search hết nếu quá nhiều
            indices = range(0, feats_np.shape[0], max(1, feats_np.shape[0]//5)) 
            for i in indices:
                vec = feats_np[i]
                n = np.linalg.norm(vec)
                if n > 0: vec = vec / n
                
                hits = search(query_vector=vec.tolist(), limit=10, similarity_threshold=threshold)
                if hits:
                    for hit in hits:
                        payload = getattr(hit, "payload", {}) or {}
                        pid = payload.get("person_id") or payload.get("user_id")
                        score = getattr(hit, "score", 0)
                        if pid:
                            candidate_ids.append(pid)
                            scores[pid] = max(scores.get(pid, 0), score)
            return list(set(candidate_ids)), scores
        except Exception as e:
            print(f"[WARN] Qdrant probe error: {e}")
            return [], {}
            
    def _seed_qdrant_for_user(self, person_id, feats_np, camera_id="import"):
        threading.Thread(target=self._async_upsert_batch, args=(person_id, feats_np, camera_id), daemon=True).start()

    def _async_upsert_batch(self, person_id, feats_np, camera_id):
        try:
            features = [f.tolist() for f in feats_np]
            if features:
                upsert(person_id=person_id, features=features, camera_id=camera_id)
        except Exception as e:
            print(f"[WARN] Seed Qdrant error: {e}")

    def refresh_users_id_and_features(self):
        """Load DB in separate thread safely"""
        try:
            # Tính toán local variables trước
            with SessionLocal() as session:
                users = session.query(AccessUser).options(joinedload(AccessUser.features)).all()
                
                local_users_id = []
                local_features = []
                local_name_cache = {}
                
                all_feats_list = []
                all_owners_list = []

                for u in users:
                    feats = []
                    if u.features:
                        for f in u.features:
                            try:
                                vec = f.feature
                                if isinstance(vec, str): vec = json.loads(vec)
                                arr = np.array(vec, dtype=np.float32)
                                if arr.shape == (512,):
                                    n = np.linalg.norm(arr)
                                    if n > 0: arr = arr / n
                                    feats.append(arr)
                            except: pass
                    
                    feats_np = np.array(feats, dtype=np.float32) if feats else np.empty((0,512), dtype=np.float32)
                    
                    # Auto-link person_id if missing
                    if not u.person_id:
                        cands, scores = self._pick_person_id_from_qdrant(feats_np)
                        if cands:
                            best = self._get_most_used_person_id(session, cands, scores)
                            u.person_id = best
                        else:
                            u.person_id = str(uuid.uuid4())
                            if feats_np.shape[0] > 0:
                                self._seed_qdrant_for_user(u.person_id, feats_np)
                        try:
                            session.commit()
                        except: 
                            session.rollback()

                    pid = u.person_id
                    if pid not in local_users_id:
                        local_users_id.append(pid)
                        local_features.append(feats_np)
                    else:
                        idx = local_users_id.index(pid)
                        if feats_np.shape[0] > 0:
                            old = local_features[idx]
                            local_features[idx] = np.vstack([old, feats_np]) if old.shape[0] > 0 else feats_np

                    local_name_cache[pid] = u.name
                    
                    # Prepare matrix
                    if feats_np.shape[0] > 0:
                        all_feats_list.append(feats_np)
                        all_owners_list.extend([pid] * feats_np.shape[0])

                # Build Matrix safely
                if all_feats_list:
                    matrix = np.vstack(all_feats_list).astype(np.float32)
                else:
                    matrix = np.empty((0,512), dtype=np.float32)

                # Atomic update
                self.users_id = local_users_id
                self.features = local_features
                self.user_id_name_cache = local_name_cache
                self.features_matrix = matrix
                self.features_owner_pid = all_owners_list
                
            # print(f"✅ Refreshed {len(self.users_id)} users.")
        except Exception as e:
            print(f"❌ Error refreshing users: {e}")
            # traceback.print_exc()
        return True

    def _pg_top1(self, f: np.ndarray):
        if self.features_matrix.shape[0] == 0:
            return None, -1.0
        try:
            sims = np.dot(self.features_matrix, f)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            best_pid = self.features_owner_pid[best_idx]
            return best_pid, best_sim
        except Exception as e:
            print(f"[ERR] Matrix dot error: {e}")
            return None, -1.0

    def _qdrant_candidates(self, f: np.ndarray, limit=30, thr=None):
        try:
            thr = max(self.qdrant_threshold, (thr if thr is not None else self.qdrant_threshold))
            hits = search(query_vector=f.tolist(), limit=limit, similarity_threshold=thr) or []
            cand, scores = [], {}
            
            for h in hits:
                payload = getattr(h, "payload", None)
                
                # --- FIX: Xử lý trường hợp payload là list ---
                if isinstance(payload, list) and len(payload) > 0:
                    payload = payload[0]
                
                # Đảm bảo payload là dict trước khi .get()
                if not isinstance(payload, dict):
                    continue
                # ---------------------------------------------

                pid = payload.get("person_id")
                if pid:
                    cand.append(pid)
                    scores[pid] = max(scores.get(pid, 0.0), float(getattr(h, "score", 0)))
            
            return list(set(cand)), scores
        except Exception as e:
            print(f"[WARN] _qdrant_candidates error: {e}")
            return [], {}

    def _finalize_new_person(self, trk, f: np.ndarray):
        key = (trk.pad_index, trk.object_id)
        info = self.pending_new.pop(key, None)
        
        avg = f.copy()
        if info and info["feature_accum"]:
            try:
                mat = np.stack(info["feature_accum"], axis=0)
                avg = np.mean(mat, axis=0)
                n = np.linalg.norm(avg)
                if n > 0: avg /= n
            except: pass

        new_pid = str(uuid.uuid4())
        trk.person_id = new_pid
        trk.label = new_pid
        trk.last_match_frame = trk.frame_num
        trk.votes = {}
        trk.unknown_count = 0
        
        self.new_person_cache[new_pid] = (avg, time.time())
        # print(f"➕ [NEW PID] {new_pid}")
        return new_pid

    def merge_person_ids(self, old_id: str, new_id: str):
        threading.Thread(target=self._async_merge, args=(old_id, new_id), daemon=True).start()
    
    def _async_merge(self, old_id, new_id):
        try:
            storage = QdrantFeatureStorage()
            storage.client.set_payload(
                collection_name=storage.collection_name,
                payload={"person_id": new_id},
                points=Filter(must=[FieldCondition(key="person_id", match=MatchValue(value=old_id))])
            )
            print(f"✅ Merged {old_id} -> {new_id}")
        except Exception as e:
            print(f"❌ Merge failed: {e}")

    def _send_event_throttled(self, pad_index, oid, label, x1, y1, x2, y2,
                              person_id, timestamp, frame_org, face_region, clothes_info=None):
        key = (pad_index, oid)
        now = time.time()
        last = self.last_alert_at.get(key, 0)
        if now - last < self.alert_cooldown_sec:
            return
        self.last_alert_at[key] = now
        
        self._send_recognition_to_kafka(pad_index, oid, x1, y1, x2, y2, label,
                                        person_id, timestamp, frame_org, face_region, clothes_info)

    def probe_face(self, pipeline):
        self.pipeline = pipeline
        face_probe = self.pipeline.get_by_name("queue_face")
        if face_probe:
            pad = face_probe.get_static_pad("src")
            if pad:
                pad.add_probe(Gst.PadProbeType.BUFFER, self.face_recognize, None)

    def is_frontal_face(self, landmarks, angle_threshold=15, symmetry_threshold=0.3):
        """
        Kiểm tra xem khuôn mặt có phải là chính diện không dựa trên 5 điểm landmarks.

        Args:
            landmarks: List hoặc array chứa 5 điểm landmarks [(x1,y1), ..., (x5,y5)]
                       Thứ tự: [mắt_trái, mắt_phải, mũi, miệng_trái, miệng_phải]
            angle_threshold: Ngưỡng roll cho phép (độ)
            symmetry_threshold: Ngưỡng đối xứng cho phép (0-1)

        Returns:
            bool: True nếu là mặt chính diện, False nếu bị lệch hoặc sai
        """
        min_eye_distance = 5

        if len(landmarks) != 5:
            return False

        points = np.array(landmarks, dtype=np.float32)
        left_eye, right_eye, nose, left_mouth, right_mouth = points

        # === 0. Kiểm tra khoảng cách 2 mắt cơ bản ===
        eye_distance = np.linalg.norm(right_eye - left_eye)
        if eye_distance < min_eye_distance:
            return False

        # === 1. Kiểm tra roll (góc xoay đầu trái/phải) ===
        eye_dx = right_eye[0] - left_eye[0]
        eye_dy = right_eye[1] - left_eye[1]
        eye_angle = math.degrees(math.atan2(eye_dy, eye_dx))
        if abs(eye_angle) > angle_threshold:
            return False

        # === 2. Kiểm tra yaw (mũi lệch ngang so với trung tâm 2 mắt) ===
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        nose_deviation = abs(nose[0] - eye_center_x)
        nose_symmetry_ratio = nose_deviation / eye_distance
        if nose_symmetry_ratio > symmetry_threshold:
            return False

        # === 3. Kiểm tra yaw (miệng lệch ngang) ===
        mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
        mouth_deviation = abs(mouth_center_x - eye_center_x)
        mouth_symmetry_ratio = mouth_deviation / eye_distance
        if mouth_symmetry_ratio > symmetry_threshold:
            return False

        # === 4. Kiểm tra pitch (đối xứng mắt – miệng) ===
        mouth_center = (left_mouth + right_mouth) / 2
        d1 = np.linalg.norm(left_eye - mouth_center)
        d2 = np.linalg.norm(right_eye - mouth_center)
        if d1 == 0 or d2 == 0:
            return False
        pitch_symmetry = min(d1, d2) / max(d1, d2)
        if pitch_symmetry < (1 - symmetry_threshold):
            return False

        # === 5. Mắt lệch cao/thấp quá nhiều (có thể do nghiêng mặt theo pitch) ===
        eye_vertical_diff = abs(left_eye[1] - right_eye[1])
        if eye_vertical_diff > 0.5 * eye_distance:
            return False

        return True

    def face_recognize(self, pad, info, user_data):
        try:
            gst_buffer = info.get_buffer()
            if not gst_buffer: return Gst.PadProbeReturn.OK

            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
            if not batch_meta: return Gst.PadProbeReturn.OK

            current_time = time.time()
            
            # GC định kỳ để dọn dẹp metadata thừa
            self.frame_counter += 1
            if self.frame_counter % 100 == 0:
                gc.collect()

            l_frame = batch_meta.frame_meta_list
            while l_frame is not None:
                try:
                    frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                except:
                    l_frame = l_frame.next
                    continue

                pad_index = frame_meta.pad_index
                frame_num = frame_meta.frame_num

                with self.lock:
                    if pad_index in self.stopping_streams:
                        l_frame = l_frame.next
                        continue
                    if pad_index not in self.tracks_his:
                        self.tracks_his[pad_index] = Sort(None, 30)

                # 1. PHÂN TÍCH METADATA TRƯỚC (TỐN 0 VRAM)
                # Chúng ta duyệt qua các object, nhận diện khuôn mặt dựa trên Tensor
                # Nhưng KHÔNG lấy ảnh từ GPU
                alert_candidates = [] 

                l_obj = frame_meta.obj_meta_list
                while l_obj is not None:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                        if obj_meta.class_id == 0:
                            # Logic nhận diện nằm hết ở đây
                            decision = self._analyze_face_metadata(obj_meta, frame_meta, pad_index, current_time)
                            
                            # Nếu logic bảo "Cần gửi cảnh báo" -> Lưu lại vào danh sách
                            if decision and decision.get("need_alert"):
                                alert_candidates.append(decision)
                    except: pass
                    try: l_obj = l_obj.next
                    except: break
                
                # 2. SNAPSHOT ON DEMAND (CHỈ MAP KHI CÓ CẢNH BÁO)
                # Nếu danh sách alert_candidates trống -> Không làm gì cả -> KHÔNG LEAK
                if len(alert_candidates) > 0:
                    surface = None
                    frame_bgr = None
                    try:
                        surface = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                        frame_bgr = None

                        if surface is not None:

                            # Surface từ NVMM → host_frame dạng RGBA
                            if hasattr(surface, "get_host_frame"):
                                frame_rgba = surface.get_host_frame()

                                # Convert RGBA → BGR
                                frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)

                            # Nếu surface đã là numpy (CPU)
                            elif isinstance(surface, np.ndarray):
                                frame_src = surface
                                if frame_src.shape[2] == 4:
                                    frame_bgr = cv2.cvtColor(frame_src, cv2.COLOR_RGBA2BGR)
                                else:
                                    frame_bgr = frame_src.copy()


                            
                        # Có ảnh rồi -> Cắt ảnh và gửi Event
                        for cand in alert_candidates:
                            self._dispatch_event(pad_index, frame_bgr, cand)
                                
                    except Exception as e:
                        print(f"Snapshot error: {e}")
                    finally:
                        # Dọn dẹp ngay lập tức
                        surface = None
                        if frame_bgr is not None: del frame_bgr
                        
                # Update FPS
                if self.perf_data: self.perf_data.update_fps(f"stream_face_{pad_index}")
                
                # Cleanup History
                with self.lock: self._cleanup_face_data(pad_index, current_time - 20)

                try: l_frame = l_frame.next
                except: break

        except Exception as e:
            print(f"[CRITICAL] Probe Error: {e}")
        
        try:
            pyds.nvds_remove_batch_meta(gst_buffer, batch_meta)
        except: pass

        return Gst.PadProbeReturn.OK

    def _analyze_face_metadata(self, obj_meta, frame_meta, pad_index, current_time):
        """Xử lý logic nhận diện hoàn toàn trên Metadata (Tensor, ID)"""
        try:
            oid = obj_meta.object_id
            rect = obj_meta.rect_params
            x1, y1 = int(rect.left), int(rect.top)
            x2, y2 = x1 + int(rect.width), y1 + int(rect.height)
            face_box = (x1, y1, x2, y2)
            
            try: ntp_ts = datetime.fromtimestamp(frame_meta.ntp_timestamp/1e9)
            except: ntp_ts = datetime.now()

            # 1. Extract Feature (Pointer Read - Safe)
            feature = None
            l_user = obj_meta.obj_user_meta_list
            while l_user is not None:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                    if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                        tens = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                        if tens.num_output_layers > 0:
                            layer0 = pyds.get_nvds_LayerInfo(tens, 0)
                            if layer0 and layer0.buffer:
                                ptr = pyds.get_ptr(layer0.buffer)
                                if ptr:
                                    raw = ctypes.string_at(ptr, 512*4)
                                    arr = np.frombuffer(raw, np.float32).copy()
                                    n = np.linalg.norm(arr)
                                    if n > 0: arr /= n
                                    feature = arr
                except: pass
                try: l_user = l_user.next
                except: break

            # 2. Update Tracker
            with self.lock:
                tracker = self.tracks_his[pad_index]
            
            ts_float = ntp_ts.timestamp()
            tracker.update([x1, y1, x2, y2, obj_meta.class_id, int(oid), ts_float, feature, frame_meta.frame_num])
            
            tgt_trk = None
            for trk in tracker.trackers:
                if trk.object_id == oid:
                    tgt_trk = trk
                    break
            if not tgt_trk: return None

            # 3. Matching Logic
            person_id = getattr(tgt_trk, 'label', None)
            if not person_id and getattr(tgt_trk, 'feature', None) is not None:
                person_id = self.match_face(tgt_trk.feature, pad_index, trk=tgt_trk, frame_num=frame_meta.frame_num)

            # 4. Save History (Chỉ lưu metadata, KHÔNG LƯU ẢNH FULL)
            with self.lock:
                if pad_index not in self.data_face: self.data_face[pad_index] = {}
                if frame_meta.frame_num not in self.data_face[pad_index]: self.data_face[pad_index][frame_meta.frame_num] = []
                # Mini info chỉ chứa tọa độ
                mini_info = {"object_id": oid, "face_box": face_box}
                self.data_face[pad_index][frame_meta.frame_num].append((current_time, face_box, mini_info))

            # 5. Check Alert Condition
            if person_id:
                key = (pad_index, person_id)
                if current_time - self.last_alert_at.get(key, 0) >= self.alert_cooldown_sec:
                    # ĐÁNH DẤU LÀ CẦN SNAPSHOT
                    self.last_alert_at[key] = current_time
                    user_name = self.user_id_name_cache.get(person_id, "Unknown")
                    return {
                        "need_alert": True,
                        "oid": oid,
                        "person_id": person_id,
                        "label": user_name,
                        "face_box": face_box,
                        "timestamp": ntp_ts,
                        "tracker_info": getattr(tgt_trk, 'clothes_info', None)
                    }
            return None
        except: return None

    def _dispatch_event(self, pad_index, frame_full, info):
        try:
            face_box = info["face_box"]
            x1, y1, x2, y2 = face_box
            
            h, w = frame_full.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            face_img = None
            if x2 > x1 and y2 > y1:
                face_img = frame_full[y1:y2, x1:x2].copy()

            clothes_info = self._ci_to_dict(info["tracker_info"])
            # print("info = ",info)
            evt = {
                "pad_index": pad_index,
                "oid": info["oid"],
                "person_id": info["person_id"],
                "label": info["label"],
                "timestamp": info["timestamp"],
                "frame_full": frame_full.copy(),
                "face_img": face_img,
                "face_region": {"x1":x1, "y1":y1, "x2":x2, "y2":y2},
                "clothes_info": clothes_info
            }
            EVENT_QUEUE.put(evt, block=False)
        except: pass

    def _process_face_object(self, obj_meta, frame_meta, frame, pad_index):
        try:
            oid = obj_meta.object_id
            rect = obj_meta.rect_params
            x1, y1 = int(rect.left), int(rect.top)
            x2, y2 = x1 + int(rect.width), y1 + int(rect.height)
            face_box = (x1, y1, x2, y2)

            # NTP Timestamp
            try:
                ntp_ts = datetime.fromtimestamp(frame_meta.ntp_timestamp / 1e9)
            except:
                ntp_ts = datetime.now()

            # --- FIX: Kiểm tra config an toàn ---
            # Nếu config bị lỗi thành list/None, dùng giá trị mặc định
            is_front_face = 1
            angle_threshold = 30
            symmetry_threshold = 0.3
            
            if isinstance(config, dict):
                is_front_face = config.get("is_front_face", 1)
                angle_threshold = config.get("angle_threshold", 30)
                symmetry_threshold = config.get("symmetry_threshold", 0.3)
            # ------------------------------------

            if is_front_face:
                landmarks = None
                try:
                    mp = obj_meta.mask_params
                    if hasattr(mp, "get_mask_array"):
                        arr = mp.get_mask_array()
                        if len(arr) == 14:
                            h, w, c = frame.shape
                            def det_scale(in_h, in_w, out_h=640, out_w=640):
                                r1 = in_h / in_w
                                r2 = out_h / out_w
                                if r1 > r2: return out_h / in_h
                                else: return out_w / in_w

                            sc = det_scale(h, w, mp.height, mp.width)
                            # Lưu ý: DeepStream landmark format
                            landmarks = [
                                (arr[1] / sc, arr[0] / sc),
                                (arr[3] / sc, arr[2] / sc),
                                (arr[5] / sc, arr[4] / sc),
                                (arr[7] / sc, arr[6] / sc),
                                (arr[9] / sc, arr[8] / sc),
                            ]
                except Exception: pass

                if landmarks:
                    if not self.is_frontal_face(landmarks, angle_threshold, symmetry_threshold):
                        return None

            # Extract Feature Tensor
            feature = None
            l_user = obj_meta.obj_user_meta_list
            while l_user is not None:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                    if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                        tens = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                        if tens.num_output_layers > 0:
                            layer0 = pyds.get_nvds_LayerInfo(tens, 0)
                            if layer0 and layer0.buffer:
                                ptr = pyds.get_ptr(layer0.buffer)
                                if ptr:
                                    raw = ctypes.string_at(ptr, 512*4)
                                    arr = np.frombuffer(raw, np.float32).copy()
                                    n = np.linalg.norm(arr)
                                    if n > 0: arr /= n
                                    feature = arr


                                    try:
                                        pyds.nvds_remove_user_meta_from_obj(obj_meta, user_meta)
                                    except Exception as e:
                                        print("Free tensor meta failed:", e)
                except Exception: pass
                try: l_user = l_user.next
                except: break

            # Update Tracker
            with self.lock:
                if pad_index not in self.tracks_his:
                    self.tracks_his[pad_index] = Sort(None, 30)
                tracker = self.tracks_his[pad_index]
            
            if tracker:
                ts_float = ntp_ts.timestamp()
                det_box = [x1, y1, x2, y2, obj_meta.class_id, int(oid), ts_float, feature, frame_meta.frame_num]
                tracker.update(det_box)

                # Run Recognition Logic
                label = None
                if feature is not None and frame is not None:
                    label = self._match_face_with_database(
                        pad_index, oid, frame, {"x1":x1, "y1":y1, "x2":x2, "y2":y2},
                        ntp_ts, x1, y1, x2, y2, face_box, frame_meta.frame_num
                    )
                return {"object_id": oid, "face_box": face_box, "timestamp": ntp_ts, "label": label}
                
        except Exception as e:
            print(f"[WARN] _process_face_object error: {e}")
            traceback.print_exc() # In ra dòng lỗi chi tiết để debug nếu còn lỗi
            return None
        return None

    def match_face(self, feature, pad_index, trk=None, frame_num=0):
        """Logic nhận diện chính (đã tối ưu flow)"""
        if trk and getattr(trk, "person_id", None):
            if (frame_num - getattr(trk, "last_match_frame", 0)) < self.stable_frames:
                return trk.person_id
        
        f = feature
        best_pid_pg, best_sim_pg = self._pg_top1(f)

        # A. Match High (Postgres)
        if best_pid_pg and best_sim_pg >= self.pg_sim_high:
            if trk: self._update_trk(trk, best_pid_pg, frame_num)
            return best_pid_pg

        # B. Match Medium (Voting)
        if best_pid_pg and (self.pg_sim_low <= best_sim_pg < self.pg_sim_high) and trk:
            # --- FIX: Kiểm tra kỹ kiểu dữ liệu của votes ---
            votes = getattr(trk, "votes", None)
            
            # Nếu votes chưa có, hoặc bị lỗi thành List -> Reset về Dict rỗng
            if not isinstance(votes, dict):
                votes = {}
            # ----------------------------------------------

            votes[best_pid_pg] = votes.get(best_pid_pg, 0) + 1
            trk.votes = votes
            
            if votes[best_pid_pg] >= 2:
                self._update_trk(trk, best_pid_pg, frame_num)
                return best_pid_pg

        # C. Qdrant Fallback
        if (best_sim_pg < self.pg_sim_floor) or not best_pid_pg:
            now = time.time()
            if trk and (now - getattr(trk, "last_qdrant_time", 0)) < 5.0:
                return getattr(trk, "person_id", None)
            
            if trk: trk.last_qdrant_time = now
            cands, scores = self._qdrant_candidates(f, limit=10, thr=self.qdrant_threshold)
            if cands:
                with SessionLocal() as session:
                    best = self._get_most_used_person_id(session, cands, scores)
                if best:
                    if trk: self._update_trk(trk, best, frame_num)
                    return best

        # D. Create New ID
        if trk:
            trk.unknown_count = getattr(trk, "unknown_count", 0) + 1
            key = (pad_index, trk.object_id)
            
            if key not in self.pending_new:
                self.pending_new[key] = {"start": frame_num, "accum": [f], "last_check": time.time()}
                return None
            
            info = self.pending_new[key]
            info["accum"].append(f)
            
            if (frame_num - info["start"]) >= self.min_unknown_frames_before_create:
                if (time.time() - info["last_check"]) > self.recheck_cooldown_sec:
                    info["last_check"] = time.time()
                    c2, _ = self._qdrant_candidates(f, thr=self.qdrant_low_threshold)
                    if c2:
                        with SessionLocal() as sess: best2 = self._get_most_used_person_id(sess, c2)
                        self._update_trk(trk, best2, frame_num)
                        del self.pending_new[key]
                        return best2

                    return self._finalize_new_person(trk, f)

        return None

    def _update_trk(self, trk, pid, fnum):
        trk.person_id = pid
        trk.label = pid
        trk.last_match_frame = fnum
        trk.votes = {}
        trk.unknown_count = 0

    def _match_face_with_database(self, pad_index, oid, frame, face_region, ntp_ts, x1, y1, x2, y2, face_box, frame_num):
        # Tìm tracker tương ứng
        tracker = self.tracks_his.get(pad_index)
        if not tracker: return None
        
        tgt_trk = None
        for trk in tracker.trackers:
            if trk.object_id == oid: # Sort object_id might differ from DS object_id, but here we assumed assignment
                tgt_trk = trk
                break
        
        if not tgt_trk: return None

        person_id = getattr(tgt_trk, 'label', None)
        
        # Nếu chưa có ID -> chạy nhận diện
        if not person_id and getattr(tgt_trk, 'feature', None) is not None:
            person_id = self.match_face(tgt_trk.feature, pad_index, trk=tgt_trk, frame_num=frame_num)
        
        # Nếu đã có ID -> Gửi event
        if person_id:
            user_name = self.user_id_name_cache.get(person_id, "Unknown")
            
            # Clothes
            ci = self._ci_to_dict(getattr(tgt_trk, 'clothes_info', None))
            if not self._ci_has_colors(ci) and self.clothes_recognizer:
                # Logic tìm quần áo ở đây...
                pass

            self._send_event_throttled(pad_index, oid, user_name, x1, y1, x2, y2, person_id, ntp_ts, frame, face_region, ci)
            
        return person_id

    def _cleanup_face_data(self, pad_index, cutoff_time):
        if pad_index in self.data_face:
            d = self.data_face[pad_index]
            # 1. Time based cleanup
            keys_to_del = [k for k, v in d.items() if v and v[0][0] < cutoff_time]
            for k in keys_to_del: del d[k]
            
            # 2. [OPTIMIZATION 4] Hard Limit (Fix RAM Leak)
            if len(d) > 500:
                sorted_frames = sorted(d.keys())
                for k in sorted_frames[:100]:
                    del d[k]

    def _send_recognition_to_kafka(self, *args):
        try:
            # Ensure frame copy for thread safety
            f_org = args[9]
            f_safe = None
            if f_org is not None:
                f_safe = f_org.copy()
            
            # Reconstruct args with safe frame
            new_args = list(args)
            new_args[9] = f_safe
            
            # Extract face crop safely here to avoid sending full frame ref
            face_reg = args[10]
            face_img = None
            if f_safe is not None and face_reg:
                x1, y1 = int(face_reg['x1']), int(face_reg['y1'])
                x2, y2 = int(face_reg['x2']), int(face_reg['y2'])
                h, w = f_safe.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    face_img = f_safe[y1:y2, x1:x2].copy()

            evt = {
                "pad_index": args[0],
                "oid": args[1],
                "label": args[6],
                "person_id": args[7],
                "timestamp": args[8],
                "frame_full": f_safe,
                "face_img": face_img,
                "clothes_info": args[11],
                "face_region": face_reg
            }
            
            EVENT_QUEUE.put(evt, block=False)
        except queue.Full:
            pass
        except Exception as e:
            print(f"[ERR] Queue put: {e}")

    def _async_upsert(self, person_id, vec, cam_id):
        try:
            upsert(person_id=person_id, features=[vec.tolist()], camera_id=cam_id)
        except: pass

    def flush_new_persons(self):
        now = time.time()
        remove = []
        for pid, (vec, t) in list(self.new_person_cache.items()):
            if now - t > 2.0:
                threading.Thread(target=self._async_upsert, args=(pid, vec, "auto"), daemon=True).start()
                remove.append(pid)
        for pid in remove:
            del self.new_person_cache[pid]
        return True