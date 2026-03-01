# Plan: Migrate FaceDB to Qdrant + Auto-Save Hard Cases

**What:** Replace the JSON-based `FaceDatabase` with Qdrant vector DB for real-time nearest-neighbor search, then add a background auto-save system that buffers hard-case embeddings (masked/covered faces) per tracker and upserts them to Qdrant upon identity confirmation.

**Why:** Qdrant enables incremental upserts without reloading the full matrix, scales better, and supports the metadata needed for the auto-save cronjob (source tagging, per-person filtering).

---

## Phase 1 — Migrate FaceDatabase → Qdrant

### Steps

1. Add `qdrant-client` to `scripts/requirements.txt`

2. Create new class `QdrantFaceDatabase` in `apps/face/processor.py` to replace `FaceDatabase`:
   - `__init__`: connect to `QdrantClient(host="localhost", port=6333)`, create collection `faces` if not exists (512-dim, `Distance.EUCLID` to match current L2 behavior)
   - `_import_from_json(path)`: reads `features.json`, upserts all entries as points with payload `{name, avatar, source="manual"}`. Only runs if collection is empty or `force_reload=True`
   - `match(embedding) → (name: str, distance: float)`: calls `client.search("faces", embedding, limit=1, with_payload=True)` — returns name from payload and score
   - `add_embeddings(name: str, embeddings: list[np.ndarray], source: str = "auto")`: upserts new points
   - `names` property: kept for compatibility (returns list of distinct names from collection)
   - `avatars` property: dict populated at load time

3. Update `FaceRecognitionProcessor.__init__` to instantiate `QdrantFaceDatabase` instead of `FaceDatabase`. Pass `features_json` path for initial import.

4. Update `apps/face/config.yaml` to add Qdrant params:
   ```yaml
   params:
     qdrant_host: "localhost"
     qdrant_port: 6333
     qdrant_collection: "faces"
     features_json: "data/face/features.json"   # migration source, loaded once if collection empty
   ```

5. Update `apps/face/extract.py` to upsert directly to Qdrant instead of writing JSON (keep JSON write as optional backup)

---

## Phase 2 — Hard-Case Auto-Save Cronjob

**Idea recap:** Use a _dual-threshold_ strategy:
- `l2_threshold` (strict, e.g. 1.0): required for identity confirmation (no mask → high confidence)
- `collect_threshold` (loose, e.g. 1.3): collect candidate embeddings even when below strict threshold (masked frames)

When the person removes the mask and gets confirmed → the buffered hard-case embeddings are auto-saved to Qdrant so the next run recognizes them while masked.

### Steps

6. Extend `TrackedFace` dataclass in `apps/face/processor.py`:
   - Add `candidate_buffer: deque` (field with `default_factory`) — stores `(frame: int, embedding: np.ndarray, distance: float)` tuples
   - Add `collect_threshold: float = 1.3` field
   - In `add_match()`: if `distance <= collect_threshold` (even if > `l2_threshold`), append to `candidate_buffer` (capped at `buffer_size`, default 50)
   - `get_best_candidates(n: int) → list[np.ndarray]`: return top-N embeddings sorted by distance ascending from `candidate_buffer`

7. Create `AutoSaveWorker` class in `apps/face/processor.py`:
   - Background thread with `queue.Queue`
   - Each task: `{name, embeddings: list[ndarray], source: "auto"}`
   - Worker calls `db.add_embeddings(name, embeddings, source="auto")`
   - Config params: `auto_save_enabled: bool`, `auto_save_min_candidates: int = 2`, `auto_save_max_per_person: int = 5`
   - Start/stop via `on_start()`/`on_stop()` lifecycle

8. Update `FaceRecognitionProcessor._process_face()`:
   - When `trk.confirm(name)` returns `True` (first confirmation):
     - If `auto_save_enabled` and `len(trk.candidate_buffer) >= min_candidates`:
       - Get best N candidates: `trk.get_best_candidates(max_per_person)`
       - Submit to `AutoSaveWorker`

9. Update `apps/face/config.yaml` with new params:
   ```yaml
   params:
     collect_threshold: 1.3
     auto_save_enabled: true
     auto_save_min_candidates: 2
     auto_save_max_per_person: 5
     auto_save_buffer_size: 50
   ```

10. Update `TrackerManager.get_or_create()` to pass `collect_threshold` and `buffer_size` from config to `TrackedFace`

---

## Migration Script (one-time)

11. Create `apps/face/migrate_json_to_qdrant.py` — standalone script:
    - Reads `features.json`, connects to Qdrant, creates collection if needed, upserts all faces
    - Run once before switching to new processor:
      ```bash
      python3 apps/face/migrate_json_to_qdrant.py --json data/face/features.json
      ```

---

## Verification

```bash
# 1. Install qdrant-client inside container
docker exec -w /app vmsx pip install qdrant-client

# 2. Run migration
docker exec -w /app vmsx python3 apps/face/migrate_json_to_qdrant.py --json data/face/features.json

# 3. Check collection via Qdrant REST
curl http://localhost:6333/collections/faces | python3 -m json.tool

# 4. Start pipeline normally
docker exec -it -w /app vmsx bash entry/run_pipeline.sh

# 5. Check auto-save triggered: watch logs for "[AutoSave]" prefix
# 6. Verify new embeddings: GET /collections/faces/points/scroll?with_payload=true
```

---

## Decisions

- **Distance metric:** `EUCLID` (L2) to preserve identical behavior with current `np.linalg.norm` matching
- **Qdrant point IDs:** `uuid5(namespace, f"{name}_{index}")` for determinism during re-import
- **`auto`-sourced embeddings** are stored separately via `source` payload field, enabling future cleanup/audit
- **`features.json` import:** runs once if the Qdrant collection is empty; subsequent pipeline starts skip import
