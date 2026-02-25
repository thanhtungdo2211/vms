#!/usr/bin/env python3
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx


def resolve_path(value: str) -> Path:
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def resolve_images(cfg: dict) -> tuple[Path, Path]:
    shared = str(cfg.get("image", "") or "").strip()
    face = str(cfg.get("image_face", "") or shared).strip()
    full = str(cfg.get("image_full", "") or shared).strip()

    missing = []
    if not face:
        missing.append("Thieu image_face hoac image.")
    if not full:
        missing.append("Thieu image_full hoac image.")
    if missing:
        raise ValueError("\n".join(missing))

    p_face = resolve_path(face)
    p_full = resolve_path(full)

    not_found = []
    for label, p in [
        ("image_face", p_face),
        ("image_full", p_full),
    ]:
        if not p.is_file():
            not_found.append(f"Khong tim thay {label}: {p}")
    if not_found:
        raise FileNotFoundError("\n".join(not_found))

    return p_face, p_full


def required_str(cfg: dict, key: str) -> str:
    value = cfg.get(key)
    if value is None:
        raise ValueError(f"Thieu truong bat buoc: {key}")
    s = str(value).strip()
    if not s:
        raise ValueError(f"Truong {key} khong duoc de trong")
    return s


def main() -> int:
    # FIX CỨNG CONFIG Ở ĐÂY
    cfg = {
        "base_url": "http://192.168.6.39:5555",
        "token": "",
        "person_id": "son123",
        "stream_id": 1,
        "time_access": "now",
        "image": "/home/mqs/Desktop/son3.jpg",
        "image_face": "",
        "image_full": "",
        "timeout": 60,
    }

    try:
        image_face, image_full = resolve_images(cfg)
    except Exception as err:
        print(f"[ERROR] {err}")
        return 2

    try:
        base_url = required_str(cfg, "base_url").rstrip("/")
        endpoint = f"{base_url}/api/v1/event/access-events"
        person_id = required_str(cfg, "person_id")
        stream_id = str(int(cfg.get("stream_id", 1)))
        timeout = float(cfg.get("timeout", 60))
    except Exception as err:
        print(f"[ERROR] Cau hinh khong hop le: {err}")
        return 2

    data = {
        "person_id": person_id,
        "stream_id": stream_id,
    }

    time_access = str(cfg.get("time_access", "") or "").strip()
    if time_access.lower() == "now":
        data["time_access"] = datetime.utcnow().replace(microsecond=0).isoformat()
    elif time_access:
        data["time_access"] = time_access

    headers = {}
    token = str(cfg.get("token", "") or "").strip()
    if token:
        if not token.lower().startswith("bearer "):
            token = f"Bearer {token}"
        headers["Authorization"] = token

    print("[INFO] Gui su kien test (fix cung config)")
    print(f"  URL: {endpoint}")
    print(f"  person_id: {person_id}")
    print(f"  stream_id: {stream_id}")
    print(f"  image_face: {image_face}")
    print(f"  image_full: {image_full}")

    try:
        with (
            image_face.open("rb") as f_face,
            image_full.open("rb") as f_full,
            httpx.Client(timeout=timeout) as client,
        ):
            files = {
                "image_face": (image_face.name, f_face, "image/jpeg"),
                "image_full": (image_full.name, f_full, "image/jpeg"),
            }
            response = client.post(endpoint, data=data, files=files, headers=headers)
    except Exception as err:
        print(f"[ERROR] Request loi: {err}")
        return 1

    print(f"[INFO] HTTP {response.status_code}")
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type.lower():
        try:
            print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        except Exception:
            print(response.text)
    else:
        print(response.text)

    return 0 if response.is_success else 1


if __name__ == "__main__":
    sys.exit(main())
