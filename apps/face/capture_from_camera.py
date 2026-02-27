import cv2, time

def try_dev(dev):
    print(f"\nTrying {dev}")
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"[FAIL] open {dev}")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # warmup
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    ok, frame = cap.read()
    cap.release()

    if not ok:
        print("[FAIL] read frame")
        return False

    filename = f"data/face/examples/test_{dev.replace('/','')}.jpg"
    cv2.imwrite(filename, frame)
    print(f"[OK] saved {filename} shape={frame.shape}")
    return True


# thử cả 2 device
try_dev("/dev/video0")
try_dev("/dev/video1")