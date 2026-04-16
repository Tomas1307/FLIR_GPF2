"""Quick smoke test to verify YOLO26 works with the current ultralytics version."""

from ultralytics import YOLO, __version__

print(f"ultralytics version: {__version__}")

for weights in ("yolo26m.pt", "yolo26l.pt"):
    try:
        model = YOLO(weights)
        print(f"{weights}: OK — {model.info(verbose=False)}")
    except Exception as e:
        print(f"{weights}: FAILED — {e}")
