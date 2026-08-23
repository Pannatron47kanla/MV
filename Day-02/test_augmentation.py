"""
Test data augmentation on a sample YOLO-format image and verify the
bounding box still lines up with the object after each augmentation.

For every augmentation this script:
  1. applies the same geometric transform to the image and to the
     bounding box corners,
  2. draws the box on both the original and augmented image,
  3. saves an original|augmented comparison image so the box can be
     checked against the dog by eye, and
  4. prints sanity checks (box stays inside the image, has positive
     width/height, count is preserved).

Usage:
    python test_augmentation.py \
        --image Day-02/Add24/myDogs2/train/images/<file>.jpg \
        --out Day-02/Add24/augmentation_preview
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def load_yolo_labels(label_path: Path):
    boxes = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        cls, xc, yc, w, h = line.split()
        boxes.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return boxes


def yolo_to_xyxy(box, W, H):
    _, xc, yc, w, h = box
    x1 = (xc - w / 2) * W
    y1 = (yc - h / 2) * H
    x2 = (xc + w / 2) * W
    y2 = (yc + h / 2) * H
    return x1, y1, x2, y2


def xyxy_to_yolo(cls, x1, y1, x2, y2, W, H):
    x1, x2 = sorted((max(0.0, min(x1, W)), max(0.0, min(x2, W))))
    y1, y2 = sorted((max(0.0, min(y1, H)), max(0.0, min(y2, H))))
    w, h = x2 - x1, y2 - y1
    xc, yc = x1 + w / 2, y1 + h / 2
    return (cls, xc / W, yc / H, w / W, h / H)


def draw_boxes(img, boxes, color=(0, 255, 0)):
    out = img.copy()
    H, W = out.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(box, W, H)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return out


# ---- augmentations: each returns (new_img, new_boxes) ----

def aug_horizontal_flip(img, boxes):
    new_img = img[:, ::-1].copy()
    new_boxes = [(cls, 1.0 - xc, yc, w, h) for cls, xc, yc, w, h in boxes]
    return new_img, new_boxes


def aug_rotate(img, boxes, angle_deg):
    H, W = img.shape[:2]
    center = (W / 2, H / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    new_img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)

    new_boxes = []
    for box in boxes:
        cls = box[0]
        x1, y1, x2, y2 = yolo_to_xyxy(box, W, H)
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        corners_h = np.hstack([corners, np.ones((4, 1))])
        warped = corners_h @ M.T
        nx1, ny1 = warped[:, 0].min(), warped[:, 1].min()
        nx2, ny2 = warped[:, 0].max(), warped[:, 1].max()
        new_boxes.append(xyxy_to_yolo(cls, nx1, ny1, nx2, ny2, W, H))
    return new_img, new_boxes


def aug_scale_crop(img, boxes, crop_frac, rng):
    """Zoom in by cropping a slightly smaller region (chosen at a random
    offset) and resizing it back up to the original size."""
    H, W = img.shape[:2]
    crop_w, crop_h = int(W * (1 - crop_frac)), int(H * (1 - crop_frac))
    x0 = rng.randint(0, W - crop_w)
    y0 = rng.randint(0, H - crop_h)
    cropped = img[y0:y0 + crop_h, x0:x0 + crop_w]
    new_img = cv2.resize(cropped, (W, H))
    sx, sy = W / crop_w, H / crop_h

    new_boxes = []
    for box in boxes:
        cls = box[0]
        x1, y1, x2, y2 = yolo_to_xyxy(box, W, H)
        nx1, ny1 = (x1 - x0) * sx, (y1 - y0) * sy
        nx2, ny2 = (x2 - x0) * sx, (y2 - y0) * sy
        if nx2 <= 0 or ny2 <= 0 or nx1 >= W or ny1 >= H:
            continue  # object cropped out entirely
        new_boxes.append(xyxy_to_yolo(cls, nx1, ny1, nx2, ny2, W, H))
    return new_img, new_boxes


def aug_brightness_contrast(img, boxes, alpha, beta):
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return new_img, boxes  # purely photometric: geometry untouched


# ---- verification ----

def verify_boxes(name, orig_boxes, new_boxes):
    ok = True
    if len(new_boxes) != len(orig_boxes):
        print(f"  [{name}] WARN: box count changed {len(orig_boxes)} -> {len(new_boxes)}")
    for cls, xc, yc, w, h in new_boxes:
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            print(f"  [{name}] FAIL: box center out of bounds ({xc:.3f}, {yc:.3f})")
            ok = False
        if w <= 0 or h <= 0:
            print(f"  [{name}] FAIL: non-positive box size (w={w:.3f}, h={h:.3f})")
            ok = False
    if ok:
        print(f"  [{name}] OK: {len(new_boxes)} box(es), all within image bounds")
    return ok


SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--image",
        type=Path,
        default=SCRIPT_DIR / "Add24" / "myDogs2" / "train" / "images"
        / "french_bulldog_01_jpg.rf.d680138bda9b9032b4b244e55929d5a2.jpg",
    )
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "Add24" / "augmentation_preview")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    label_path = args.image.parent.parent / "labels" / (args.image.stem + ".txt")
    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"could not read image: {args.image}")
    boxes = load_yolo_labels(label_path)
    rng = random.Random(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    orig_preview = draw_boxes(img, boxes)

    augmentations = {
        "horizontal_flip": lambda: aug_horizontal_flip(img, boxes),
        "rotate_15deg": lambda: aug_rotate(img, boxes, 15),
        "rotate_-20deg": lambda: aug_rotate(img, boxes, -20),
        "scale_crop_15pct": lambda: aug_scale_crop(img, boxes, 0.15, rng),
        "brightness_up": lambda: aug_brightness_contrast(img, boxes, alpha=1.0, beta=40),
        "contrast_down": lambda: aug_brightness_contrast(img, boxes, alpha=0.7, beta=0),
    }

    print(f"source image : {args.image}")
    print(f"source labels: {label_path} ({len(boxes)} box(es))")
    print()

    all_ok = True
    for name, fn in augmentations.items():
        new_img, new_boxes = fn()
        ok = verify_boxes(name, boxes, new_boxes)
        all_ok = all_ok and ok

        aug_preview = draw_boxes(new_img, new_boxes, color=(0, 0, 255))
        h1, w1 = orig_preview.shape[:2]
        h2, w2 = aug_preview.shape[:2]
        canvas = np.zeros((max(h1, h2), w1 + w2 + 10, 3), dtype=np.uint8)
        canvas[:h1, :w1] = orig_preview
        canvas[:h2, w1 + 10 : w1 + 10 + w2] = aug_preview
        cv2.imwrite(str(args.out / f"{name}.jpg"), canvas)

    print()
    print(f"comparison images written to: {args.out}/")
    print("all checks passed" if all_ok else "SOME CHECKS FAILED -- see above")


if __name__ == "__main__":
    main()
