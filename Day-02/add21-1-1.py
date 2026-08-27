"""
Geometric Transformations demo on add21.jpg.

Produces:
  1.0 -- Original Image
  1.1 -- Resizing (Scaling)
  1.2 -- Rotation
  1.3 -- Affine Transformation
  1.4 -- Perspective Transformation

Each saved image gets a short red Thai caption strip underneath explaining
what was done, so screenshots can be dropped straight into a report. cv2's
own text drawing (Hershey fonts) can't render Thai glyphs, so captions are
rendered with Pillow using a Thai-capable font and composited onto the
OpenCV image. A combined summary image (1.0 on top, 1.1|1.2, 1.3|1.4 below,
matching the report layout) is also written.

Usage:
    python add21-1-1 --image add21.jpg --out geo_transform_output
    python add21-1-1 --no-show   # skip the interactive windows, just save files
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RED = (255, 0, 0)  # RGB (Pillow draws in RGB)
CAPTION_BG = (255, 255, 255)

THAI_FONT_CANDIDATES = [
    "/usr/share/fonts/noto/NotoSansThai-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansThai-Medium.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
]


def _load_font(size):
    for path in THAI_FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _wrap_text(text, font, draw, max_width):
    """Greedily wrap `text` into lines that each fit within max_width pixels."""
    words = text.split(" ")
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        box = draw.textbbox((0, 0), candidate, font=font)
        if box[2] - box[0] > max_width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def add_caption(img, text, font_size=22, pad=10):
    """Return a copy of img (BGR ndarray) with a white strip added below it
    and `text` drawn on the strip in red Thai-capable text, wrapped to fit
    the image width."""
    h, w = img.shape[:2]
    font = _load_font(font_size)

    pil_img = Image.new("RGB", (w, 10))  # throwaway canvas just to measure text
    draw = ImageDraw.Draw(pil_img)
    max_width = w - 2 * pad
    lines = _wrap_text(text, font, draw, max_width)

    ascent, descent = font.getmetrics()
    line_step = ascent + descent + 6
    caption_height = pad + line_step * len(lines) + pad // 2

    canvas = Image.new("RGB", (w, h + caption_height), CAPTION_BG)
    canvas.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (0, 0))
    draw = ImageDraw.Draw(canvas)

    y = h + pad
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        text_w = box[2] - box[0]
        x = max(pad, (w - text_w) // 2)
        draw.text((x, y), line, font=font, fill=RED)
        y += line_step

    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def resize_to_width(img, width):
    h, w = img.shape[:2]
    scale = width / w
    return cv2.resize(img, (width, int(h * scale)))


# ---- 1.1 resizing / scaling ----
def do_resize(img, scale=0.6):
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


# ---- 1.2 rotation ----
def do_rotate(img, angle_deg=45):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # expand canvas so the rotated image isn't cropped
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img, M, (new_w, new_h), borderValue=(255, 255, 255))


# ---- 1.3 affine transformation (3-point mapping) ----
def do_affine(img):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    dst = np.float32([[w * 0.05, h * 0.15], [w * 0.9, 0], [w * 0.15, h * 0.9]])
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))


# ---- 1.4 perspective transformation (4-point mapping) ----
def do_perspective(img):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(
        [
            [w * 0.10, h * 0.05],
            [w * 0.95, h * 0.15],
            [w * 0.85, h * 0.95],
            [0, h * 0.85],
        ]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))


def hstack_captioned(a, b):
    """Stack two captioned images side by side, padding the shorter one."""
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = max(ha, hb)
    a_pad = np.full((h, wa, 3), CAPTION_BG, dtype=np.uint8)
    b_pad = np.full((h, wb, 3), CAPTION_BG, dtype=np.uint8)
    a_pad[:ha] = a
    b_pad[:hb] = b
    gap = np.full((h, 10, 3), CAPTION_BG, dtype=np.uint8)
    return np.hstack([a_pad, gap, b_pad])


def vstack_center(*rows):
    """Stack rows of possibly different widths, centering the narrower ones."""
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        h, w = r.shape[:2]
        if w == max_w:
            padded.append(r)
            continue
        canvas = np.full((h, max_w, 3), CAPTION_BG, dtype=np.uint8)
        x0 = (max_w - w) // 2
        canvas[:, x0 : x0 + w] = r
        padded.append(canvas)
    gap = np.full((10, max_w, 3), CAPTION_BG, dtype=np.uint8)
    out = padded[0]
    for r in padded[1:]:
        out = np.vstack([out, gap, r])
    return out


SCRIPT_DIR = Path(__file__).resolve().parent


WINDOW_NAME = "Geometric Transformations"


def show_step(img, wait_hint=True):
    """Display img in a cv2 window, waiting for a keypress before returning.
    Returns False if the user pressed q/Esc to quit early."""
    h, w = img.shape[:2]
    max_h = 900
    if h > max_h:
        scale = max_h / h
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, img)
    if wait_hint:
        print("  (window open -- press any key to continue, q/Esc to skip remaining)")
    key = cv2.waitKey(0) & 0xFF
    return key not in (27, ord("q"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, default=SCRIPT_DIR / "add21.jpg")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "geo_transform_output")
    ap.add_argument(
        "--no-show", action="store_true", help="skip the interactive cv2 windows"
    )
    args = ap.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"could not read image: {args.image}")
    img = resize_to_width(img, 500)  # keep output sizes manageable
    args.out.mkdir(parents=True, exist_ok=True)

    steps = [
        ("1.0_original", img.copy(), "1.0 -- ภาพต้นฉบับก่อนทำการแปลงใดๆ"),
        (
            "1.1_resizing",
            do_resize(img),
            "1.1 -- ปรับขนาดภาพ (ย่อ/ขยาย) โดยคงสัดส่วนเดิม ที่นี่ย่อเหลือ 0.6 เท่า",
        ),
        (
            "1.2_rotation",
            do_rotate(img),
            "1.2 -- หมุนภาพรอบจุดศูนย์กลาง 45 องศา ขยายเฟรมกันภาพถูกตัดขอบ",
        ),
        (
            "1.3_affine",
            do_affine(img),
            "1.3 -- แปลงภาพแบบ Affine ใช้จุดอ้างอิง 3 จุด เส้นขนานยังคงขนานกันหลังแปลง",
        ),
        (
            "1.4_perspective",
            do_perspective(img),
            "1.4 -- แปลงภาพแบบ Perspective ใช้จุดอ้างอิง 4 จุด จำลองมุมมองเปลี่ยนไป เส้นขนานไม่จำเป็นต้องขนานกัน",
        ),
    ]

    captioned = {}
    keep_going = True
    for name, result, caption in steps:
        out_img = add_caption(result, caption)
        captioned[name] = out_img
        cv2.imwrite(str(args.out / f"{name}.jpg"), out_img)
        print(
            f"wrote {args.out / f'{name}.jpg'}  ({result.shape[1]}x{result.shape[0]})"
        )

        if not args.no_show and keep_going:
            keep_going = show_step(out_img)

    row0 = captioned["1.0_original"]
    row1 = hstack_captioned(captioned["1.1_resizing"], captioned["1.2_rotation"])
    row2 = hstack_captioned(captioned["1.3_affine"], captioned["1.4_perspective"])
    summary = vstack_center(row0, row1, row2)
    summary_path = args.out / "summary.jpg"
    cv2.imwrite(str(summary_path), summary)
    print(f"wrote {summary_path}  ({summary.shape[1]}x{summary.shape[0]})")

    if not args.no_show:
        show_step(summary, wait_hint=False)
        print("  (press any key to close)")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
