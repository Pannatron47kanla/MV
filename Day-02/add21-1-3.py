"""
Image Filtering (Smoothing) demo on add21-1-3.jpg.

Produces:
  3.0 -- Original Image
  3.1 -- 2-D Convolution (Image Filtering)
  3.2 -- Average Blurring
  3.3 -- Gaussian Blurring
  3.4 -- Median Blur
  3.5 -- Bilateral Filter

Each saved image gets a short red Thai/English caption strip underneath.
cv2's own text drawing (Hershey fonts) can't render Thai glyphs and the
Thai-only Noto font is missing Latin/digit glyphs, so captions are rendered
character-by-character with Pillow, picking a Thai font for Thai code
points and a Latin font for everything else, then composited onto the
OpenCV image. A combined summary image (3.0 alone, 3.1|3.2, 3.3|3.4, 3.5
alone) is also written.

Usage:
    python add21-1-3.py --image add21-1-3.jpg --out filter_output
    python add21-1-3.py --no-show   # skip the interactive windows, just save files
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
]
LATIN_FONT_CANDIDATES = [
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]


def _load_font(candidates, size):
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _is_thai(ch):
    return "฀" <= ch <= "๿"


def _runs(text):
    """Split text into consecutive runs of (is_thai, substring)."""
    if not text:
        return []
    runs = []
    cur_is_thai, cur = _is_thai(text[0]), text[0]
    for ch in text[1:]:
        if _is_thai(ch) == cur_is_thai:
            cur += ch
        else:
            runs.append((cur_is_thai, cur))
            cur_is_thai, cur = _is_thai(ch), ch
    runs.append((cur_is_thai, cur))
    return runs


def _mixed_width(text, thai_font, latin_font):
    return sum(
        (thai_font if is_thai else latin_font).getlength(run)
        for is_thai, run in _runs(text)
    )


def _draw_mixed(draw, xy, text, thai_font, latin_font, fill):
    x, y = xy
    for is_thai, run in _runs(text):
        font = thai_font if is_thai else latin_font
        draw.text((x, y), run, font=font, fill=fill)
        x += font.getlength(run)


def _wrap_text(text, thai_font, latin_font, max_width):
    """Greedily wrap `text` into lines that each fit within max_width pixels."""
    words = text.split(" ")
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if _mixed_width(candidate, thai_font, latin_font) > max_width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def add_caption(img, text, font_size=20, pad=8):
    """Return a copy of img (BGR ndarray, same size) with `text` overlaid
    directly on top of the image in red Thai+Latin-capable text, sitting on
    a semi-transparent dark bar at the bottom for legibility."""
    h, w = img.shape[:2]
    thai_font = _load_font(THAI_FONT_CANDIDATES, font_size)
    latin_font = _load_font(LATIN_FONT_CANDIDATES, font_size)

    max_width = w - 2 * pad
    lines = _wrap_text(text, thai_font, latin_font, max_width)

    ascent, descent = thai_font.getmetrics()
    line_step = ascent + descent + 6
    bar_height = min(h, pad + line_step * len(lines) + pad // 2)

    base = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([0, h - bar_height, w, h], fill=(0, 0, 0, 150))

    y = h - bar_height + pad // 2
    for line in lines:
        text_w = _mixed_width(line, thai_font, latin_font)
        x = max(pad, (w - text_w) // 2)
        _draw_mixed(draw, (x, y), line, thai_font, latin_font, RED + (255,))
        y += line_step

    combined = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)


def resize_to_width(img, width):
    h, w = img.shape[:2]
    scale = width / w
    return cv2.resize(img, (width, int(h * scale)))


# ---- 3.1 2-D convolution (image filtering) ----
def do_convolution(img, ksize=5):
    """Apply a normalized box-averaging kernel via cv2.filter2D -- the
    generic 2-D convolution operation that cv2.blur/GaussianBlur/etc. all
    build on top of."""
    kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    return cv2.filter2D(img, -1, kernel)


# ---- 3.2 average blurring ----
def do_average_blur(img, ksize=15):
    return cv2.blur(img, (ksize, ksize))


# ---- 3.3 gaussian blurring ----
def do_gaussian_blur(img, ksize=15, sigma=0):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


# ---- 3.4 median blur ----
def do_median_blur(img, ksize=15):
    return cv2.medianBlur(img, ksize)


# ---- 3.5 bilateral filter ----
def do_bilateral_filter(img, d=15, sigma_color=80, sigma_space=80):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


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
WINDOW_NAME = "Image Filtering"


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
    ap.add_argument("--image", type=Path, default=SCRIPT_DIR / "add21-1-3.jpg")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "filter_output")
    ap.add_argument("--ksize", type=int, default=15, help="kernel size (px, odd)")
    ap.add_argument("--no-show", action="store_true", help="skip the interactive cv2 windows")
    args = ap.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"could not read image: {args.image}")
    img = resize_to_width(img, 500)  # keep output sizes manageable
    args.out.mkdir(parents=True, exist_ok=True)

    steps = [
        ("3.0_original", img.copy(), "3.0 -- ภาพต้นฉบับ"),
        (
            "3.1_convolution",
            do_convolution(img, ksize=5),
            "3.1 -- 2-D Convolution: filter2D กวาดเคอร์เนล 5x5 ทั่วภาพ",
        ),
        (
            "3.2_average_blur",
            do_average_blur(img, ksize=args.ksize),
            f"3.2 -- Average Blurring: หาค่าเฉลี่ยพิกเซลรอบข้าง {args.ksize}x{args.ksize}",
        ),
        (
            "3.3_gaussian_blur",
            do_gaussian_blur(img, ksize=args.ksize),
            f"3.3 -- Gaussian Blurring: ถ่วงน้ำหนักแบบ Gaussian {args.ksize}x{args.ksize}",
        ),
        (
            "3.4_median_blur",
            do_median_blur(img, ksize=args.ksize),
            f"3.4 -- Median Blur: ใช้ค่ามัธยฐานในหน้าต่าง {args.ksize}x{args.ksize} ลด noise รักษาขอบภาพ",
        ),
        (
            "3.5_bilateral",
            do_bilateral_filter(img),
            "3.5 -- Bilateral Filter: ลด noise พร้อมรักษาขอบภาพไว้ชัดเจน",
        ),
    ]

    captioned = {}
    keep_going = True
    for name, result, caption in steps:
        out_img = add_caption(result, caption)
        captioned[name] = out_img
        cv2.imwrite(str(args.out / f"{name}.jpg"), out_img)
        print(f"wrote {args.out / f'{name}.jpg'}  ({result.shape[1]}x{result.shape[0]})")

        if not args.no_show and keep_going:
            keep_going = show_step(out_img)

    row0 = captioned["3.0_original"]
    row1 = hstack_captioned(captioned["3.1_convolution"], captioned["3.2_average_blur"])
    row2 = hstack_captioned(captioned["3.3_gaussian_blur"], captioned["3.4_median_blur"])
    row3 = captioned["3.5_bilateral"]
    summary = vstack_center(row0, row1, row2, row3)
    summary_path = args.out / "summary.jpg"
    cv2.imwrite(str(summary_path), summary)
    print(f"wrote {summary_path}  ({summary.shape[1]}x{summary.shape[0]})")

    if not args.no_show:
        show_step(summary, wait_hint=False)
        print("  (press any key to close)")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
