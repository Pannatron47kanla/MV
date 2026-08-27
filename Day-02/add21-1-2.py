"""
Morphological Transformations demo on number8.png.

Produces:
  2.0 -- Original Image
  2.1 -- Erosion            (การกร่อน)
  2.2 -- Dilation            (การพองตัว)
  2.3 -- Opening              (การเปิดภาพ)
  2.4 -- Closing              (การปิดภาพ)
  2.5 -- Morphological Gradient
  2.6 -- Top Hat
  2.7 -- Black Hat

number8.png has a transparent background, so it's first composited onto
white, then thresholded (Otsu) into a binary mask (shape = white, 255;
background = black, 0) that the morphological ops run on.

Each saved image gets a short red Thai/English caption strip underneath.
cv2's own text drawing (Hershey fonts) can't render Thai glyphs and the
Thai-only Noto font is missing Latin/digit glyphs, so captions are rendered
character-by-character with Pillow, picking a Thai font for Thai code
points and a Latin font for everything else, then composited onto the
OpenCV image. A combined summary image (2.0 alone, 2.1|2.2, 2.3|2.4, 2.5
alone, 2.6|2.7 -- matching the report layout) is also written.

Usage:
    python add21-1-2.py --image number8.png --out morph_output
    python add21-1-2.py --no-show   # skip the interactive windows, just save files
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


def add_caption(img, text, font_size=20, pad=10):
    """Return a copy of img (BGR ndarray) with a white strip added below it
    and `text` drawn on the strip in red Thai+Latin-capable text, wrapped to
    fit the image width."""
    h, w = img.shape[:2]
    thai_font = _load_font(THAI_FONT_CANDIDATES, font_size)
    latin_font = _load_font(LATIN_FONT_CANDIDATES, font_size)

    max_width = w - 2 * pad
    lines = _wrap_text(text, thai_font, latin_font, max_width)

    ascent, descent = thai_font.getmetrics()
    line_step = ascent + descent + 6
    caption_height = pad + line_step * len(lines) + pad // 2

    canvas = Image.new("RGB", (w, h + caption_height), CAPTION_BG)
    canvas.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (0, 0))
    draw = ImageDraw.Draw(canvas)

    y = h + pad
    for line in lines:
        text_w = _mixed_width(line, thai_font, latin_font)
        x = max(pad, (w - text_w) // 2)
        _draw_mixed(draw, (x, y), line, thai_font, latin_font, RED)
        y += line_step

    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def resize_to_width(img, width):
    h, w = img.shape[:2]
    scale = width / w
    return cv2.resize(img, (width, int(h * scale)))


def load_binary_mask(path, width=400):
    """Load number8.png (has an alpha channel), composite it onto white for
    display, then threshold to a binary mask (shape=255, background=0) for
    the morphological ops to run on."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise SystemExit(f"could not read image: {path}")

    if raw.ndim == 3 and raw.shape[2] == 4:
        bgr = raw[:, :, :3].astype(np.float32)
        alpha = raw[:, :, 3:4].astype(np.float32) / 255.0
        white = np.full_like(bgr, 255.0)
        composited = (bgr * alpha + white * (1 - alpha)).astype(np.uint8)
    else:
        composited = raw[:, :, :3] if raw.ndim == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

    composited = resize_to_width(composited, width)
    gray = cv2.cvtColor(composited, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return composited, binary


def to_bgr(binary):
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


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
WINDOW_NAME = "Morphological Transformations"


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
    ap.add_argument("--image", type=Path, default=SCRIPT_DIR / "number8.png")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "morph_output")
    ap.add_argument("--kernel", type=int, default=35, help="structuring element size (px)")
    ap.add_argument("--no-show", action="store_true", help="skip the interactive cv2 windows")
    args = ap.parse_args()

    original, binary = load_binary_mask(args.image)
    args.out.mkdir(parents=True, exist_ok=True)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.kernel, args.kernel))

    erosion = cv2.erode(binary, kernel)
    dilation = cv2.dilate(binary, kernel)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)

    steps = [
        ("2.0_original", original, "2.0 -- ภาพต้นฉบับก่อนทำการแปลงใดๆ"),
        (
            "2.1_erosion",
            to_bgr(erosion),
            "2.1 -- Erosion (การกร่อน): กัดขอบวัตถุให้บางลง เส้น/จุดเล็กถูกกำจัด",
        ),
        (
            "2.2_dilation",
            to_bgr(dilation),
            "2.2 -- Dilation (การพองตัว): ขยายขอบวัตถุให้หนาขึ้น รูเล็กถูกเติมเต็ม",
        ),
        (
            "2.3_opening",
            to_bgr(opening),
            "2.3 -- Opening (การเปิดภาพ): erode แล้ว dilate ลบจุดรบกวนเล็กๆ ที่ยื่นออกมา",
        ),
        (
            "2.4_closing",
            to_bgr(closing),
            "2.4 -- Closing (การปิดภาพ): dilate แล้ว erode ปิดรู/ช่องว่างเล็กๆ ในวัตถุ",
        ),
        (
            "2.5_gradient",
            to_bgr(gradient),
            "2.5 -- Morphological Gradient: dilation ลบ erosion ได้เส้นขอบของวัตถุ",
        ),
        (
            "2.6_tophat",
            to_bgr(tophat),
            "2.6 -- Top Hat: ต้นฉบับ ลบ Opening เน้นส่วนที่นูน/สว่างเล็กกว่าโครงสร้าง",
        ),
        (
            "2.7_blackhat",
            to_bgr(blackhat),
            "2.7 -- Black Hat: Closing ลบต้นฉบับ เน้นส่วนที่เป็นรู/มืดเล็กกว่าโครงสร้าง",
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

    row0 = captioned["2.0_original"]
    row1 = hstack_captioned(captioned["2.1_erosion"], captioned["2.2_dilation"])
    row2 = hstack_captioned(captioned["2.3_opening"], captioned["2.4_closing"])
    row3 = captioned["2.5_gradient"]
    row4 = hstack_captioned(captioned["2.6_tophat"], captioned["2.7_blackhat"])
    summary = vstack_center(row0, row1, row2, row3, row4)
    summary_path = args.out / "summary.jpg"
    cv2.imwrite(str(summary_path), summary)
    print(f"wrote {summary_path}  ({summary.shape[1]}x{summary.shape[0]})")

    if not args.no_show:
        show_step(summary, wait_hint=False)
        print("  (press any key to close)")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
