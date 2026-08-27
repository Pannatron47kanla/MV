"""
Thresholding and Image Gradients demo on sudoku.png.

Produces:
  4.0 -- Original Image
  4.1 -- Simple Thresholding  {Binary, Binary Inverse, Trunc, ToZero, ToZero Inverse}
  4.2 -- Adaptive Thresholding  {Mean, Gaussian}
  4.3 -- Sobel Derivative  {Sobel X, Sobel Y}
  4.4 -- Laplacian Derivative

Each saved image gets a short red Thai/English caption overlaid directly on
top of it (on a semi-transparent dark bar for legibility). cv2's own text
drawing (Hershey fonts) can't render Thai glyphs and the Thai-only Noto font
is missing Latin/digit glyphs, so captions are rendered character-by-
character with Pillow, picking a Thai font for Thai code points and a Latin
font for everything else, then composited onto the OpenCV image. A combined
summary image (matching the report layout) is also written.

Usage:
    python add21-1-4.py --image sudoku.png --out threshold_derivative_output
    python add21-1-4.py --no-show   # skip the interactive windows, just save files
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


def add_caption(img, text, font_size=18, pad=6):
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


def add_title_bar(img, text, font_size=20, pad=10):
    """Add a plain white strip with red title text ABOVE img (used to label
    a multi-tile grid as one combined frame, as opposed to add_caption's
    per-image overlay)."""
    h, w = img.shape[:2]
    thai_font = _load_font(THAI_FONT_CANDIDATES, font_size)
    latin_font = _load_font(LATIN_FONT_CANDIDATES, font_size)

    max_width = w - 2 * pad
    lines = _wrap_text(text, thai_font, latin_font, max_width)

    ascent, descent = thai_font.getmetrics()
    line_step = ascent + descent + 6
    bar_height = pad + line_step * len(lines) + pad // 2

    canvas = Image.new("RGB", (w, h + bar_height), CAPTION_BG)
    draw = ImageDraw.Draw(canvas)
    y = pad // 2
    for line in lines:
        text_w = _mixed_width(line, thai_font, latin_font)
        x = max(pad, (w - text_w) // 2)
        _draw_mixed(draw, (x, y), line, thai_font, latin_font, RED)
        y += line_step

    canvas.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (0, bar_height))
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def resize_to_width(img, width):
    h, w = img.shape[:2]
    scale = width / w
    return cv2.resize(img, (width, int(h * scale)))


def to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ---- 4.1 simple thresholding ----
def do_simple_thresholds(gray, thresh=127):
    modes = [
        ("Binary", cv2.THRESH_BINARY),
        ("Binary Inverse", cv2.THRESH_BINARY_INV),
        ("Trunc", cv2.THRESH_TRUNC),
        ("ToZero", cv2.THRESH_TOZERO),
        ("ToZero Inverse", cv2.THRESH_TOZERO_INV),
    ]
    return {name: cv2.threshold(gray, thresh, 255, mode)[1] for name, mode in modes}


# ---- 4.2 adaptive thresholding ----
def do_adaptive_thresholds(gray, block_size=11, c=2):
    mean = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c
    )
    gaussian = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )
    return {"Mean": mean, "Gaussian": gaussian}


# ---- 4.3 sobel derivative ----
def do_sobel(gray, ksize=3):
    sobel_x = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize))
    sobel_y = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize))
    return {"Sobel X": sobel_x, "Sobel Y": sobel_y}


# ---- 4.4 laplacian derivative ----
def do_laplacian(gray):
    return cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))


def hstack_captioned(*imgs):
    """Stack captioned images side by side, padding shorter ones to match height."""
    h = max(img.shape[0] for img in imgs)
    padded = []
    for img in imgs:
        ih, iw = img.shape[:2]
        canvas = np.full((h, iw, 3), CAPTION_BG, dtype=np.uint8)
        canvas[:ih] = img
        padded.append(canvas)
    gap = np.full((h, 10, 3), CAPTION_BG, dtype=np.uint8)
    out = padded[0]
    for p in padded[1:]:
        out = np.hstack([out, gap, p])
    return out


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
WINDOW_NAME = "Thresholding and Gradients"


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
    ap.add_argument("--image", type=Path, default=SCRIPT_DIR / "sudoku.png")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "threshold_derivative_output")
    ap.add_argument("--thresh", type=int, default=127, help="simple threshold value")
    ap.add_argument("--no-show", action="store_true", help="skip the interactive cv2 windows")
    args = ap.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"could not read image: {args.image}")
    img = resize_to_width(img, 400)  # keep output sizes manageable
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    args.out.mkdir(parents=True, exist_ok=True)

    simple = do_simple_thresholds(gray, thresh=args.thresh)
    adaptive = do_adaptive_thresholds(gray)
    sobel = do_sobel(gray)
    laplacian = do_laplacian(gray)

    # ---- 4.1: combine all 5 simple-threshold variants into one frame ----
    tile_width = 240
    tile_labels = [
        ("Binary", simple["Binary"]),
        ("Binary Inverse", simple["Binary Inverse"]),
        ("Trunc", simple["Trunc"]),
        ("ToZero", simple["ToZero"]),
        ("ToZero Inverse", simple["ToZero Inverse"]),
    ]
    tiles = {
        label: add_caption(resize_to_width(to_bgr(mat), tile_width), label, font_size=14, pad=4)
        for label, mat in tile_labels
    }
    grid_row1 = hstack_captioned(tiles["Binary"], tiles["Binary Inverse"], tiles["Trunc"])
    grid_row2 = hstack_captioned(tiles["ToZero"], tiles["ToZero Inverse"])
    simple_grid = vstack_center(grid_row1, grid_row2)
    simple_combined = add_title_bar(
        simple_grid,
        f"4.1 -- Simple Thresholding (threshold={args.thresh}): "
        "Binary / Binary Inverse / Trunc / ToZero / ToZero Inverse",
    )

    steps = [
        ("4.0_original", img.copy(), "4.0 -- ภาพต้นฉบับ"),
        (
            "4.2_adaptive_mean",
            to_bgr(adaptive["Mean"]),
            "4.2 -- Adaptive Mean: threshold จากค่าเฉลี่ยของพื้นที่ใกล้เคียง",
        ),
        (
            "4.2_adaptive_gaussian",
            to_bgr(adaptive["Gaussian"]),
            "4.2 -- Adaptive Gaussian: threshold แบบถ่วงน้ำหนัก Gaussian รอบพิกเซล",
        ),
        (
            "4.3_sobel_x",
            to_bgr(sobel["Sobel X"]),
            "4.3 -- Sobel X: อนุพันธ์ตามแกน x เน้นขอบแนวตั้ง",
        ),
        (
            "4.3_sobel_y",
            to_bgr(sobel["Sobel Y"]),
            "4.3 -- Sobel Y: อนุพันธ์ตามแกน y เน้นขอบแนวนอน",
        ),
        (
            "4.4_laplacian",
            to_bgr(laplacian),
            "4.4 -- Laplacian: อนุพันธ์อันดับสอง เน้นขอบทุกทิศทาง",
        ),
    ]

    captioned = {}
    keep_going = True

    # 4.1 is already a fully-captioned combined frame -- write/show it directly
    cv2.imwrite(str(args.out / "4.1_simple_threshold.jpg"), simple_combined)
    print(
        f"wrote {args.out / '4.1_simple_threshold.jpg'}  "
        f"({simple_combined.shape[1]}x{simple_combined.shape[0]})"
    )
    if not args.no_show:
        keep_going = show_step(simple_combined)

    for name, result, caption in steps:
        out_img = add_caption(result, caption)
        captioned[name] = out_img
        cv2.imwrite(str(args.out / f"{name}.jpg"), out_img)
        print(f"wrote {args.out / f'{name}.jpg'}  ({result.shape[1]}x{result.shape[0]})")

        if not args.no_show and keep_going:
            keep_going = show_step(out_img)

    row0 = captioned["4.0_original"]
    row1 = simple_combined
    row3 = hstack_captioned(captioned["4.2_adaptive_mean"], captioned["4.2_adaptive_gaussian"])
    row4 = hstack_captioned(captioned["4.3_sobel_x"], captioned["4.3_sobel_y"])
    row5 = captioned["4.4_laplacian"]
    summary = vstack_center(row0, row1, row3, row4, row5)
    summary_path = args.out / "summary.jpg"
    cv2.imwrite(str(summary_path), summary)
    print(f"wrote {summary_path}  ({summary.shape[1]}x{summary.shape[0]})")

    if not args.no_show:
        show_step(summary, wait_hint=False)
        print("  (press any key to close)")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
