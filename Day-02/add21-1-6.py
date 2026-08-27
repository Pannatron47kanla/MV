"""
Mission 1-6: Using morphological transformations to cut out part of an image.

Produces (on model.jpg):
  6.0 -- Original Image
  6.1 -- cv2.threshold() to select only the white areas, rest made transparent
  6.2 -- Erode the mask before cutting -- result gets chipped/broken (แหว่ง)
  6.3 -- MORPH_OPEN (erode then dilate back) -- cleaner result, no chipping
  6.4 -- MORPH_GRADIENT -- a nice decorative outline of the shape

Each mask-based result is produced as a true transparent RGBA PNG (the real
deliverable -- open it to see the actual transparency), plus a "preview" JPG
where the RGBA is composited over a checkerboard so transparency is visible
in a flat screenshot, with a short red Thai/English caption overlaid on top.
A combined summary image (all preview panels) is also written.

Usage:
    python add21-1-6.py --image model.jpg --out morph_cutout_output
    python add21-1-6.py --thresh 150 --kernel 3
    python add21-1-6.py --no-show   # skip the interactive windows, just save files
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


def add_caption(img, text, font_size=18, pad=8):
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


# ---- 6.1 threshold to select white areas ----
def do_white_mask(gray, thresh=150):
    return cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]


# ---- 6.2 erode the mask ----
def do_erode(mask, kernel):
    return cv2.erode(mask, kernel)


# ---- 6.3 opening (erode then dilate back) ----
def do_open(mask, kernel):
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


# ---- 6.4 gradient (outline) ----
def do_gradient(mask, kernel):
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)


def to_rgba(bgr, mask):
    """Compose a transparent RGBA image: keeps bgr's colors where mask is
    nonzero (opaque), transparent everywhere else."""
    b, g, r = cv2.split(bgr)
    return cv2.merge([b, g, r, mask])


def checkerboard(w, h, cell=16):
    board = np.full((h, w, 3), 235, dtype=np.uint8)
    dark = 200
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                board[y : y + cell, x : x + cell] = dark
    return board

def composite_over_checkerboard(rgba):
    h, w = rgba.shape[:2]
    bg = checkerboard(w, h)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    fg = rgba[:, :, :3].astype(np.float32)
    out = fg * alpha + bg.astype(np.float32) * (1 - alpha)
    return out.astype(np.uint8)


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
WINDOW_NAME = "Morphological Cutout"


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
    ap.add_argument("--image", type=Path, default=SCRIPT_DIR / "model.jpg")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "morph_cutout_output")
    ap.add_argument("--thresh", type=int, default=90, help="white-selection threshold")
    ap.add_argument("--kernel", type=int, default=5, help="structuring element size (px)")
    ap.add_argument("--no-show", action="store_true", help="skip the interactive cv2 windows")
    args = ap.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"could not read image: {args.image}")
    img = resize_to_width(img, 450)  # keep output sizes manageable
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    args.out.mkdir(parents=True, exist_ok=True)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.kernel, args.kernel))

    white_mask = do_white_mask(gray, thresh=args.thresh)
    eroded_mask = do_erode(white_mask, kernel)
    opened_mask = do_open(white_mask, kernel)
    gradient_mask = do_gradient(white_mask, kernel)

    rgba_steps = [
        ("6.1_white_mask", to_rgba(img, white_mask), "6.1 -- Threshold: คัดเฉพาะส่วนสีขาว ส่วนที่เหลือโปร่งใส"),
        ("6.2_eroded", to_rgba(img, eroded_mask), "6.2 -- Erode mask ก่อนตัด: ภาพบางส่วนแหว่งหายไป"),
        ("6.3_opened", to_rgba(img, opened_mask), "6.3 -- MORPH_OPEN (กร่อนแล้วพองคืน): ผลลัพธ์สวยงามกว่า"),
        ("6.4_gradient", to_rgba(img, gradient_mask), "6.4 -- MORPH_GRADIENT: ได้เค้าโครง (outline) ของภาพ"),
    ]

    captioned = {"6.0_original": add_caption(img.copy(), "6.0 -- ภาพต้นฉบับ")}
    cv2.imwrite(str(args.out / "6.0_original.jpg"), captioned["6.0_original"])
    print(f"wrote {args.out / '6.0_original.jpg'}  ({img.shape[1]}x{img.shape[0]})")

    keep_going = True
    if not args.no_show:
        keep_going = show_step(captioned["6.0_original"])

    for name, rgba, caption in rgba_steps:
        png_path = args.out / f"{name}.png"
        cv2.imwrite(str(png_path), rgba)
        print(f"wrote {png_path}  ({rgba.shape[1]}x{rgba.shape[0]}, RGBA)")

        preview = add_caption(composite_over_checkerboard(rgba), caption)
        captioned[name] = preview
        preview_path = args.out / f"{name}_preview.jpg"
        cv2.imwrite(str(preview_path), preview)
        print(f"wrote {preview_path}  ({preview.shape[1]}x{preview.shape[0]})")

        if not args.no_show and keep_going:
            keep_going = show_step(preview)

    row0 = captioned["6.0_original"]
    row1 = hstack_captioned(captioned["6.1_white_mask"], captioned["6.2_eroded"])
    row2 = hstack_captioned(captioned["6.3_opened"], captioned["6.4_gradient"])
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
