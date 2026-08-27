"""
Gamma Correction demo on mars.jpg.

Produces:
  5.0 -- Original Image
  5.1 -- Gamma Correction
  5.2 -- Histogram of Input/Output Image

Gamma correction remaps pixel intensities with output = 255*(input/255)^(1/gamma)
via a lookup table (cv2.LUT). gamma > 1 brightens (mars.jpg is quite dark,
mean intensity ~60/255, so it benefits from brightening); gamma < 1 darkens.

The histogram in 5.2 is drawn manually with Pillow (line plot of the two
grayscale histograms overlaid) rather than matplotlib, to avoid pulling in
an extra dependency -- this project's other scripts also stick to
cv2/numpy/Pillow only.

Each photo gets a short red Thai/English caption overlaid directly on top
(semi-transparent dark bar for legibility); the histogram plot gets a red
title bar above it. A combined summary image is also written.

Usage:
    python add21-1-5.py --image mars.jpg --out gamma_output
    python add21-1-5.py --gamma 2.2   # tweak brightening strength
    python add21-1-5.py --no-show     # skip the interactive windows, just save files
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


def add_title_bar(img, text, font_size=20, pad=10):
    """Add a plain white strip with red title text ABOVE img (used to label
    a standalone diagram, e.g. the histogram plot)."""
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


# ---- 5.1 gamma correction ----
def do_gamma_correction(img, gamma=2.2):
    """output = 255 * (input/255)^(1/gamma), applied via a lookup table.
    gamma > 1 brightens, gamma < 1 darkens."""
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype(np.uint8)
    return cv2.LUT(img, table)


# ---- 5.2 histogram of input/output ----
def draw_histogram_plot(hist_in, hist_out, width=460, height=300):
    """Draw a simple line-plot comparing two grayscale histograms with
    Pillow (axes, ticks, legend) -- no matplotlib dependency needed.
    Frequencies are log-scaled (log1p) since mars.jpg has a large black
    background that spikes intensity 0 and would otherwise flatten out
    the rest of the distribution."""
    pad_left, pad_right, pad_top, pad_bottom = 55, 20, 45, 45
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    hist_in = np.log1p(hist_in)
    hist_out = np.log1p(hist_out)
    max_val = max(hist_in.max(), hist_out.max(), 1)

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    x0, y0 = pad_left, pad_top
    x1, y1 = pad_left + plot_w, pad_top + plot_h
    draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=1)

    def to_point(i, v):
        x = x0 + (i / 255) * plot_w
        y = y1 - (v / max_val) * plot_h
        return (x, y)

    draw.line([to_point(i, hist_in[i]) for i in range(256)], fill=(30, 110, 220), width=2)
    draw.line([to_point(i, hist_out[i]) for i in range(256)], fill=(230, 120, 20), width=2)

    tick_font = _load_font(LATIN_FONT_CANDIDATES, 12)
    for xi in (0, 64, 128, 192, 255):
        x, _ = to_point(xi, 0)
        draw.line([x, y1, x, y1 + 4], fill=(0, 0, 0))
        draw.text((x - 8, y1 + 6), str(xi), font=tick_font, fill=(0, 0, 0))
    draw.text((x0, y1 + 24), "Pixel Intensity", font=tick_font, fill=(0, 0, 0))
    draw.text((5, y0 - 5), "Freq (log)", font=tick_font, fill=(0, 0, 0))

    legend_font = _load_font(LATIN_FONT_CANDIDATES, 13)
    draw.line([x0 + 10, 12, x0 + 30, 12], fill=(30, 110, 220), width=3)
    draw.text((x0 + 35, 5), "Input (Original)", font=legend_font, fill=(0, 0, 0))
    draw.line([x0 + 10, 30, x0 + 30, 30], fill=(230, 120, 20), width=3)
    draw.text((x0 + 35, 23), "Output (Gamma Corrected)", font=legend_font, fill=(0, 0, 0))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


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
WINDOW_NAME = "Gamma Correction"


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
    ap.add_argument("--image", type=Path, default=SCRIPT_DIR / "mars.jpg")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "gamma_output")
    ap.add_argument("--gamma", type=float, default=2.2, help="gamma value (>1 brightens, <1 darkens)")
    ap.add_argument("--no-show", action="store_true", help="skip the interactive cv2 windows")
    args = ap.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"could not read image: {args.image}")
    img = resize_to_width(img, 450)  # keep output sizes manageable
    args.out.mkdir(parents=True, exist_ok=True)

    gamma_img = do_gamma_correction(img, gamma=args.gamma)

    gray_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_out = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2GRAY)
    hist_in = cv2.calcHist([gray_in], [0], None, [256], [0, 256]).flatten()
    hist_out = cv2.calcHist([gray_out], [0], None, [256], [0, 256]).flatten()
    hist_plot = draw_histogram_plot(hist_in, hist_out)
    hist_combined = add_title_bar(
        hist_plot, f"5.2 -- Histogram: Input vs Output (gamma={args.gamma})"
    )

    steps = [
        ("5.0_original", img.copy(), "5.0 -- ภาพต้นฉบับ"),
        (
            "5.1_gamma_correction",
            gamma_img,
            f"5.1 -- Gamma Correction (gamma={args.gamma}): ปรับความสว่างด้วย LUT "
            "output = 255*(input/255)^(1/gamma)",
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

    cv2.imwrite(str(args.out / "5.2_histogram.jpg"), hist_combined)
    print(
        f"wrote {args.out / '5.2_histogram.jpg'}  "
        f"({hist_combined.shape[1]}x{hist_combined.shape[0]})"
    )
    if not args.no_show and keep_going:
        keep_going = show_step(hist_combined)

    row0 = hstack_captioned(captioned["5.0_original"], captioned["5.1_gamma_correction"])
    row1 = hist_combined
    summary = vstack_center(row0, row1)
    summary_path = args.out / "summary.jpg"
    cv2.imwrite(str(summary_path), summary)
    print(f"wrote {summary_path}  ({summary.shape[1]}x{summary.shape[0]})")

    if not args.no_show:
        show_step(summary, wait_hint=False)
        print("  (press any key to close)")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
