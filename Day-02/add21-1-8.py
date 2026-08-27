"""
Motorcycle vs Car counting on street.mp4, classified purely by contour size.

Pipeline per frame:
  1. Mask (Gray Scale)          -- MOG2 foreground mask + light MORPH_OPEN
  2. Contour All Object (Green) -- every contour found on that mask, as-is
                                    (noisy: fragmented vehicle parts, specks)
  3. Contour Big Object (Green) -- mask further cleaned with a big
                                    MORPH_CLOSE (bridges gaps so each vehicle
                                    becomes one solid blob) then filtered to
                                    drop anything still noise-sized -- one
                                    clean contour per real object, motorcycle
                                    or car alike
  4. Result (Blue Box)          -- bounding boxes from step 3, classified as
                                    Motorcycle or Car purely by contour area
                                    at the moment it crosses a counting line
                                    (measured: motorcycle+rider ~9k-17k px^2,
                                    car/truck ~20k-85k px^2 at this camera
                                    distance -- clean gap around 20000),
                                    with a running count per class overlaid

A screenshot set (all 4 categories) is saved every time a NEW vehicle is
counted, named to match the report's figure numbering
(Fig-21.._Fig-24.._{Motercycle,Car}.{1,2}), until 2 examples of each class
are captured (16 images total, matching the "at least 16" requirement).

Usage:
    python add21-1-8.py --video street.mp4 --out traffic_count_output
    python add21-1-8.py --show   # preview live while processing
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RED = (255, 0, 0)  # RGB (Pillow draws in RGB)
GREEN_BGR = (0, 220, 0)
BLUE_BGR = (255, 60, 0)
LINE_BGR = (0, 200, 255)

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


def add_caption(img, text, font_size=16, pad=6):
    """Overlay `text` directly on top of img (BGR ndarray) in red, sitting
    on a semi-transparent dark bar at the bottom for legibility."""
    h, w = img.shape[:2]
    thai_font = _load_font(THAI_FONT_CANDIDATES, font_size)
    latin_font = _load_font(LATIN_FONT_CANDIDATES, font_size)

    max_width = w - 2 * pad
    lines = _wrap_text(text, thai_font, latin_font, max_width)

    ascent, descent = thai_font.getmetrics()
    line_step = ascent + descent + 5
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


def to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class Track:
    _next_id = 1

    def __init__(self, centroid):
        self.id = Track._next_id
        Track._next_id += 1
        self.centroid = centroid
        self.start_y = centroid[1]
        self.missed = 0
        self.counted = False


class ObjectCounter:
    """MOG2 background subtraction -> light opening -> (all contours) ->
    heavier closing + noise-floor filter -> (big/clean contours) ->
    nearest-centroid tracking -> line-crossing counting, classified as
    Motorcycle/Car by contour area."""

    def __init__(self, line_y, min_area=500, big_area_thresh=20000):
        self.line_y = line_y
        self.min_area = min_area
        self.big_area_thresh = big_area_thresh
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False
        )
        self.tracks = []
        self.moto_count = 0
        self.car_count = 0

    def warm_up(self, frame_bgr):
        self.bg_sub.apply(frame_bgr)

    def classify(self, area):
        return "Car" if area >= self.big_area_thresh else "Motorcycle"

    def process(self, frame_bgr):
        fg_mask = self.bg_sub.apply(frame_bgr)
        opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.open_kernel)

        all_contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.close_kernel)
        big_raw, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big_contours = [c for c in big_raw if cv2.contourArea(c) >= self.min_area]

        detections = []
        for c in big_contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            detections.append(((x + w // 2, y + h // 2), (x, y, w, h), area))

        self._update_tracks(detections)
        events = self._count_crossings()

        return opened, all_contours, big_contours, detections, events

    def _update_tracks(self, detections):
        unmatched = list(range(len(detections)))
        for track in self.tracks:
            best_i, best_dist = None, 90
            for i in unmatched:
                cx, cy = detections[i][0]
                dx = cx - track.centroid[0]
                dy = cy - track.centroid[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist, best_i = dist, i
            if best_i is not None:
                track.centroid = detections[best_i][0]
                track.area = detections[best_i][2]
                track.missed = 0
                unmatched.remove(best_i)
            else:
                track.missed += 1

        for i in unmatched:
            t = Track(detections[i][0])
            t.area = detections[i][2]
            self.tracks.append(t)

        self.tracks = [t for t in self.tracks if t.missed <= 8]

    def _count_crossings(self):
        events = []
        for t in self.tracks:
            if t.counted:
                continue
            if t.start_y > self.line_y >= t.centroid[1]:
                t.counted = True
                cls = self.classify(t.area)
                if cls == "Car":
                    self.car_count += 1
                else:
                    self.moto_count += 1
                events.append(cls)
        return events

    def draw_all_contours(self, frame_bgr, all_contours):
        out = frame_bgr.copy()
        cv2.drawContours(out, all_contours, -1, GREEN_BGR, 1)
        cv2.line(out, (0, self.line_y), (out.shape[1], self.line_y), LINE_BGR, 1)
        return out

    def draw_big_contours(self, frame_bgr, big_contours):
        out = frame_bgr.copy()
        cv2.drawContours(out, big_contours, -1, GREEN_BGR, 2)
        cv2.line(out, (0, self.line_y), (out.shape[1], self.line_y), LINE_BGR, 1)
        return out

    def draw_result(self, frame_bgr, detections):
        out = frame_bgr.copy()
        cv2.line(out, (0, self.line_y), (out.shape[1], self.line_y), LINE_BGR, 2)
        for _, (x, y, w, h), area in detections:
            cls = self.classify(area)
            cv2.rectangle(out, (x, y), (x + w, y + h), BLUE_BGR, 2)
            cv2.putText(
                out, cls, (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE_BGR, 2, cv2.LINE_AA,
            )
        cv2.rectangle(out, (0, 0), (260, 60), (0, 0, 0), -1)
        cv2.putText(
            out, f"Motorcycle: {self.moto_count}", (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            out, f"Car: {self.car_count}", (8, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        return out


SCRIPT_DIR = Path(__file__).resolve().parent
WINDOW_NAME = "Motorcycle vs Car Counting"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, default=SCRIPT_DIR / "street.mp4")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "traffic_count_output")
    ap.add_argument("--warm-up", type=int, default=1400, help="frames to warm up the background model on")
    ap.add_argument("--max-frames", type=int, default=2400, help="frames to process after warm-up")
    ap.add_argument("--examples-per-class", type=int, default=2, help="screenshot sets to save per class")
    ap.add_argument("--show", action="store_true", help="preview live while processing")
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {args.video}")

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(height * 0.52)

    args.out.mkdir(parents=True, exist_ok=True)
    counter = ObjectCounter(line_y)

    print(f"warming up background model on {args.warm_up} frames...")
    for _ in range(args.warm_up):
        ret, frame = cap.read()
        if not ret:
            break
        counter.warm_up(frame)

    fig_names = {
        "mask": "Fig-21",
        "all": "Fig-22",
        "big": "Fig-23",
        "result": "Fig-24",
    }
    # keep the assignment's exact (mis)spelling so filenames match the report
    class_label = {"Motorcycle": "Motercycle", "Car": "Car"}
    saved = {"Motorcycle": 0, "Car": 0}

    def save_set(cls, opened, all_c, big_c, result_img, frame_bgr):
        idx = saved[cls] + 1
        label = class_label[cls]
        tag = f"{cls} #{sum(saved.values()) + 1}"

        mask_img = add_caption(to_bgr(opened), f"1. Mask (Gray Scale) -- {tag}")
        all_img = add_caption(counter.draw_all_contours(frame_bgr, all_c), f"2. Contour All Object (Green) -- {tag}")
        big_img = add_caption(counter.draw_big_contours(frame_bgr, big_c), f"3. Contour Big Object (Green) -- {tag}")
        result_img = add_caption(result_img, f"4. Result (Blue Box) with counting number -- {tag}")

        cv2.imwrite(str(args.out / f"{fig_names['mask']}_{label}.{idx}.jpg"), mask_img)
        cv2.imwrite(str(args.out / f"{fig_names['all']}_{label}.{idx}.jpg"), all_img)
        cv2.imwrite(str(args.out / f"{fig_names['big']}_{label}.{idx}.jpg"), big_img)
        cv2.imwrite(str(args.out / f"{fig_names['result']}_{label}.{idx}.jpg"), result_img)
        print(f"  captured {label}.{idx}  (Motorcycle={counter.moto_count} Car={counter.car_count})")
        saved[cls] += 1

    frame_idx = 0
    while frame_idx < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        opened, all_c, big_c, detections, events = counter.process(frame)
        result_img = counter.draw_result(frame, detections)

        for cls in events:
            if saved[cls] < args.examples_per_class:
                save_set(cls, opened, all_c, big_c, result_img, frame)

        if args.show:
            cv2.imshow(WINDOW_NAME, result_img)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

        if all(saved[c] >= args.examples_per_class for c in saved):
            print(f"stopping early at frame {frame_idx}: got {args.examples_per_class} examples of each class")
            break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"\nprocessed {frame_idx} frames after warm-up")
    print(f"Motorcycle total count: {counter.moto_count}  ({saved['Motorcycle']} example sets saved)")
    print(f"Car total count: {counter.car_count}  ({saved['Car']} example sets saved)")
    total_images = sum(saved.values()) * 4
    print(f"Total screenshots saved: {total_images}")
    if total_images < 16:
        print("WARNING: fewer than 16 images captured -- try increasing --max-frames")


if __name__ == "__main__":
    main()
