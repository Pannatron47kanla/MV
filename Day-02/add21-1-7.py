"""
Mission 1A/1B: Incoming / Outgoing car counting on car.mp4.

The video is a fixed overhead highway shot: the LEFT carriageway carries
traffic driving TOWARD the camera (Incoming, growing bigger / moving down
the frame), the RIGHT carriageway carries traffic driving AWAY from the
camera (Outgoing, shrinking / moving up the frame). Each side is processed
independently (its own background subtractor, its own vehicle counter) and
captures its own screenshots:

  1. Opening Image (Gray Scale)  -- background-subtracted mask after
     cv2.morphologyEx(..., MORPH_OPEN, ...) to clean noise
  2. Contour Image (Green)       -- valid vehicle contours drawn in green
  3. Result (Blue Box)           -- bounding boxes in blue + running count

A screenshot triple is captured every time a new vehicle is counted (crosses
the counting line), so with several vehicles per lane you comfortably clear
the "at least 9 images" requirement per mission (3 categories x N events).

Usage:
    python add21-1-7.py --video car.mp4 --out car_count_output
    python add21-1-7.py --max-frames 3600   # ~2 min of video at 30fps
    python add21-1-7.py --show              # preview live while processing
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


class LaneCounter:
    """Background subtraction + morphological opening + contour detection +
    lightweight nearest-centroid tracking + line-crossing counting for one
    lane (one half of the frame)."""

    def __init__(self, name, direction, line_y, roi_mask, min_area=140, kernel_size=(9, 9)):
        self.name = name
        self.direction = direction  # "down" (incoming) or "up" (outgoing)
        self.line_y = line_y
        self.roi_mask = roi_mask  # zeroes out sky/mountains/off-road areas
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False
        )
        self.tracks = []
        self.count = 0

    def warm_up(self, roi_bgr):
        """Feed a frame to the subtractor without detecting -- MOG2 needs a
        few dozen frames before its background model stabilizes, otherwise
        early frames spuriously flag sky/treeline contrast as foreground."""
        self.bg_sub.apply(roi_bgr)

    def process(self, roi_bgr):
        fg_mask = self.bg_sub.apply(roi_bgr)
        fg_mask = cv2.bitwise_and(fg_mask, self.roi_mask)
        opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) >= self.min_area]

        detections = []
        for c in valid:
            x, y, w, h = cv2.boundingRect(c)
            detections.append(((x + w // 2, y + h // 2), (x, y, w, h)))

        self._update_tracks(detections)
        newly_counted = self._count_crossings()

        return opened, valid, detections, newly_counted

    def _update_tracks(self, detections):
        unmatched = list(range(len(detections)))
        for track in self.tracks:
            best_i, best_dist = None, 70
            for i in unmatched:
                cx, cy = detections[i][0]
                dx = cx - track.centroid[0]
                dy = cy - track.centroid[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist, best_i = dist, i
            if best_i is not None:
                track.centroid = detections[best_i][0]
                track.missed = 0
                unmatched.remove(best_i)
            else:
                track.missed += 1

        for i in unmatched:
            self.tracks.append(Track(detections[i][0]))

        self.tracks = [t for t in self.tracks if t.missed <= 8]

    def _count_crossings(self):
        newly_counted = 0
        for t in self.tracks:
            if t.counted:
                continue
            crossed = (
                self.direction == "down" and t.start_y < self.line_y <= t.centroid[1]
            ) or (
                self.direction == "up" and t.start_y > self.line_y >= t.centroid[1]
            )
            if crossed:
                t.counted = True
                self.count += 1
                newly_counted += 1
        return newly_counted

    def draw_contours(self, roi_bgr, valid_contours):
        out = roi_bgr.copy()
        cv2.drawContours(out, valid_contours, -1, GREEN_BGR, 2)
        cv2.line(out, (0, self.line_y), (out.shape[1], self.line_y), LINE_BGR, 1)
        return out

    def draw_result(self, roi_bgr, detections):
        out = roi_bgr.copy()
        cv2.line(out, (0, self.line_y), (out.shape[1], self.line_y), LINE_BGR, 2)
        for _, (x, y, w, h) in detections:
            cv2.rectangle(out, (x, y), (x + w, y + h), BLUE_BGR, 2)
        cv2.rectangle(out, (0, 0), (230, 34), (0, 0, 0), -1)
        cv2.putText(
            out, f"{self.name}: {self.count}", (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
        return out


SCRIPT_DIR = Path(__file__).resolve().parent
WINDOW_NAME = "Car Counting"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, default=SCRIPT_DIR / "car.mp4")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "car_count_output")
    ap.add_argument("--max-frames", type=int, default=3600, help="frames to process (~2min @30fps)")
    ap.add_argument("--min-events", type=int, default=5, help="stop early once each side has this many counted vehicles")
    ap.add_argument("--show", action="store_true", help="preview live while processing")
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    split_x = width // 2
    line_y = int(height * 0.55)

    # Exclude sky/mountains (contrast noise MOG2 can flag as foreground) and
    # the gas-station/roundabout lot in the lower-left background (moving
    # vehicles there aren't highway traffic) so only the actual road surface
    # can produce detections.
    roi_mask_full = np.full((height, width), 255, dtype=np.uint8)
    roi_mask_full[: int(height * 0.31), :] = 0
    roi_mask_full[int(height * 0.29) : int(height * 0.51), : int(width * 0.32)] = 0
    roi_mask_left = roi_mask_full[:, :split_x]
    roi_mask_right = roi_mask_full[:, split_x:]

    incoming_dir = args.out / "incoming"
    outgoing_dir = args.out / "outgoing"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    outgoing_dir.mkdir(parents=True, exist_ok=True)

    incoming = LaneCounter("Incoming", "down", line_y, roi_mask_left)
    outgoing = LaneCounter("Outgoing", "up", line_y, roi_mask_right)

    warm_up_frames = 90
    print(f"warming up background model on {warm_up_frames} frames...")
    for _ in range(warm_up_frames):
        ret, frame = cap.read()
        if not ret:
            break
        incoming.warm_up(frame[:, :split_x])
        outgoing.warm_up(frame[:, split_x:])

    def save_triple(out_dir, prefix, tag_th, roi_opened, roi_contour, roi_result, event_idx):
        opening_img = add_caption(to_bgr(roi_opened), f"{tag_th} 1. Opening Image (Gray Scale)")
        contour_img = add_caption(roi_contour, f"{tag_th} 2. Contour Image (Green)")
        result_img = add_caption(roi_result, f"{tag_th} 3. Result (Blue Box) with counting number")
        cv2.imwrite(str(out_dir / f"{prefix}_{event_idx:02d}_1_opening.jpg"), opening_img)
        cv2.imwrite(str(out_dir / f"{prefix}_{event_idx:02d}_2_contour.jpg"), contour_img)
        cv2.imwrite(str(out_dir / f"{prefix}_{event_idx:02d}_3_result.jpg"), result_img)
        print(f"  captured {prefix} event {event_idx}: opening/contour/result -> {out_dir}")

    frame_idx = 0
    in_events = 0
    out_events = 0
    while frame_idx < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        left_roi = frame[:, :split_x]
        right_roi = frame[:, split_x:]

        in_opened, in_valid, in_dets, in_new = incoming.process(left_roi)
        out_opened, out_valid, out_dets, out_new = outgoing.process(right_roi)

        in_contour_img = incoming.draw_contours(left_roi, in_valid)
        in_result_img = incoming.draw_result(left_roi, in_dets)
        out_contour_img = outgoing.draw_contours(right_roi, out_valid)
        out_result_img = outgoing.draw_result(right_roi, out_dets)

        if in_new and in_events < args.min_events:
            in_events += 1
            save_triple(
                incoming_dir, "incoming", "1A -- Incoming:",
                in_opened, in_contour_img, in_result_img, in_events,
            )
        if out_new and out_events < args.min_events:
            out_events += 1
            save_triple(
                outgoing_dir, "outgoing", "1B -- Outgoing:",
                out_opened, out_contour_img, out_result_img, out_events,
            )

        if args.show:
            preview = np.hstack([in_result_img, out_result_img])
            cv2.imshow(WINDOW_NAME, preview)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

        if in_events >= args.min_events and out_events >= args.min_events:
            print(f"stopping early at frame {frame_idx}: both sides reached {args.min_events} events")
            break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"\nprocessed {frame_idx} frames")
    print(f"Incoming total count: {incoming.count}  ({in_events} screenshot events saved)")
    print(f"Outgoing total count: {outgoing.count}  ({out_events} screenshot events saved)")
    if in_events < 3 or out_events < 3:
        print(
            "WARNING: fewer than 3 events captured on one side (< 9 images) -- "
            "try increasing --max-frames"
        )


if __name__ == "__main__":
    main()
