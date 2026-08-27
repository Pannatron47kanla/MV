# Count M&Ms in MM.mp4 and save 6 evenly-spaced result captures.
# Pipeline: Hough Circle Transform (the candies are uniform glossy circles on
#           a plain background, so this is more robust here than color/contour
#           masking, which mis-picks up soft shadow gradients as extra blobs)
#           -> per-circle color classification by HSV hue/sat/val
#           -> nearest-centroid tracking across frames so each physical candy
#              is only counted once as the camera scrolls past it
#           -> annotate + save one result image per capture point, printing
#              "Result-Picture Capture i/6" as each one is produced.
import math
import os
import cv2
import numpy as np

VIDEO_PATH = "MM.mp4"
OUTPUT_DIR = "mm_count_output"
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "MM_result.mp4")
NUM_CAPTURES = 6

MIN_RADIUS, MAX_RADIUS = 25, 75  # px, candy radius range measured on this video
CANDY_MIN_DIST = 60  # px between candy centers, avoids double-hits on one candy

# Tracking: candies sit ~210-250px apart and shift well under 20px per frame
# (measured on this video), so a generous-but-smaller-than-spacing match
# radius reliably follows one candy across frames without jumping to its
# neighbor.
TRACK_MAX_DISTANCE = 100  # px, farthest a match can be from a track's last position
TRACK_MAX_MISSED = 10  # frames a track can go undetected before it's dropped


class CandyTracker:
    """Nearest-centroid tracker: assigns a stable id to each candy so it is
    only counted once no matter how many frames it's visible across."""

    def __init__(self, max_distance=TRACK_MAX_DISTANCE, max_missed=TRACK_MAX_MISSED):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.tracks = {}  # id -> {"x", "y", "missed", "color"}
        self.next_id = 0
        self.total = 0
        self.color_totals = {}

    def update(self, detections):
        # detections: list of (x, y, color)
        matched_ids = set()
        for x, y, color in detections:
            best_id, best_dist = None, self.max_distance
            for tid, tr in self.tracks.items():
                if tid in matched_ids:
                    continue
                dist = math.hypot(tr["x"] - x, tr["y"] - y)
                if dist < best_dist:
                    best_dist, best_id = dist, tid

            if best_id is not None:
                self.tracks[best_id].update(x=x, y=y, missed=0)
                matched_ids.add(best_id)
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"x": x, "y": y, "missed": 0, "color": color}
                matched_ids.add(tid)
                self.total += 1
                self.color_totals[color] = self.color_totals.get(color, 0) + 1

        for tid in list(self.tracks.keys()):
            if tid not in matched_ids:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]


def classify_color(hsv_pixel):
    # Hue ranges tuned by sampling real candies from this video (OpenCV hue is 0-179).
    # Orange/Red/Brown all share the same warm hue band, so they're split by how
    # bright (V) and how saturated (S) the candy is instead of hue alone.
    h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])
    if h < 15 or h > 170:
        if v > 150:
            return "Orange"
        if s > 140:
            return "Red"
        return "Brown"
    if 18 <= h <= 34:
        return "Yellow"
    if 35 <= h <= 85:
        return "Green"
    if 90 <= h <= 140:
        return "Blue"
    return "Unknown"


def find_content_bounds(frame):
    # MM.mp4 pads a portrait clip with black side bars; crop them out so the
    # flat black margin can't distort the Hough circle search.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cols = np.where(gray.max(axis=0) > 20)[0]
    return cols.min(), cols.max()


def detect_candies(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=CANDY_MIN_DIST,
        param1=80,
        param2=35,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS,
    )
    if circles is None:
        return []
    return np.round(circles[0]).astype(int)


def annotate_and_count(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    counts = {}
    detections = []

    for x, y, r in detect_candies(frame):
        # Median over a small patch at the candy center, not a single pixel,
        # so a stray glare highlight doesn't skew the color reading.
        patch = hsv[max(y - 5, 0):y + 5, max(x - 5, 0):x + 5].reshape(-1, 3)
        color = classify_color(np.median(patch, axis=0))
        counts[color] = counts.get(color, 0) + 1
        detections.append((x, y, color))

        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
        cv2.putText(frame, color, (x - r, y - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    return frame, counts, detections


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    ret, first = cap.read()
    if not ret:
        raise SystemExit(f"Could not read any frames from {VIDEO_PATH}")
    x0, x1 = find_content_bounds(first)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind after probing the crop bounds

    # One capture frame index per 1/6th of the video's duration.
    capture_indices = {
        min(int(total_frames * i / NUM_CAPTURES), total_frames - 1): i
        for i in range(1, NUM_CAPTURES + 1)
    }

    out_w, out_h = (x1 - x0 + 1), first.shape[0]
    writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
    )

    tracker = CandyTracker()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame[:, x0:x1 + 1]
        frame, counts, detections = annotate_and_count(frame)
        n = sum(counts.values())
        tracker.update(detections)

        cv2.putText(frame, f"Visible now: {n}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Total M&Ms: {tracker.total}", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

        if frame_idx in capture_indices:
            i = capture_indices[frame_idx]
            label = f"Result-Picture Capture {i}/{NUM_CAPTURES}"
            cv2.putText(frame, label, (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

            out_path = os.path.join(OUTPUT_DIR, f"capture_{i}_of_{NUM_CAPTURES}.jpg")
            cv2.imwrite(out_path, frame)
            print(f"{label}  (visible now: {n}, running total: {tracker.total})  -> saved {out_path}")

        writer.write(frame)
        cv2.imshow("Result", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("Total M&Ms counted in video:", tracker.total)
    print("By color:", tracker.color_totals)
    print("Annotated video saved to:", OUTPUT_VIDEO_PATH)


if __name__ == "__main__":
    main()
