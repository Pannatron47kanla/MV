# Q3 -- Shape Detection: Classify Shape + Measure Size (mm) on images and video
# Pipeline: Grayscale -> Blur -> Otsu Threshold (inverted) -> Morphological
#           Open then Close -> Contours -> Shape Classification
#           (circularity + polygon approx) -> Size Measurement
#           (minEnclosingCircle / minAreaRect) -> Annotate
#
# Source media: every *.jpg image and *.mp4 video under images/. Each item
# shows a single flat-color shape (with a watermark and decorative texture)
# on a plain light background, so a global Otsu threshold segments it
# cleanly -- no adaptive thresholding or belt reference lines needed here.
import glob
import os

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "images")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# ---------------------------------------------------------------------------
# Calibration: how many millimeters one pixel represents. These are stock
# illustrations with no physical reference object in frame, so this is a
# placeholder -- point it at a real known-length/known-pixel ratio if you
# swap in footage of actual parts.
# ---------------------------------------------------------------------------
MM_PER_PIXEL = 0.5

MIN_CONTOUR_AREA = 3000  # px^2, drops watermark specks and stray noise
MAX_CONTOUR_AREA_RATIO = 0.85  # contours covering more of the frame than this
# are threshold artifacts from the near-blank flash/transition frames between
# shapes in the source clips, not real shapes -- discard them.
CIRCULARITY_THRESHOLD = 0.82  # 4*pi*Area/Perimeter^2, 1.0 = perfect circle
SQUARE_ASPECT_TOLERANCE = 0.12  # |long/short - 1| below this => "Square"
PLAYBACK_DELAY_MS = 250  # ms between displayed frames; raise to slow playback down

DISPLAY_AVAILABLE = True  # flips off automatically if there's no GUI backend


def classify_shape(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return "Unknown"

    circularity = 4 * np.pi * area / (perimeter**2)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    vertices = len(approx)

    if circularity >= CIRCULARITY_THRESHOLD:
        return "Circle"
    if vertices == 3:
        return "Triangle"
    if vertices == 4:
        (_, _), (w, h), _ = cv2.minAreaRect(contour)
        if min(w, h) == 0:
            return "Quadrilateral"
        ratio = max(w, h) / min(w, h)
        return "Square" if abs(ratio - 1) <= SQUARE_ASPECT_TOLERANCE else "Rectangle"
    if vertices == 5:
        return "Pentagon"
    if vertices == 6:
        return "Hexagon"
    if vertices == 12:
        return "Cross"
    return "Polygon-{}".format(vertices)


def measure_and_draw(img_out, contour, shape):
    # minAreaRect/minEnclosingCircle (not boundingRect) so size stays correct
    # even when the shape is rotated.
    if shape == "Circle":
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        diameter_mm = 2 * radius * MM_PER_PIXEL
        cv2.circle(img_out, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
        label = "{} d={:.1f}mm".format(shape, diameter_mm)
        origin = (int(cx - radius), max(int(cy - radius) - 8, 15))
    else:
        rect = cv2.minAreaRect(contour)
        (_, _), (w_px, h_px), _ = rect
        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(img_out, [box], 0, (0, 255, 0), 2)
        width_mm = min(w_px, h_px) * MM_PER_PIXEL
        length_mm = max(w_px, h_px) * MM_PER_PIXEL
        label = "{} {:.1f}x{:.1f}mm".format(shape, width_mm, length_mm)
        bx, by, _, _ = cv2.boundingRect(contour)
        origin = (bx, max(by - 8, 15))

    cv2.putText(
        img_out, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA
    )
    return label


def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Inverted Otsu: shapes are colored/dark against a near-white background,
    # so the foreground ends up as the bright class after inversion.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Opening first clears the faint watermark text and hatching specks;
    # closing then fills any small gaps left inside a shape's silhouette.
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned


def find_valid_contours(cleaned):
    frame_area = cleaned.shape[0] * cleaned.shape[1]
    max_area = MAX_CONTOUR_AREA_RATIO * frame_area
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if MIN_CONTOUR_AREA < cv2.contourArea(c) < max_area]


def show(window_name, img):
    global DISPLAY_AVAILABLE
    if not DISPLAY_AVAILABLE:
        return
    try:
        cv2.imshow(window_name, img)
    except cv2.error:
        DISPLAY_AVAILABLE = False


def process_image(path):
    name = os.path.basename(path)
    frame = cv2.imread(path)
    if frame is None:
        print("skip (unreadable):", name)
        return

    cleaned = preprocess(frame)
    contours = find_valid_contours(cleaned)

    vis = frame.copy()
    results = {}
    for cnt in contours:
        shape = classify_shape(cnt)
        measure_and_draw(vis, cnt, shape)
        results[shape] = results.get(shape, 0) + 1

    out_path = os.path.join(OUTPUT_DIR, "annotated_" + name)
    cv2.imwrite(out_path, vis)
    print("{}: {}".format(name, results))

    show("Threshold - " + name, cleaned)
    show("Result - " + name, vis)
    if DISPLAY_AVAILABLE:
        cv2.waitKey(1200)  # brief preview; never blocks the script on a keypress
        cv2.destroyAllWindows()


def process_video(path):
    name = os.path.basename(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("skip (unreadable):", name)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(OUTPUT_DIR, "annotated_" + os.path.splitext(name)[0] + ".mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # These clips flash one shape on screen for a single frame at a time,
    # separated by near-blank frames (not a continuous belt/pan shot), so a
    # piece is counted on the rising edge: the frame where a shape first
    # appears after a gap with none detected.
    shape_present = False
    total_pieces = 0
    results = {}

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        cleaned = preprocess(frame)
        contours = find_valid_contours(cleaned)
        vis = frame

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            shape = classify_shape(cnt)
            measure_and_draw(vis, cnt, shape)

            if not shape_present:
                total_pieces += 1
                results[shape] = results.get(shape, 0) + 1
            shape_present = True
        else:
            shape_present = False

        cv2.putText(
            vis, "n=" + str(total_pieces), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA,
        )

        writer.write(vis)
        show("Video - " + name, vis)
        if DISPLAY_AVAILABLE and cv2.waitKey(PLAYBACK_DELAY_MS) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    if DISPLAY_AVAILABLE:
        cv2.destroyAllWindows()
    print("{}: total={} {}".format(name, total_pieces, results))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for image_path in sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))):
        process_image(image_path)

    for video_path in sorted(glob.glob(os.path.join(IMAGE_DIR, "*.mp4"))):
        process_video(video_path)

    print("Annotated output saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
