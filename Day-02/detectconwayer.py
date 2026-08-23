# Q3 -- Conveyor Belt: Count Pieces + Classify Shape + Measure Size (mm)
# Pipeline: Adaptive Threshold -> Morphological Closing then Opening -> Contours
#           -> Shape Classification (circularity + polygon approx)
#           -> Size Measurement (minEnclosingCircle / minAreaRect) -> Annotate
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Calibration: how many millimeters one pixel represents in the cropped belt
# ROI below. Measure a known real-world length in the video (e.g. belt width,
# or a reference object of known size) and its pixel length, then set:
#   MM_PER_PIXEL = known_length_mm / known_length_px
# All size readouts on screen are only as accurate as this constant.
# ---------------------------------------------------------------------------
MM_PER_PIXEL = 0.5

MIN_CONTOUR_AREA = 3000  # px^2, drops noise specks and edge-of-frame slivers
CIRCULARITY_THRESHOLD = 0.82  # 4*pi*Area/Perimeter^2, 1.0 = perfect circle
SQUARE_ASPECT_TOLERANCE = 0.12  # |long/short - 1| below this => "Square"

PLAYBACK_DELAY_MS = 30  # cv2.waitKey delay per frame; raise this to slow playback

frameCount, delayFrame = 1234, 10
nCounter = 0
stsCount = 0
lastShape = None
results = {}  # shape name -> count of confirmed pieces, e.g. {"Circle": 5}


def classify_shape(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return "Unknown"

    circularity = 4 * np.pi * area / (perimeter**2)
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
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
    return "Polygon"


def measure_and_draw(imgOut, contour, shape):
    # minAreaRect/minEnclosingCircle (not boundingRect) so size stays correct
    # even when the piece is rotated on the belt.
    if shape == "Circle":
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        diameter_mm = 2 * radius * MM_PER_PIXEL
        cv2.circle(imgOut, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
        label = f"{shape} d={diameter_mm:.1f}mm"
        origin = (int(cx - radius), int(cy - radius) - 8)
    else:
        rect = cv2.minAreaRect(contour)
        (_, _), (w_px, h_px), _ = rect
        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(imgOut, [box], 0, (0, 255, 0), 2)
        width_mm = min(w_px, h_px) * MM_PER_PIXEL
        length_mm = max(w_px, h_px) * MM_PER_PIXEL
        label = f"{shape} {width_mm:.1f}x{length_mm:.1f}mm"
        bx, by, _, _ = cv2.boundingRect(contour)
        origin = (bx, by - 8)

    cv2.putText(
        imgOut, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA
    )
    return cv2.boundingRect(contour)


cap = cv2.VideoCapture("q3.avi")
while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        break
    else:
        # Cut the Area, Blur, Change to Gray
        belt = frame[50:475, 80:400]
        belt = cv2.medianBlur(belt, 5)
        gray_belt = cv2.cvtColor(belt, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Frame-Input", belt)

        # Adaptive Thresholding: the belt has an uneven lighting gradient
        # (bright glare in one corner), so a single global cutoff -- fixed
        # or Otsu -- misreads background glare as part of a piece. Adaptive
        # thresholding compares each pixel to its own local neighborhood,
        # so it stays correct across the gradient.
        threshold = cv2.adaptiveThreshold(
            gray_belt,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,
            -5,
        )

        # Morphological Closing (dilate-then-erode: bridges gaps/rings inside
        # a piece into one solid blob) then Opening (erode-then-dilate:
        # clears small noise specks left over)
        kernel = np.ones((7, 7), np.uint8)
        cleaned = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=3)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
        cv2.imshow("Frame-Threshold", cleaned)

        # Draw Ref-line
        H, W, C = belt.shape
        LineInRef = int(H * 0.87)
        LineOutRef = int(H * 0.97)
        cv2.line(belt, (0, LineInRef), (W, LineInRef), (255, 192, 203), 2)
        cv2.line(belt, (0, LineOutRef), (W, LineOutRef), (255, 0, 0), 2)

        # Find each Piece, Classify its Shape, Measure its Size, Draw + Count
        contours, hierarchy = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        frameCount = frameCount + 1
        pieceCountThisFrame = 0
        for contour in contours:
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue

            pieceCountThisFrame += 1
            shape = classify_shape(contour)
            x, y, w, h = measure_and_draw(belt, contour, shape)

            if (y + h) > LineInRef:
                stsLine_In = 1
            else:
                stsLine_In = 0

            if (y + h) > LineOutRef:
                stsLine_Out = 1
            else:
                stsLine_Out = 0

            if stsLine_Out == 0 and stsLine_In == 0:
                stsCount = 1
                lastShape = shape

            if stsLine_Out == 1 and stsLine_In == 1 and frameCount > delayFrame:
                nCounter = nCounter + stsCount
                if stsCount and lastShape is not None:
                    results[lastShape] = results.get(lastShape, 0) + 1
                frameCount = 0
                stsCount = 0

        # Put Text - Totals
        cv2.putText(
            belt,
            "n=" + str(nCounter),
            (5, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            belt,
            "On-belt: " + str(pieceCountThisFrame),
            (5, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Show Result and Wait Exit
        cv2.imshow("Output-Frame", belt)
        if cv2.waitKey(PLAYBACK_DELAY_MS) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print("nCounter (total pieces passed) =", nCounter)
print("สรุปผล:", results)
