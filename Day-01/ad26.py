import cv2 as cv
import numpy as np
import math
import os

WINDOW_NAME = "B6610364-นายปัณณธร-ขันละ"
root = os.getcwd()
imageFiles = ["box1.jpeg", "box2.jpeg", "box3.jpeg"]

TARGET_WIDTH = 700
ANGLE_TOL = 10  # degrees of tolerance to call a line "horizontal" or "vertical"

cv.namedWindow(WINDOW_NAME)

for fname in imageFiles:
    imgPath = os.path.join(root, fname)
    img = cv.imread(imgPath)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {imgPath}")

    h0, w0 = img.shape[:2]
    scale = TARGET_WIDTH / w0
    img = cv.resize(img, (TARGET_WIDTH, int(h0 * scale)))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 50, 150)

    lines = cv.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=int(TARGET_WIDTH * 0.25),
        maxLineGap=15,
    )

    horiz, vert = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            length = math.hypot(x2 - x1, y2 - y1)
            if abs(angle) < ANGLE_TOL or abs(abs(angle) - 180) < ANGLE_TOL:
                horiz.append((x1, y1, x2, y2, length, (y1 + y2) / 2))
            elif abs(abs(angle) - 90) < ANGLE_TOL:
                vert.append((x1, y1, x2, y2, length, (x1 + x2) / 2))

    # The workpiece's true border is the OUTERMOST horizontal/vertical line,
    # not necessarily the longest one - a grid-like surface (e.g. a
    # checkerboard) produces internal lines just as long as the border.
    sides = {}
    if horiz:
        sides["Top"] = min(horiz, key=lambda l: l[5])
        sides["Bottom"] = max(horiz, key=lambda l: l[5])
    if vert:
        sides["Left"] = min(vert, key=lambda l: l[5])
        sides["Right"] = max(vert, key=lambda l: l[5])

    vis = img.copy()
    print(f"\n{fname}:")
    colors = {"Top": (0, 255, 0), "Bottom": (0, 255, 255), "Left": (255, 0, 0), "Right": (0, 0, 255)}
    for name, (x1, y1, x2, y2, length, _) in sides.items():
        cv.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), colors[name], 3)
        midX, midY = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv.putText(vis, f"{name}:{length:.0f}px", (midX, midY), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 2)
        print(f"  {name}: {length:.1f} px")

    for missing in {"Top", "Bottom", "Left", "Right"} - sides.keys():
        print(f"  {missing}: not detected")

    cv.putText(vis, fname, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.imshow(WINDOW_NAME, vis)
    cv.waitKey(0)

cv.destroyAllWindows()
