import cv2 as cv
import numpy as np
import os

WINDOW_NAME = "B6610364-นายปัณณธร-ขันละ"
root = os.getcwd()
imgPath = os.path.join(root, "IMG_1221.JPG")
img = cv.imread(imgPath)
if img is None:
    raise FileNotFoundError(f"Could not read image at {imgPath}")

TARGET_WIDTH = 500
h0, w0 = img.shape[:2]
scale = TARGET_WIDTH / w0
img = cv.resize(img, (TARGET_WIDTH, int(h0 * scale)))
imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray, (9, 9), 2)

circles = cv.HoughCircles(
    imgBlur,
    cv.HOUGH_GRADIENT,
    dp=1.2,
    minDist=22,
    param1=100,
    param2=22,
    minRadius=12,
    maxRadius=32,
)

if circles is None:
    raise RuntimeError("No circles detected - try adjusting HoughCircles params")

circles = np.uint16(np.around(circles))[0]

# 1-baht coins are silver (low/mid saturation) and fall in a narrow radius
# band, distinct from the copper coins (high saturation, small radius) and
# the gold-bimetal 10-baht coins (high saturation, large radius). Ranges
# below were measured directly from this photo's coins.
MIN_RADIUS, MAX_RADIUS = 21, 23
MIN_SAT, MAX_SAT = 50, 95
MIN_HUE, MAX_HUE = 15, 28

baht1_count = 0
for x, y, r in circles:
    mask = np.zeros(imgHSV.shape[:2], np.uint8)
    cv.circle(mask, (int(x), int(y)), max(int(r * 0.6), 3), 255, -1)
    hue, sat, val, _ = cv.mean(imgHSV, mask=mask)

    is_baht1 = (
        MIN_RADIUS <= r <= MAX_RADIUS
        and MIN_SAT <= sat <= MAX_SAT
        and MIN_HUE <= hue <= MAX_HUE
    )

    if is_baht1:
        baht1_count += 1
        cv.circle(imgRGB, (x, y), r, (0, 255, 0), 3)
    else:
        cv.circle(imgRGB, (x, y), r, (0, 165, 255), 1)

cv.putText(
    imgRGB,
    f"1 Baht coins: {baht1_count}",
    (10, 30),
    cv.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    3,
)

cv.namedWindow(WINDOW_NAME)
cv.imshow(WINDOW_NAME, imgRGB)
cv.waitKey(0)
cv.destroyAllWindows()
