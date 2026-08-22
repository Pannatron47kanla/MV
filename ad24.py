import cv2 as cv
import numpy as np
import os

WINDOW_NAME = "B6610364-นายปัณณธร-ขันละ"
root = os.getcwd()
imgPath = os.path.join(root, "IMG_1218.jpg")
img = cv.imread(imgPath)
if img is None:
    raise FileNotFoundError(f"Could not read image at {imgPath}")

TARGET_WIDTH = 900
h0, w0 = img.shape[:2]
scale = TARGET_WIDTH / w0
img = cv.resize(img, (TARGET_WIDTH, int(h0 * scale)))

imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Yellow sits in the middle of the hue range, so unlike red it doesn't
# wrap around 0/180 - one range is enough.
lowerYellow = np.array([20, 100, 60])
upperYellow = np.array([35, 255, 255])
mask = cv.inRange(imgHSV, lowerYellow, upperYellow)

kernel = np.ones((5, 5), np.uint8)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

# Caps touching each other merge into one non-circular blob in the mask.
# Eroding shrinks each blob so touching caps separate into distinct
# contours; the radius is compensated back afterwards for drawing.
ERODE_ITERS = 3
RADIUS_COMPENSATE = ERODE_ITERS * 2  # ~2px shrink per iteration with a 5x5 kernel
eroded = cv.erode(mask, kernel, iterations=ERODE_ITERS)

contours, _ = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

yellowCapCount = 0
for c in contours:
    area = cv.contourArea(c)
    if area < 20:
        continue
    (x, y), r = cv.minEnclosingCircle(c)
    yellowCapCount += 1
    cv.circle(img, (int(x), int(y)), int(r) + RADIUS_COMPENSATE, (0, 255, 0), 3)

cv.putText(
    img,
    f"Yellow caps: {yellowCapCount}",
    (10, 30),
    cv.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    3,
)

cv.namedWindow(WINDOW_NAME)
cv.imshow(WINDOW_NAME, mask)
cv.waitKey(0)

cv.imshow(WINDOW_NAME, img)
cv.waitKey(0)
cv.destroyAllWindows()
