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

# Red wraps around the hue circle (0 and 180), so two ranges are needed.
lowerRed1 = np.array([0, 100, 60])
upperRed1 = np.array([10, 255, 255])
lowerRed2 = np.array([170, 100, 60])
upperRed2 = np.array([180, 255, 255])
mask = cv.inRange(imgHSV, lowerRed1, upperRed1) | cv.inRange(imgHSV, lowerRed2, upperRed2)

kernel = np.ones((5, 5), np.uint8)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

redCapCount = 0
for c in contours:
    area = cv.contourArea(c)
    if area < 100:
        continue
    perimeter = cv.arcLength(c, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if circularity < 0.6:
        continue

    (x, y), r = cv.minEnclosingCircle(c)
    redCapCount += 1
    cv.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 3)

cv.putText(
    img,
    f"Red caps: {redCapCount}",
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
