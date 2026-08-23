import cv2 as cv
import numpy as np
import os

WINDOW_NAME = "B6610364-นายปัณณธร-ขันละ"
root = os.getcwd()
imgPath = os.path.join(root, "IMG_1218.jpg")
img = cv.imread(imgPath)
if img is None:
    raise FileNotFoundError(f"Could not read image at {imgPath}")

TARGET_WIDTH = 800
h0, w0 = img.shape[:2]
scale = TARGET_WIDTH / w0
img = cv.resize(img, (TARGET_WIDTH, int(h0 * scale)))
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

HOUGH_KWARGS = dict(dp=1.2, minDist=20, param1=50, param2=22, minRadius=9, maxRadius=28)


def detect_and_draw(blurred, label):
    circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, **HOUGH_KWARGS)
    vis = img.copy()
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        count = len(circles[0])
        for x, y, r in circles[0]:
            cv.circle(vis, (x, y), r, (0, 255, 0), 2)
    cv.putText(vis, f"{label} count:{count}", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis, count


cv.namedWindow(WINDOW_NAME)

for ksize in (9, 5):
    gaussian = cv.GaussianBlur(imgGray, (ksize, ksize), 2)
    median = cv.medianBlur(imgGray, ksize)

    visGaussian, countGaussian = detect_and_draw(gaussian, f"Gaussian k={ksize}")
    visMedian, countMedian = detect_and_draw(median, f"Median k={ksize}")

    print(f"kernel={ksize}  Gaussian count={countGaussian}  Median count={countMedian}")

    cv.imshow(WINDOW_NAME, visGaussian)
    cv.waitKey(0)
    cv.imshow(WINDOW_NAME, visMedian)
    cv.waitKey(0)

cv.destroyAllWindows()
