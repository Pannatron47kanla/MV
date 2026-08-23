import cv2 as cv
import numpy as np
import os

WINDOW_NAME = "B6610364-นายปัณณธร-ขันละ"
root = os.getcwd()
imgPath = os.path.join(root, "IMG_1218.jpg")
img = cv.imread(imgPath)

TARGET_WIDTH = 800
h0, w0 = img.shape[:2]
scale = TARGET_WIDTH / w0
img = cv.resize(img, (TARGET_WIDTH, int(h0 * scale)))
imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray, (9, 9), 2)

circles = cv.HoughCircles(
    imgBlur,
    cv.HOUGH_GRADIENT,
    dp=1.2,
    minDist=20,
    param1=50,
    param2=22,
    minRadius=9,
    maxRadius=28,
)

if circles is None:
    raise RuntimeError("No circles detected - try adjusting HoughCircles params")

circles = np.uint16(np.around(circles))

for circle in circles[0, :]:
    cv.circle(imgRGB, (circle[0], circle[1]), circle[2], (0, 165, 255), 3)

cv.putText(
    imgRGB,
    f"Coins: {len(circles[0])}",
    (10, 30),
    cv.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 165, 255),
    3,
)
cv.namedWindow(WINDOW_NAME)

cv.imshow(WINDOW_NAME, imgGray)
cv.waitKey(0)


cv.imshow(WINDOW_NAME, imgBlur)
cv.waitKey(0)

cv.imshow(WINDOW_NAME, imgRGB)
cv.waitKey(0)

# plt.figure()
# plt.imshow(imgRGB)
# plt.show()
# cv.waitKey(0)

cv.destroyAllWindows()
