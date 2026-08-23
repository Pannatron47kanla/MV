import cv2 as cv
import numpy as np
import os

WINDOW_NAME = "B6610364-นายปัณณธร-ขันละ"
root = os.getcwd()
imageFiles = ["romdom1.jpg", "random2.jpg", "random3.jpg"]

PANEL_WIDTH = 350


def labeled(img, text):
    img = img.copy()
    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.putText(img, text, (8, 22), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


cv.namedWindow(WINDOW_NAME)

for fname in imageFiles:
    imgPath = os.path.join(root, fname)
    img = cv.imread(imgPath)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {imgPath}")

    h0, w0 = img.shape[:2]
    scale = PANEL_WIDTH / w0
    img = cv.resize(img, (PANEL_WIDTH, int(h0 * scale)))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    sobelX = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
    sobelY = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)
    sobel = cv.convertScaleAbs(cv.magnitude(sobelX, sobelY))

    laplacian = cv.convertScaleAbs(cv.Laplacian(blurred, cv.CV_64F, ksize=3))

    canny = cv.Canny(blurred, 50, 150)

    topRow = np.hstack([labeled(img, "Original"), labeled(sobel, "Sobel")])
    bottomRow = np.hstack([labeled(laplacian, "Laplacian"), labeled(canny, "Canny")])
    grid = np.vstack([topRow, bottomRow])

    cv.imshow(WINDOW_NAME, grid)
    cv.waitKey(0)

cv.destroyAllWindows()
