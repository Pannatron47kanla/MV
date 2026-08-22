import cv2
import os

WINDOW_NAME = "B6610364-นายปัณณธร-ขันละ"
root = os.getcwd()
imgPath = os.path.join(root, "IMG_1220.jpg")
img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(
        f"Could not read image at {imgPath} (check path/format support)"
    )

h, w = img.shape[:2]
chanels = img.shape[2] if len(img.shape) == 3 else 1

text = f"h:{h} w:{w} datatype: {img.dtype} chanels: {chanels} : {img.shape}"
cv2.putText(
    img, text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 2, cv2.LINE_AA
)
resized = cv2.resize(img, (640, 480))

cv2.namedWindow(WINDOW_NAME)
cv2.imshow(WINDOW_NAME, resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
