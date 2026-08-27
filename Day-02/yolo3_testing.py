# Detect people in an image with YOLOv3 (classic Darknet weights) and draw boxes.
# Pipeline: cv2.dnn.readNetFromDarknet -> blobFromImage -> forward pass on the
#           3 YOLO output layers -> keep only "person" detections above
#           CONF_THRESHOLD -> Non-Max Suppression to drop duplicate boxes on
#           the same person -> draw box + confidence label.
#
# NOTE: OpenCV 5.x removed the Darknet (.cfg/.weights) importer, so this
# script needs an older opencv-python. Run it with the project's venv-yolo3
# environment (created via: python3 -m venv venv-yolo3 && venv-yolo3/bin/pip
# install "opencv-python==4.10.0.84" numpy), e.g.:
#   venv-yolo3/bin/python Day-02/yolo3_testing.py [path/to/image.jpg]
import os
import sys

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_DIR = os.path.join(SCRIPT_DIR, "Add24", "data", "yoloV3")
CFG_PATH = os.path.join(YOLO_DIR, "yolov3.cfg")
WEIGHTS_PATH = os.path.join(YOLO_DIR, "yolov3.weights")
NAMES_PATH = os.path.join(YOLO_DIR, "coco.names")

DEFAULT_IMAGE = os.path.join(SCRIPT_DIR, "Add24", "images", "people01.jpg")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "yolo3_output")

INPUT_SIZE = (416, 416)  # network's expected square input
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4


def load_class_names(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_net(cfg_path, weights_path):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def output_layer_names(net):
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


def detect_people(net, out_layers, image, person_id):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, INPUT_SIZE, swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_layers)

    boxes, confidences = [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id != person_id or confidence < CONF_THRESHOLD:
                continue

            # YOLO gives box center + size as fractions of the image; convert
            # to absolute pixel top-left + size for cv2.rectangle/NMSBoxes.
            cx, cy, bw, bh = detection[0:4] * np.array([w, h, w, h])
            x, y = int(cx - bw / 2), int(cy - bh / 2)
            boxes.append([x, y, int(bw), int(bh)])
            confidences.append(float(confidence))

    keep = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    keep = keep.flatten() if len(keep) > 0 else []
    return [(boxes[i], confidences[i]) for i in keep]


def draw_detections(image, detections):
    for (x, y, w, h), confidence in detections:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Person {confidence:.2f}"
        label_y = y - 8 if y - 8 > 10 else y + 20
        cv2.putText(image, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return image


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE

    class_names = load_class_names(NAMES_PATH)
    person_id = class_names.index("person")

    net = load_net(CFG_PATH, WEIGHTS_PATH)
    out_layers = output_layer_names(net)

    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    detections = detect_people(net, out_layers, image, person_id)
    draw_detections(image, detections)

    cv2.putText(image, f"People: {len(detections)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(image_path))[0] + "_result.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, image)

    print(f"People detected: {len(detections)}")
    for (x, y, w, h), confidence in detections:
        print(f"  box=({x},{y},{w},{h})  confidence={confidence:.2f}")
    print("Saved:", out_path)

    cv2.imshow("YOLOv3 - Person Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
