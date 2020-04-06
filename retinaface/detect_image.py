import argparse
import datetime

import cv2
import imutils
import numpy as np
from retinaface import RetinaFace


# parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the image")
args = vars(ap.parse_args())

# initialize face detector
detector = RetinaFace('./models/R50', 0, 0, 'net3')

# detect face in image
image = cv2.imread(args["image"])
faces, landmarks = detector.detect(image)

before = datetime.datetime.now()
faces, _ = detector.detect(image)
after = datetime.datetime.now()
print(f"[INFO]: It took {(after - before).total_seconds()} to detect")

for i in range(faces.shape[0]):
    box = faces[i].astype(np.int)
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
