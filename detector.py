import os
import shutil
import time
import cv2
import numpy as np

import settings
from retinaface.retinaface import RetinaFace


class FaceDetector:
    def __init__(self):
        self.detector = RetinaFace(
            settings.RETINAFACE_MODEL, 0, settings.GPU, 'net3'
        )
        self.threshold = settings.THRESHOLD
        self.scales = settings.SCALES
        self.max_face_number = settings.MAX_FACE_NUMBER
        self.counter = 0
        self.image_size = settings.IMAGE_SIZE

    def detect(self, frame):
        boxes, landmarks = self.detector.detect(frame,
                                                self.threshold,
                                                scales=self.scales)

        sorted_index = boxes[:, 0].argsort()
        boxes = boxes[sorted_index]
        landmarks = landmarks[sorted_index]
        aligned = self.preprocess(frame, boxes, landmarks)

        return zip(aligned, boxes)

    def crop_faces(self, image_paths, save_original=True):
        """Crop faces from the raw training images and save the images.
        """
        if not isinstance(image_paths, list):
            image_paths = list(image_paths)

        total_images = len(image_paths)
        if not os.path.exists(settings.TRAIN_RAW_DATA):
            os.mkdir(settings.TRAIN_RAW_DATA)

        for i, image_path in enumerate(image_paths):
            base_path, file_name = os.path.split(image_path)
            if file_name.startswith('cropped'):
                continue

            image = cv2.imread(image_path)
            for face, _ in self.detect(image):
                cv2.imwrite(f'{base_path}/cropped-{time.time()}.jpg', face)

            shutil.move(image_path, os.path.join(settings.TRAIN_RAW_DATA,
                                                 file_name))

            print('Processing crop faces from raw images. Done:'
                  f'{(i + 1) * 100 / total_images:.2f}%')

    def preprocess(self, img, boxes, landmarks, **kwargs):
        aligned = []
        if len(boxes) == len(landmarks):

            for bbox, landmark in zip(boxes, landmarks):
                margin = kwargs.get('margin', 0)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(bbox[0] - margin / 2, 0)
                bb[1] = np.maximum(bbox[1] - margin / 2, 0)
                bb[2] = np.minimum(bbox[2] + margin / 2, img.shape[1])
                bb[3] = np.minimum(bbox[3] + margin / 2, img.shape[0])
                ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
                warped = cv2.resize(ret,
                                    (self.image_size[1], self.image_size[0]))
                aligned.append(warped)

        return aligned
