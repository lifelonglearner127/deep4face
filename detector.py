import os
import datetime
import random
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
        self.image_types = (
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".jfif"
        )
        self.video_types = (
            ".avi", ".mp4"
        )
        self.file_types = self.image_types + self.video_types

    def list_images(self, base_path):
        for (dir_name, sub_dirlist, file_list) in os.walk(base_path):
            for filename in file_list:
                ext = filename[filename.rfind("."):].lower()

                if ext.endswith(self.image_types):
                    image_path = os.path.join(dir_name, filename)
                    yield image_path

    def detect(self, frame):
        boxes, landmarks = self.detector.detect(frame,
                                                self.threshold,
                                                scales=self.scales)

        sorted_index = boxes[:, 0].argsort()
        boxes = boxes[sorted_index]
        landmarks = landmarks[sorted_index]
        aligned = self.preprocess(frame, boxes, landmarks)

        return zip(aligned, boxes)

    def crop_faces(self, image_paths, output_folder, save_original=True):
        """Crop faces from the raw training images and save the images.
        """
        if not isinstance(image_paths, list):
            image_paths = list(image_paths)

        total_images = len(image_paths)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        for i, image_path in enumerate(image_paths):
            person = image_path.split(os.path.sep)[-2]
            person_folder = os.path.join(output_folder, person)
            if not os.path.exists(person_folder):
                os.mkdir(person_folder)

            image = cv2.imread(image_path)

            j = 0
            for face, _ in self.detect(image):
                output_filename = os.path.join(person_folder, f"{i}_{j}.jpg")
                cv2.imwrite(output_filename, face)
                j += 1

            print(f'Cropping {person} faces from raw images'
                  f'. Written into {person_folder} Done:'
                  f'{(i + 1) * 100 / total_images:.2f}%')

    def crop_faces_from_dataset(self, input_folder, output_folder,
                                img_nums=20, debug=True):
        """Crop faces from raw training dataset

        Traing dataset can be video or images that contain human faces.
        The directory hierachy should be like as followings.
            - <base_path>/<person_name>/<...>/<files>

        Key Arguments:
        input_folder: The path to the training data folder
        output_folder: The path to the output data folder
        """

        # loop over the training data directory structure
        for entry in os.scandir(input_folder):
            if entry.is_file():
                continue

            person_name = entry.path.split(os.path.sep)[-1]
            person_folder = os.path.join(output_folder, person_name)

            if debug:
                print(f"[INFO] Processing {person_name}")
            try:
                os.makedirs(person_folder)
            except FileExistsError:
                pass

            image_paths = list(self.list_images(entry))
            if len(image_paths) > img_nums:
                image_paths = random.sample(image_paths, img_nums)

            for i, image_path in enumerate(image_paths):
                if debug:
                    print(f"[INFO] Processing {image_path}")

                image = cv2.imread(image_path)
                face_id = 0
                for face, _ in self.detect(image):
                    now_time = datetime.datetime.now()
                    cv2.imwrite(
                        f'{person_folder}/'
                        f'{now_time.strftime("%Y_%m_%d_%H_%M_%S")}_'
                        f'{i}_{face_id}.jpg',
                        face
                    )
                    face_id += 1

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
