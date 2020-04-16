import os
import datetime
import cv2
import numpy as np
import progressbar

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

    def __save_face(self, base_path, image):
        i = 0
        for face, _ in self.detect(image):
            now_time = datetime.datetime.now()
            cv2.imwrite(
                f'{base_path}/{now_time.strftime("%Y_%m_%d_%H_%M_%S")}_{i}.jpg',
                face
            )
            i += 1

    def crop_faces_from_dataset(self, input_folder, output_folder, debug=True):
        """Crop faces from raw training dataset

        Traing dataset can be video or images that contain human faces.
        The directory hierachy should be like as followings.
            - <base_path>/<person_name>/<...>/<files>

        Key Arguments:
        input_folder: The path to the training data folder
        output_folder: The path to the output data folder
        """

        image_types = (
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".jfif"
        )
        video_types = (
            ".avi", ".mp4"
        )
        file_types = image_types + video_types

        # loop over the training data directory structure
        for (dir_name, sub_dirlist, file_list) in os.walk(input_folder):
            # calculate the current directory level
            relative_path = dir_name.replace(input_folder, '')
            level = relative_path.count(os.sep)

            if level == 0:
                continue

            person_name = relative_path.split(os.path.sep)[1]
            person_folder = os.path.join(output_folder, person_name)

            # make the output folder
            if level == 1:
                if debug:
                    print(f"[INFO] Processing {person_name}")
                try:
                    os.makedirs(person_folder)
                except FileExistsError:
                    pass

            # process the files
            for filename in file_list:
                # determine the file extension of the current file
                ext = filename[filename.rfind("."):].lower()

                # check to see if the file is
                # an image or video that should be processed
                if not ext.endswith(file_types):
                    continue

                if debug:
                    print(f"[INFO] Processing {filename}")

                # process the image type file
                if ext in image_types:
                    image = cv2.imread(os.path.join(dir_name, filename))
                    self.__save_face(person_folder, image)

                # process video type file
                else:
                    vs = cv2.VideoCapture(os.path.join(dir_name, filename))
                    while True:
                        grabbed, frame = vs.read()
                        if not grabbed:
                            break

                        self.__save_face(person_folder, image)

                    vs.release()

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
