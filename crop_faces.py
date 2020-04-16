import argparse
import os
import settings
from detector import FaceDetector


os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",
                default=settings.TRAIN_DATA,
                help="path to the training data folder")
ap.add_argument("-o", "--output",
                default=settings.TRAIN_CROPPED_DATA,
                help="path to the output folder")
args = vars(ap.parse_args())

detector = FaceDetector()

detector.crop_faces_from_dataset(args["dataset"], args["output"])
