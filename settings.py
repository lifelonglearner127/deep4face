# Maximum number of faces detected simultaneously
MAX_FACE_NUMBER = 30

# Max frame rate
MAX_FRAME_RATE = 25

# MP Queue size
QUEUE_BUFFER_SIZE = 12

# Code of usb camera. (You can use media file path to test with videos.)
USB_CAMERA_CODE = [0]

# IP address of web camera
ADDRESS_LIST = ['10.41.0.198', '10.41.0.199']

# Image dimension of the feature extraction network
IMAGE_SIZE = [112, 112]

# Feature extraction Pre-trained model path
ARCFACE_MODEL_PREFIX = "/home/dev/Projects/deep4face/models/model"

ARCFACE_EPOCH = 0

# Retinaface pre-trained model path
RETINAFACE_MODEL = "/home/dev/Projects/deep4face/models/R50"

# face recognition classifier model path
RECOGNIZER = "/home/dev/Projects/deep4face/models/output.pkl"

# GPU device ID, -1 means use CPU
GPU = 0

# Specify whether to add horizontal flip during training
FLIP = 1

# RetinaNet face detection threshold
THRESHOLD = 0.6

# Face reliability threshold for feature extraction
EMBEDDING_THRESHOLD = 0.85

# RetinaNet image scaling factor
SCALES = [1.0]

# Training data path
TRAIN_DATA = "/home/dev/Projects/deep4face/data/train"
TRAIN_CROPPED_DATA = "/home/dev/Projects/deep4face/data/cropped"
