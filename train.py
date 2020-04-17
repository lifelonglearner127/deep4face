import argparse
import os
import pickle
import settings
from imutils.paths import list_images
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

from recognition import FaceEmbedding


os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",
                default=settings.TRAIN_CROPPED_DATA,
                help="path to the training data folder")
args = vars(ap.parse_args())

embedding = FaceEmbedding()

train_X = embedding.extract_face_embeddings(list_images(args["dataset"]))

le = LabelEncoder()
labels = [image_path.split(os.path.sep)[-2]
          for image_path in list_images(args["dataset"])]
train_y = le.fit_transform(labels)

mlp = MLPClassifier(hidden_layer_sizes=(550, ), verbose=True,
                    activation='relu', solver='adam', tol=10e-7,
                    n_iter_no_change=100, learning_rate_init=1e-3,
                    max_iter=50000)

mlp.fit(train_X, train_y)

with open(settings.RECOGNIZER, "wb") as f:
    pickle.dump((mlp, le), f)
