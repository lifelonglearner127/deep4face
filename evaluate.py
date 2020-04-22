import argparse
import os
import pickle
import cv2
import imutils
import detector as fd
import recognition
import settings
from utils import FPS


os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                required=True,
                help="path to video file")
ap.add_argument("-o", "--output",
                help="path to the output folder")
args = vars(ap.parse_args())

detector = fd.FaceDetector()
embedding = recognition.FaceEmbedding()
with open(settings.RECOGNIZER, 'rb') as f:
    (mlp, le) = pickle.load(f)

vs = cv2.VideoCapture(args["video"])
frame = vs.read()[1]

fps = FPS()
fps.start()

original_width = frame.shape[1]
original_height = frame.shape[0]
resized_width = 400
scale = original_width / resized_width

if args["output"] is not None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(args["output"], fourcc, 24,
                             (original_width, original_height), True)

while True:
    grabbed, frame = vs.read()
    if not grabbed:
        break

    original = frame.copy()
    frame = imutils.resize(frame, width=resized_width)

    face_candidates = detector.detect(frame)

    for face, box in face_candidates:
        if box[4] < settings.EMBEDDING_THRESHOLD:
            continue

        start_x = int(scale * box[0])
        start_y = int(scale * box[1])
        end_x = int(scale * box[2])
        end_y = int(scale * box[3])
        predict = mlp.predict_proba([embedding.extract_face_embedding(face)])
        prob = predict.max(1)[0]
        if prob > 0.7:
            name = le.classes_[predict.argmax(1)[0]]
            label = f'{name}: {prob * 100:.2f}%'

            cv2.rectangle(
                original, (start_x, start_y), (end_x, end_y),
                (0, 255, 0), 1)
            cv2.putText(original, label, (start_x, start_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        else:
            cv2.rectangle(
                original, (start_x, start_y), (end_x, end_y),
                (0, 0, 255), 1)

    fps.update()
    cv2.putText(original, f'FPS: {fps.fps():.2f}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    cv2.imshow("Result", original)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if writer is not None:
        writer.write(original)

if writer is not None:
    writer.release()

vs.release()
cv2.destroyAllWindows()
