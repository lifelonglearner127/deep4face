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
fps = FPS()
fps.start()

while True:
    grabbed, frame = vs.read()
    if not grabbed:
        break

    original = frame.copy()
    original_width = original.shape[1]
    frame = imutils.resize(frame, width=400)

    face_candidates = detector.detect(frame)

    for face, box in face_candidates:
        if box[4] < settings.EMBEDDING_THRESHOLD:
            continue

        predict = mlp.predict_proba([embedding.extract_face_embedding(face)])
        prob = predict.max(1)[0]
        if prob > 0.85:
            name = le.classes_[predict.argmax(1)[0]]
            label = f'{name}: {prob * 100:.2f}%'
            cv2.rectangle(
                frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 255, 0), 1)
            cv2.putText(frame, label, (int(box[0]), int(box[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        else:
            cv2.rectangle(
                frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 0, 255), 1)

    fps.update()
    cv2.putText(frame, f'FPS: {fps.fps():.2f}', (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
    cv2.imshow("Result", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()
