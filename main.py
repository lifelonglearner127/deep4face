import asyncio
import os
import pickle
import time
import cv2
import socketio
import detector as fd
import recognition
import settings
from queue import Full
from termcolor import colored
from multiprocessing import Process, Queue


async def upload_stream(url="http://127.0.0.1:6789"):

    sio = socketio.AsyncClient()

    @sio.on("response", namespace="/video")
    async def on_response(data):
        image_string = 0
        result_string = 0

        for pair in upstream_queue.get():
            if pair[0] == data:
                image_string = cv2.imencode('.jpg', pair[-1])[1].tostring()
                if len(pair) == 3:
                    result_string = pair[1]
                break

        frame_data = {
            "frame": image_string, "result": result_string
        }
        await sio.emit("frame_data", frame_data, namespace="/video")

        try:
            face, dt, prob, name, ip = result_queue.get_nowait()
            jstr = {
                "image": cv2.imencode('.jpg', face)[1].tostring(),
                "time": dt,
                "name": name,
                "prob": prob,
                "ip": ip,
            }
            await sio.emit("result_data", jstr, namespace="/video")
        except Exception:
            pass

    @sio.on("connect", namespace="/video")
    async def on_connect():
        frame_data = {
            "frame": 0,
            "result": 0
        }
        await sio.emit("frame_data", frame_data, namespace="/video")

    await sio.connect(url)
    await sio.wait()


async def detection_loop():
    print("[INFO] loading RetinaFace Pre-trained model")
    detector = fd.FaceDetector()

    rate = settings.MAX_FRAME_RATE
    loop = asyncio.get_running_loop()

    while True:
        start_time = loop.time()
        frame_list = []
        for pair in frame_queue.get():
            address, frame = pair
            face_rects = []
            face_candidates = detector.detect(frame)

            for face, box in face_candidates:
                face_rects.append(box.tolist())
                if box[4] > settings.EMBEDDING_THRESHOLD:
                    try:
                        suspicion_face_queue.put_nowait((address, face))
                    except Full:
                        print(colored(f'Suspicion face queue is full', 'red'),
                              flush=True)

            if len(face_rects):
                res = (address, face_rects, frame)
            else:
                res = (address, frame)

            frame_list.append(res)

        print(colored(f'Detection cost: {loop.time() - start_time}', 'red'),
              flush=True)
        upstream_queue.put(frame_list)

        for _ in range(int((loop.time() - start_time) * rate)):
            upstream_queue.put(frame_queue.get())


async def recognition_loop():
    embedding = recognition.FaceEmbedding()
    with open(settings.RECOGNIZER, 'rb') as f:
        (mlp, le) = pickle.load(f)

    while True:
        ip, face = suspicion_face_queue.get()
        dt = time.strftime("%m-%d %H:%M:%S")
        predict = mlp.predict_proba([embedding.extract_face_embedding(face)])
        prob = predict.max(1)[0]
        name = le.classes_[predict.argmax(1)[0]]
        result_queue.put((face, dt, prob, name, ip))


async def camera_loop():
    rate = 1 / settings.MAX_FRAME_RATE
    loop = asyncio.get_event_loop()

    while True:
        start_time = loop.time()
        pairs = []

        for code, camera in camera_dict.items():
            grabbed, frame = camera.read()
            if not grabbed:
                continue

            pairs.append((str(code), cv2.resize(frame, (672, 672))))

        if len(pairs) > 0:
            frame_queue.put(pairs)

        rest_time = start_time + rate - loop.time()
        if rest_time > 0:
            await asyncio.sleep(rest_time)


if __name__ == "__main__":
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    frame_queue = Queue(settings.QUEUE_BUFFER_SIZE)
    upstream_queue = Queue(settings.QUEUE_BUFFER_SIZE)
    suspicion_face_queue = Queue(settings.MAX_FACE_NUMBER)
    result_queue = Queue(settings.MAX_FACE_NUMBER)
    camera_dict = {}

    for code in settings.USB_CAMERA_CODE:
        # camera = cv2.VideoCapture(code)
        camera = cv2.VideoCapture("/home/dev/Downloads/future.mp4")
        camera_dict[code] = camera

    Process(target=lambda: asyncio.run(recognition_loop())).start()
    Process(target=lambda: asyncio.run(detection_loop())).start()
    Process(target=lambda: asyncio.run(upload_stream())).start()
    asyncio.run(camera_loop())
