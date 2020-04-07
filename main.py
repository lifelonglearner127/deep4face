import asyncio
import os
import cv2
import imutils
import socketio
import settings

from multiprocessing import Process, Queue

async def upload_stream(url="http://127.0.0.1:6789"):

    sio = socketio.AsyncClient()

    @sio.on("response", namespace="/video")
    async def on_response(data):
        image_string = 0
        for pair in frame_queue.get():
            if pair[0] == data:
                image_string = cv2.imencode('.jpg', pair[-1])[1].tostring()

        frame_data = {
            "frame": image_string
        }
        await sio.emit("frame_data", frame_data, namespace="/video")

    @sio.on("connect", namespace="/video")
    async def on_connect():
        frame_data = {
            "frame": 0
        }
        await sio.emit("frame_data", frame_data, namespace="/video")

    await sio.connect(url)
    await sio.wait()


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
    camera_dict = {}

    for code in settings.USB_CAMERA_CODE:
        # camera = cv2.VideoCapture(code)
        camera = cv2.VideoCapture("/home/dev/Downloads/future.mp4")
        camera_dict[code] = camera

    Process(target=lambda: asyncio.run(upload_stream())).start()
    asyncio.run(camera_loop())
