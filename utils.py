from datetime import datetime


class FPS:
    def __init__(self):
        self._start = None
        self._num_frames = 0

    def start(self):
        self._start = datetime.now()

    def update(self):
        self._num_frames += 1

    def fps(self):
        elapsed = (datetime.now() - self._start).total_seconds()
        return self._num_frames / elapsed
