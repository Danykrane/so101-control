"""Камеры: список устройств macOS и захват OpenCV/AVFoundation в своем потоке на камеру.

Без Qt: используют и окно, и robotd. Логика захвата взята из so101_camera_monitor и упрощена.
Доступ к камере macOS выдает приложению, из которого запущен процесс (Terminal), а не Python.
"""

from __future__ import annotations

import json
import platform
import subprocess
import threading
import time
from collections import deque

import cv2
import numpy as np


def list_cameras() -> list[dict]:
    """Камеры в порядке индексов OpenCV: AVFoundation-бэкенд OpenCV сортирует устройства по
    uniqueID, system_profiler дает имя и uniqueID. Пустой список, если узнать не удалось."""
    if platform.system() != "Darwin":
        return []
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"], capture_output=True, text=True, timeout=10
        ).stdout
        items = json.loads(out).get("SPCameraDataType", [])
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
    cams = sorted(
        ({"name": c.get("_name", "?"), "uid": c.get("spcamera_unique-id", "")} for c in items),
        key=lambda c: c["uid"],
    )
    for i, c in enumerate(cams):
        c["index"] = i
        c["builtin"] = "facetime" in c["name"].lower()
    return cams


class Grabber(threading.Thread):
    """Захват одной камеры. Последний кадр (BGR) и FPS читаются из любого потока."""

    def __init__(self, index: int):
        super().__init__(name=f"camera-{index}", daemon=True)
        self.index = index
        self.fps = 0.0
        self.error = ""
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._halt = threading.Event()

    def stop(self, timeout: float = 2.0) -> None:
        self._halt.set()
        if self.is_alive():
            self.join(timeout)

    def latest(self) -> tuple[int, np.ndarray | None]:
        """(номер кадра, кадр BGR); номер растет с каждым новым кадром."""
        with self._lock:
            return self._seq, self._frame

    def jpeg(self, width: int | None = None, quality: int = 85) -> bytes | None:
        _, frame = self.latest()
        if frame is None:
            return None
        if width and frame.shape[1] > width:
            frame = cv2.resize(frame, (width, round(frame.shape[0] * width / frame.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def run(self) -> None:
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        try:
            if not cap.isOpened():
                self.error = f"камера #{self.index} не открылась (индекс, доступ к камере у Terminal, занята?)"
                return
            times: deque[float] = deque(maxlen=30)
            failures = 0
            while not self._halt.is_set():
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    failures += 1
                    if failures > 40:
                        self.error = f"камера #{self.index}: нет кадров"
                        return
                    time.sleep(0.05)
                    continue
                failures = 0
                times.append(time.monotonic())
                if len(times) > 1:
                    self.fps = (len(times) - 1) / (times[-1] - times[0])
                with self._lock:
                    self._frame, self._seq = bgr, self._seq + 1
        finally:
            cap.release()
