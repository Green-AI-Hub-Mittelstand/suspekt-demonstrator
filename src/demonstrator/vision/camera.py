"""Camera abstractions for OAK and USB sources."""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import depthai as dai
import numpy as np


class FrameGrabber:
    """Base class for camera sources with background capture threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._is_running = False

    def start(self) -> None:
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._is_running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)

    def _capture_loop(self) -> None:
        raise NotImplementedError("Subclasses must implement _capture_loop")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def _set_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._latest_frame = frame

    def clear_latest_frame(self) -> None:
        with self._lock:
            self._latest_frame = None


class OAK1MaxCamera(FrameGrabber):
    """Frame grabber for an OAK-1 Max camera (raw BGR preview frames)."""

    def __init__(
        self,
        device_id: Optional[str],
        width: int,
        height: int,
        fps: int,
        use_macro_focus: bool = False,
        manual_focus: Optional[int] = None,
        anti_banding: str = "auto",
        manual_exposure_us: Optional[int] = None,
        manual_iso: Optional[int] = None,
        luma_denoise: Optional[int] = None,
        chroma_denoise: Optional[int] = None,
        sharpness: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._device_id = device_id
        self._frame_width = width
        self._frame_height = height
        self._fps = fps
        self._use_macro_focus = use_macro_focus
        self._manual_focus = manual_focus
        self._anti_banding = anti_banding
        self._manual_exposure_us = manual_exposure_us
        self._manual_iso = manual_iso
        self._luma_denoise = luma_denoise
        self._chroma_denoise = chroma_denoise
        self._sharpness = sharpness
        self._pipeline: Optional[dai.Pipeline] = None
        self._device: Optional[dai.Device] = None
        self._out_queue: Optional[dai.DataOutputQueue] = None
        self._initialize_depthai()

    def _initialize_depthai(self) -> None:
        pipeline = dai.Pipeline()
        color_cam = pipeline.createColorCamera()
        color_cam.setPreviewSize(self._frame_width, self._frame_height)
        color_cam.setInterleaved(False)
        color_cam.setFps(self._fps)

        if self._use_macro_focus:
            color_cam.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.MACRO)
        if self._manual_focus is not None:
            color_cam.initialControl.setManualFocus(int(self._manual_focus))
        anti_banding_key = str(self._anti_banding).strip().lower()
        anti_banding_map = {
            "off": dai.CameraControl.AntiBandingMode.OFF,
            "50hz": dai.CameraControl.AntiBandingMode.MAINS_50_HZ,
            "60hz": dai.CameraControl.AntiBandingMode.MAINS_60_HZ,
            "auto": dai.CameraControl.AntiBandingMode.AUTO,
        }
        color_cam.initialControl.setAntiBandingMode(
            anti_banding_map.get(anti_banding_key, dai.CameraControl.AntiBandingMode.AUTO)
        )
        if self._manual_exposure_us is not None and self._manual_iso is not None:
            color_cam.initialControl.setManualExposure(
                int(self._manual_exposure_us),
                int(self._manual_iso),
            )
        else:
            color_cam.initialControl.setAutoExposureEnable()
        if self._luma_denoise is not None:
            color_cam.initialControl.setLumaDenoise(int(self._luma_denoise))
        if self._chroma_denoise is not None:
            color_cam.initialControl.setChromaDenoise(int(self._chroma_denoise))
        if self._sharpness is not None:
            color_cam.initialControl.setSharpness(int(self._sharpness))

        xlink_out = pipeline.createXLinkOut()
        xlink_out.setStreamName("color_stream")
        color_cam.preview.link(xlink_out.input)

        if self._device_id:
            all_devices = dai.Device.getAllAvailableDevices()
            matching = [d for d in all_devices if d.getMxId() == self._device_id]
            if not matching:
                raise RuntimeError(f"No OAK device found with serial: {self._device_id!r}")
            self._device = dai.Device(pipeline, matching[0])
        else:
            self._device = dai.Device(pipeline)

        self._out_queue = self._device.getOutputQueue(
            name="color_stream",
            maxSize=4,
            blocking=False,
        )
        self._pipeline = pipeline

    def _capture_loop(self) -> None:
        if self._out_queue is None:
            return
        while self._is_running:
            packet = self._out_queue.tryGet()
            if packet is not None:
                frame_bgr = packet.getCvFrame()
                if frame_bgr is not None:
                    self._set_frame(frame_bgr)
            time.sleep(0.001)


class USBWebcamCamera(FrameGrabber):
    """Frame grabber for a USB camera accessible via OpenCV."""

    def __init__(self, device_index: int, width: int, height: int, fps: int) -> None:
        super().__init__()
        self._device_index = device_index
        self._capture = cv2.VideoCapture(self._device_index, cv2.CAP_ANY)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, fps)
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open USB webcam at index {self._device_index}")

    def _capture_loop(self) -> None:
        while self._is_running:
            ret, frame = self._capture.read()
            if not ret:
                time.sleep(0.01)
                continue
            self._set_frame(frame)
            time.sleep(0.001)

    def stop(self) -> None:
        super().stop()
        if self._capture is not None:
            self._capture.release()


def list_oak_devices() -> list[str]:
    """Return available OAK device serial numbers."""
    return [device_info.getMxId() for device_info in dai.Device.getAllAvailableDevices()]
