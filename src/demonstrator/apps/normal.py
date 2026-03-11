# main.py

import base64
import io
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import qrcode
from ultralytics import YOLO
from fastapi import HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from demonstrator.apps.common import build_app, mjpeg_stream_generator, register_video_route
from demonstrator.vision.aruco import detect_aruco_markers_with_tracking, calculate_dimensions
from demonstrator.vision.camera import FrameGrabber, OAK1MaxCamera, USBWebcamCamera, list_oak_devices
from demonstrator.config.camera_ids import LEFT_CAMERA_SERIAL, RIGHT_CAMERA_SERIAL
from niimprint import PrinterClient, SerialTransport
from demonstrator.config.settings import (
    ARUCO_ENABLED,
    ARUCO_SKIP_FRAMES,
    BATCH_SIZE,
    DETECTION_PERSIST_SECONDS,
    INFERENCE_MAX_FPS,
    DEFAULT_CENTER_CAMERA_ROI_REL,
    OAK_FPS,
    OAK_FRAME_HEIGHT,
    OAK_FRAME_WIDTH,
    OAK_ANTI_BANDING,
    OAK_CHROMA_DENOISE,
    OAK_LUMA_DENOISE,
    OAK_MANUAL_EXPOSURE_US,
    OAK_MANUAL_ISO,
    OAK_SHARPNESS,
    SKIP_FRAMES,
    USB_CAPTURE_FPS,
    USB_CAPTURE_HEIGHT,
    USB_CAPTURE_WIDTH,
    USB_DEVICE_INDEX,
    YOLO_CONF_THRESH,
    YOLO_DEVICE,
    YOLO_ENGINE_CENTER,
    YOLO_ENGINE_LEFT,
    YOLO_ENGINE_RIGHT,
    YOLO_IOU_THRESH,
    YOLO_MAX_DET,
    YOLO_MODEL_INPUT_SIZE,
    YOLO_SOURCE_CENTER,
    YOLO_SOURCE_LEFT,
    YOLO_SOURCE_RIGHT,
    get_persisted_center_roi,
    get_persisted_side_camera_mapping,
    normalise_roi_tuple,
    persist_center_roi,
    persist_side_camera_mapping,
    resolve_runtime_engine_path,
    resolve_side_camera_serials,
)
# ============================================
# FASTAPI-Setup
# ============================================
app, templates, video_dir = build_app(log_prefix="Normal")
register_video_route(app, video_dir, log_prefix="normal")

def _initial_center_roi() -> Optional[Tuple[float, float, float, float]]:
    persisted = get_persisted_center_roi()
    if persisted is not None:
        return persisted
    return normalise_roi_tuple(DEFAULT_CENTER_CAMERA_ROI_REL)


CENTER_CAMERA_ROI_REL = _initial_center_roi()
CURRENT_CENTER_CAMERA_ROI_REL = CENTER_CAMERA_ROI_REL


# ============================================
# DetectCamera mit Ensemble-Prediction
# ============================================
class DetectCamera(FrameGrabber):
    """
    Wrappt einen Roh-FrameGrabber (z.B. USBWebcamCamera) mit einer YOLO-Engine.
    Zusätzlich führt es ArUco-Erkennung auf 320×320 aus (alle ARUCO_SKIP_FRAMES),
    um Pixel→mm-Ratio zu berechnen. Zeigt reale Maße und FPS neben Box-Labels.
    """
    def __init__(
        self,
        base_cam: FrameGrabber,
        engine_path: str,
        model_input_size: int = YOLO_MODEL_INPUT_SIZE,
        conf_thresh: float = YOLO_CONF_THRESH,
        iou_thresh: float = YOLO_IOU_THRESH,
        max_det: int = YOLO_MAX_DET,
        device: str = YOLO_DEVICE,
        skip_frames: int = SKIP_FRAMES,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        super().__init__()
        self.base_cam = base_cam
        self.model_input_size = model_input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det
        self.device = device
        self.skip_frames = skip_frames
        self.batch_size = batch_size

        self.model = YOLO(engine_path, task="detect")
        self.model.overrides["imgsz"] = (model_input_size, model_input_size)
        self.model.overrides["conf"] = conf_thresh
        self.model.overrides["iou"] = iou_thresh
        self.model.overrides["max_det"] = max_det

        self._lock = threading.Lock()
        self._latest_annotated: Optional[np.ndarray] = None
        self._latest_annotated_full: Optional[np.ndarray] = None
        self._is_running = False
        self._frame_counter = 0
        self.last_known_ratio: Optional[float] = None  # Pixel→mm-Ratio
        self._fps_start_time = time.time()            # Startzeit für FPS-Berechnung
        self._fps_frame_count = 0                     # Frame-Zähler für FPS
        self._fps: float = 0.0                        # zuletzt berechnete FPS
        self._persist_seconds = max(0.0, float(DETECTION_PERSIST_SECONDS))
        self._infer_min_interval = (
            0.0 if INFERENCE_MAX_FPS <= 0 else (1.0 / float(INFERENCE_MAX_FPS))
        )
        self._last_infer_time: float = 0.0
        self._cached_boxes: np.ndarray = np.empty((0, 4), dtype=np.int32)
        self._cached_scores: np.ndarray = np.empty((0,), dtype=np.float32)
        self._cached_classes: np.ndarray = np.empty((0,), dtype=np.int32)
        self._roi_rel: Optional[Tuple[float, float, float, float]] = None
        self._roi_pixels: Optional[Tuple[int, int, int, int]] = None
        self._apply_roi_locked(CENTER_CAMERA_ROI_REL)

    def start(self) -> None:
        self.base_cam.start()
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._is_running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)

    def _inference_loop(self) -> None:
        while self._is_running:
            raw_full = self.base_cam.get_latest_frame()  # 640×480 BGR
            if raw_full is None:
                time.sleep(0.001)
                continue

            self._frame_counter += 1
            now = time.time()
            should_infer_frame = (self._frame_counter % max(1, self.skip_frames)) == 0
            should_infer_time = (
                self._infer_min_interval <= 0.0
                or (now - self._last_infer_time) >= self._infer_min_interval
            )
            run_inference = should_infer_frame and should_infer_time
            need_320 = run_inference or (
                ARUCO_ENABLED and (self._frame_counter % ARUCO_SKIP_FRAMES) == 0
            )
            frame_320 = None

            # 1) Downsample nur wenn ArUco oder Inferenz diesen Frame benötigen
            if need_320:
                frame_320 = cv2.resize(
                    raw_full,
                    (self.model_input_size, self.model_input_size),
                    interpolation=cv2.INTER_LINEAR,
                )

            # 2) ArUco nur, wenn enabled UND bestimmte Frame-Nummer:
            if (
                frame_320 is not None
                and ARUCO_ENABLED
                and (self._frame_counter % ARUCO_SKIP_FRAMES) == 0
            ):
                ratio, _ = detect_aruco_markers_with_tracking(frame_320)
                if ratio is not None:
                    self.last_known_ratio = ratio

            if run_inference and frame_320 is not None:
                img_rgb = cv2.cvtColor(frame_320, cv2.COLOR_BGR2RGB)
                results = self.model.predict(
                    source=[img_rgb],
                    imgsz=self.model_input_size,
                    device=self.device,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    max_det=self.max_det,
                    augment=False,
                    verbose=False,
                )
                self._cached_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                self._cached_scores = results[0].boxes.conf.cpu().numpy()
                self._cached_classes = results[0].boxes.cls.cpu().numpy().astype(int)
                self._last_infer_time = now

                # FPS based on completed inference passes, not raw capture loop.
                self._fps_frame_count += 1
                elapsed_time = now - self._fps_start_time
                if self._fps_frame_count >= 30 or elapsed_time >= 2.0:
                    if elapsed_time > 0:
                        self._fps = self._fps_frame_count / elapsed_time
                    self._fps_start_time = now
                    self._fps_frame_count = 0

            # 7) Zeichne die kombinierten Boxen auf dem vollen Kameraframe (sauberere Darstellung)
            annotated_full = raw_full.copy()
            frame_h, frame_w = annotated_full.shape[:2]
            scale_x = frame_w / float(self.model_input_size)
            scale_y = frame_h / float(self.model_input_size)
            roi_pixels = self._roi_pixels
            use_cached_detections = (
                self._last_infer_time > 0.0 and (now - self._last_infer_time) <= self._persist_seconds
            )
            boxes = self._cached_boxes if use_cached_detections else ()
            scores = self._cached_scores if use_cached_detections else ()
            classes = self._cached_classes if use_cached_detections else ()

            cropped_view = annotated_full
            for (x1, y1, x2, y2), conf, cls_id in zip(boxes, scores, classes):
                if roi_pixels:
                    rx1, ry1, rx2, ry2 = roi_pixels
                    cx = 0.5 * (x1 + x2)
                    cy = 0.5 * (y1 + y2)
                    if cx < rx1 or cx > rx2 or cy < ry1 or cy > ry2:
                        continue

                try:
                    class_name = self.model.names[cls_id]
                except (KeyError, IndexError):
                    class_name = f"class{cls_id}"
                normalized_class = (
                    str(class_name)
                    .lower()
                    .replace("-", "")
                    .replace("_", "")
                    .replace(" ", "")
                )
                is_nubs_helper = normalized_class in {"nubsup", "nubsdown"}
                box_color = (0, 165, 255) if is_nubs_helper else (0, 255, 0)

                # Reale Maße falls Ratio bekannt
                if (not is_nubs_helper) and self.last_known_ratio is not None:
                    w_mm, h_mm = calculate_dimensions([x1, y1, x2, y2], self.last_known_ratio)
                    if w_mm is not None and h_mm is not None:
                        label = f"{class_name} {conf:.2f}: {w_mm:.1f}x{h_mm:.1f}mm"
                    else:
                        label = f"{class_name} {conf:.2f}"
                else:
                    label = f"{class_name} {conf:.2f}"

                draw_x1 = int(round(x1 * scale_x))
                draw_y1 = int(round(y1 * scale_y))
                draw_x2 = int(round(x2 * scale_x))
                draw_y2 = int(round(y2 * scale_y))
                cv2.rectangle(annotated_full, (draw_x1, draw_y1), (draw_x2, draw_y2), box_color, 2)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_height = t_size[1] + 6
                if is_nubs_helper:
                    label_y1 = min(frame_h - label_height - 1, draw_y2 + 4)
                    label_y1 = max(0, label_y1)
                    label_y2 = min(frame_h - 1, label_y1 + label_height)
                    text_baseline = max(2, min(frame_h - 2, label_y2 - 4))
                else:
                    label_y2 = max(0, draw_y1)
                    label_y1 = max(0, label_y2 - label_height)
                    text_baseline = max(2, min(frame_h - 2, label_y2 - 4))
                cv2.rectangle(
                    annotated_full,
                    (draw_x1, label_y1),
                    (draw_x1 + t_size[0] + 6, label_y2),
                    box_color,
                    -1,
                )
                cv2.putText(
                    annotated_full,
                    label,
                    (draw_x1 + 3, text_baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    thickness=1,
                )

            if roi_pixels:
                rx1, ry1, rx2, ry2 = roi_pixels
                crop_x1 = max(0, min(frame_w - 1, int(round(rx1 * scale_x))))
                crop_y1 = max(0, min(frame_h - 1, int(round(ry1 * scale_y))))
                crop_x2 = max(1, min(frame_w, int(round(rx2 * scale_x))))
                crop_y2 = max(1, min(frame_h, int(round(ry2 * scale_y))))
                if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                    cropped_view = annotated_full[crop_y1:crop_y2, crop_x1:crop_x2].copy()

            display_frame = cropped_view
            display_h = display_frame.shape[0]

            # 8) FPS und Frame-Nummer einblenden
            fps_text = f"FPS: {self._fps:.1f}"
            cv2.putText(
                display_frame,
                fps_text,
                (5, display_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                thickness=1,
            )
            frame_text = f"Frame #{self._frame_counter}"
            cv2.putText(
                display_frame,
                frame_text,
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                thickness=1,
            )

            # 9) Speichern
            with self._lock:
                self._latest_annotated_full = annotated_full
                self._latest_annotated = display_frame

            time.sleep(0.001)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_annotated is None:
                return None
            return self._latest_annotated.copy()

    def get_latest_full_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_annotated_full is None:
                return None
            return self._latest_annotated_full.copy()

    def _apply_roi_locked(
        self, roi_rel: Optional[Tuple[float, float, float, float]]
    ) -> None:
        if roi_rel is None:
            self._roi_rel = None
            self._roi_pixels = None
            return
        x1_rel, y1_rel, x2_rel, y2_rel = roi_rel
        x1_rel = float(max(0.0, min(1.0, x1_rel)))
        y1_rel = float(max(0.0, min(1.0, y1_rel)))
        x2_rel = float(max(0.0, min(1.0, x2_rel)))
        y2_rel = float(max(0.0, min(1.0, y2_rel)))
        if x2_rel <= x1_rel or y2_rel <= y1_rel:
            self._roi_rel = None
            self._roi_pixels = None
            return
        self._roi_rel = (x1_rel, y1_rel, x2_rel, y2_rel)
        size = float(self.model_input_size)
        self._roi_pixels = (
            int(round(x1_rel * size)),
            int(round(y1_rel * size)),
            int(round(x2_rel * size)),
            int(round(y2_rel * size)),
        )

    def set_roi(self, roi_rel: Optional[Tuple[float, float, float, float]]) -> None:
        with self._lock:
            self._apply_roi_locked(roi_rel)

    def get_roi(self) -> Optional[Tuple[float, float, float, float]]:
        with self._lock:
            return None if self._roi_rel is None else tuple(self._roi_rel)

    def get_current_detections(self) -> List[Dict[str, Optional[float]]]:
        with self._lock:
            now = time.time()
            if (
                self._last_infer_time <= 0.0
                or (now - self._last_infer_time) > self._persist_seconds
            ):
                return []
            boxes = self._cached_boxes.copy()
            scores = self._cached_scores.copy()
            classes = self._cached_classes.copy()
            ratio = self.last_known_ratio
            roi_pixels = self._roi_pixels

        detections: List[Dict[str, Optional[float]]] = []
        for index, ((x1, y1, x2, y2), conf, cls_id) in enumerate(zip(boxes, scores, classes), start=1):
            if roi_pixels:
                rx1, ry1, rx2, ry2 = roi_pixels
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                if cx < rx1 or cx > rx2 or cy < ry1 or cy > ry2:
                    continue

            try:
                class_name = self.model.names[cls_id]
            except (KeyError, IndexError, TypeError):
                class_name = f"class{cls_id}"

            normalized_class = (
                str(class_name)
                .lower()
                .replace("-", "")
                .replace("_", "")
                .replace(" ", "")
            )
            if normalized_class in {"nubsup", "nubsdown"}:
                continue

            width_mm: Optional[float] = None
            height_mm: Optional[float] = None
            length_mm: Optional[float] = None
            if ratio is not None:
                width_mm, height_mm = calculate_dimensions([x1, y1, x2, y2], ratio)
                if width_mm is not None and height_mm is not None:
                    shorter_side_mm = min(width_mm, height_mm)
                    longer_side_mm = max(width_mm, height_mm)
                    width_mm = shorter_side_mm
                    height_mm = longer_side_mm
                    length_mm = longer_side_mm

            detections.append(
                {
                    "id": index,
                    "name": str(class_name),
                    "confidence": round(float(conf), 3),
                    "width_mm": None if width_mm is None else round(float(width_mm), 1),
                    "height_mm": None if height_mm is None else round(float(height_mm), 1),
                    "length_mm": None if length_mm is None else round(float(length_mm), 1),
                }
            )

        detections.sort(
            key=lambda item: (
                item["name"].lower(),
                -(item["length_mm"] or 0.0),
                -item["confidence"],
            )
        )
        return detections


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_qr_payload(detection: Dict[str, Optional[float]], zustand: Optional[str] = None) -> str:
    payload = {
        "typ": "komponente",
        "name": detection["name"],
        "laenge_mm": detection["length_mm"],
        "breite_mm": detection["width_mm"],
        "hoehe_mm": detection["height_mm"],
        "zustand": zustand,
        "erfasst_am": datetime.now().isoformat(timespec="seconds"),
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_qr_image(payload: str) -> Image.Image:
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=2,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGB")


def _build_label_image(
    detection: Dict[str, Optional[float]],
    qr_image: Image.Image,
    zustand: Optional[str] = None,
) -> Image.Image:
    # Vorschau für ein 50x30 mm Etikett bei 300 dpi.
    label_width_px = 591
    label_height_px = 354
    label = Image.new("RGB", (label_width_px, label_height_px), "white")
    draw = ImageDraw.Draw(label)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font_bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        bold_font = ImageFont.truetype(font_bold_path, 22)
        text_font = ImageFont.truetype(font_path, 22)
    except OSError:
        bold_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

    safe_top = 18
    safe_bottom = 30
    qr_size = 196
    qr_resized = qr_image.resize((qr_size, qr_size))
    qr_x = 24
    qr_y = safe_top + 18
    label.paste(qr_resized, (qr_x, qr_y))

    logo_path = Path(__file__).resolve().parents[3] / "static" / "system-180-logo.jpg"
    if logo_path.exists():
        try:
            logo = Image.open(logo_path).convert("RGB")
            target_width = 150
            target_height = 42
            logo.thumbnail((target_width, target_height))
            right_margin = 26
            logo_x = label_width_px - right_margin - logo.width
            logo_y = qr_y
            label.paste(logo, (logo_x, logo_y))
        except OSError:
            pass

    text_x = qr_x + qr_size + 30
    info_lines = [
        f"{detection['name']}",
        f"Länge: {detection['length_mm']:.1f} mm" if detection["length_mm"] is not None else "Länge: n. v.",
        f"Breite: {detection['width_mm']:.1f} mm" if detection["width_mm"] is not None else "Breite: n. v.",
        f"Zustand: {zustand or 'Gut'}",
    ]
    line_fonts = [bold_font, text_font, text_font, text_font]
    line_gap = 10
    text_heights = []
    for line, font in zip(info_lines, line_fonts):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_heights.append(bbox[3] - bbox[1])
    total_text_height = sum(text_heights) + line_gap * (len(info_lines) - 1)
    available_height = label_height_px - safe_top - safe_bottom
    text_y = safe_top + max(32, (available_height - total_text_height) // 2) - 28

    for index, (line, font) in enumerate(zip(info_lines, line_fonts)):
        draw.text((text_x, text_y), line, fill="black", font=font)
        text_y += text_heights[index] + line_gap

    return label


def _prepare_label_for_niimprint(
    label_image: Image.Image,
    max_width_px: int = 384,
    bottom_padding_px: int = 16,
) -> Image.Image:
    printable_image = label_image
    if printable_image.width > max_width_px:
        scale = max_width_px / float(printable_image.width)
        resized_height = max(1, int(round(printable_image.height * scale)))
        printable_image = printable_image.resize((max_width_px, resized_height), Image.Resampling.LANCZOS)

    canvas = Image.new(
        "RGB",
        (printable_image.width, printable_image.height + max(0, bottom_padding_px)),
        "white",
    )
    canvas.paste(printable_image, (0, 0))
    return canvas


def _print_label_via_usb(detection: Dict[str, Optional[float]], zustand: str) -> None:
    qr_payload = _build_qr_payload(detection, zustand=zustand)
    qr_image = _build_qr_image(qr_payload)
    label_image = _build_label_image(detection, qr_image, zustand=zustand)
    printable_image = _prepare_label_for_niimprint(label_image)
    printer = PrinterClient(SerialTransport(port="auto"))
    printer.print_image(printable_image, density=5)


# ============================================
# SegmentCamera (Masken-Inference + FPS)
# ============================================
class SegmentCamera(FrameGrabber):
    """
    Wrappt einen Roh-FrameGrabber (z.B. OAK1MaxCamera) mit einer YOLO-Detect-Engine.
    Zeigt ebenfalls FPS und Frame-Nummer an.
    """
    def __init__(
        self,
        base_cam: FrameGrabber,
        engine_path: str,
        model_input_size: int = YOLO_MODEL_INPUT_SIZE,
        conf_thresh: float = YOLO_CONF_THRESH,
        iou_thresh: float = YOLO_IOU_THRESH,
        max_det: int = YOLO_MAX_DET,
        device: str = YOLO_DEVICE,
        skip_frames: int = SKIP_FRAMES,
        batch_size: int = BATCH_SIZE,
        rotation: str = "none",  # Neu: Rotation Parameter
    ) -> None:
        super().__init__()
        self.base_cam = base_cam
        self.model_input_size = model_input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det
        self.device = device
        self.skip_frames = skip_frames
        self.batch_size = batch_size
        self.rotation = rotation  # Speichere Rotation

        # Left/right use detect engines in current config.
        self.model = YOLO(engine_path, task="detect")
        self.model.overrides["imgsz"]   = (model_input_size, model_input_size)
        self.model.overrides["conf"]    = conf_thresh
        self.model.overrides["iou"]     = iou_thresh
        self.model.overrides["max_det"] = max_det

        self._lock = threading.Lock()
        self._latest_annotated_320: Optional[np.ndarray] = None
        self._is_running = False
        self._frame_counter = 0
        self._fps_start_time = time.time()
        self._fps_frame_count = 0
        self._fps: float = 0.0
        self._persist_seconds = max(0.0, float(DETECTION_PERSIST_SECONDS))
        self._infer_min_interval = (
            0.0 if INFERENCE_MAX_FPS <= 0 else (1.0 / float(INFERENCE_MAX_FPS))
        )
        self._last_infer_time: float = 0.0
        self._cached_boxes: np.ndarray = np.empty((0, 4), dtype=np.int32)
        self._cached_scores: np.ndarray = np.empty((0,), dtype=np.float32)
        self._cached_classes: np.ndarray = np.empty((0,), dtype=np.int32)

    def start(self) -> None:
        self.base_cam.start()
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._is_running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)

    def _inference_loop(self) -> None:
        while self._is_running:
            raw_full = self.base_cam.get_latest_frame()
            if raw_full is None:
                time.sleep(0.001)
                continue

            self._frame_counter += 1
            now = time.time()
            should_infer_frame = (self._frame_counter % max(1, self.skip_frames)) == 0
            should_infer_time = (
                self._infer_min_interval <= 0.0
                or (now - self._last_infer_time) >= self._infer_min_interval
            )
            run_inference = should_infer_frame and should_infer_time

            if run_inference:
                # Downsample auf 320×320 und BGR→RGB
                frame_320 = cv2.resize(
                    raw_full,
                    (self.model_input_size, self.model_input_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                img_rgb = cv2.cvtColor(frame_320, cv2.COLOR_BGR2RGB)

                # YOLO Detect Inferenz
                results = self.model.predict(
                    source=[img_rgb],
                    imgsz=self.model_input_size,
                    device=self.device,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    max_det=self.max_det,
                    augment=False,
                    verbose=False,
                )
                self._cached_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                self._cached_scores = results[0].boxes.conf.cpu().numpy()
                self._cached_classes = results[0].boxes.cls.cpu().numpy().astype(int)
                self._last_infer_time = now

                # FPS based on completed inference passes, not raw capture loop.
                self._fps_frame_count += 1
                elapsed_time = now - self._fps_start_time
                if self._fps_frame_count >= 30 or elapsed_time >= 2.0:
                    if elapsed_time > 0:
                        self._fps = self._fps_frame_count / elapsed_time
                    self._fps_start_time = now
                    self._fps_frame_count = 0

            # Box-Overlay in voller Auflösung zeichnen (schärfer als 320er Upscale)
            annotated_full = raw_full.copy()
            frame_h, frame_w = annotated_full.shape[:2]
            scale_x = frame_w / float(self.model_input_size)
            scale_y = frame_h / float(self.model_input_size)
            use_cached_detections = (
                self._last_infer_time > 0.0 and (now - self._last_infer_time) <= self._persist_seconds
            )
            boxes = self._cached_boxes if use_cached_detections else ()
            scores = self._cached_scores if use_cached_detections else ()
            classes = self._cached_classes if use_cached_detections else ()
            for (x1, y1, x2, y2), conf, cls_id in zip(boxes, scores, classes):
                draw_x1 = int(round(x1 * scale_x))
                draw_y1 = int(round(y1 * scale_y))
                draw_x2 = int(round(x2 * scale_x))
                draw_y2 = int(round(y2 * scale_y))
                try:
                    class_name = self.model.names[cls_id]
                except (KeyError, IndexError, TypeError):
                    class_name = f"class{cls_id}"
                label_text = f"{class_name} {conf:.2f}"
                cv2.rectangle(annotated_full, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 0), 2)
                t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(
                    annotated_full,
                    (draw_x1, draw_y1 - t_size[1] - 6),
                    (draw_x1 + t_size[0] + 6, draw_y1),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    annotated_full,
                    label_text,
                    (draw_x1 + 3, draw_y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    thickness=1,
                )

            # Rotation ZUERST anwenden (vor FPS-Overlay)
            if self.rotation == "left_90":  # Linke Kamera: 90° im Uhrzeigersinn
                annotated_full = cv2.rotate(annotated_full, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation == "right_90":  # Rechte Kamera: 90° gegen Uhrzeigersinn
                annotated_full = cv2.rotate(annotated_full, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # NACH der Rotation: FPS-Anzeige
            fps_text = f"FPS: {self._fps:.1f}"
            cv2.putText(
                annotated_full,
                fps_text,
                (5, annotated_full.shape[0] - 5),  # Dynamische Position basierend auf aktueller Höhe
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                thickness=1,
            )

            # Frame-Nummer einblenden
            label = f"Frame #{self._frame_counter}"
            cv2.putText(
                annotated_full,
                label,
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                thickness=1,
            )

            with self._lock:
                self._latest_annotated_320 = annotated_full

            time.sleep(0.001)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_annotated_320 is None:
                return None
            return self._latest_annotated_320.copy()


oak_serials = list_oak_devices()
print(f"OAK Serials: {oak_serials}")
left_oak_serial, right_oak_serial = resolve_side_camera_serials(
    oak_serials,
    fallback_left_serial=LEFT_CAMERA_SERIAL,
    fallback_right_serial=RIGHT_CAMERA_SERIAL,
)
print(f"[INFO] Normal camera mapping: left={left_oak_serial}, right={right_oak_serial}")

left_engine_path = str(
    resolve_runtime_engine_path(
        Path(YOLO_ENGINE_LEFT),
        YOLO_SOURCE_LEFT,
        role_label="left side camera",
    )
)
right_engine_path = str(
    resolve_runtime_engine_path(
        Path(YOLO_ENGINE_RIGHT),
        YOLO_SOURCE_RIGHT,
        role_label="right side camera",
    )
)
center_engine_path = str(
    resolve_runtime_engine_path(
        Path(YOLO_ENGINE_CENTER),
        YOLO_SOURCE_CENTER,
        role_label="center camera",
    )
)


# ============================================
# Instanziiere alle Kameras & starte Threads
# ============================================
oak_left_raw   = OAK1MaxCamera(
    device_id=left_oak_serial,
    width=OAK_FRAME_WIDTH,
    height=OAK_FRAME_HEIGHT,
    fps=OAK_FPS,
    use_macro_focus=True,
    manual_focus=220,
    anti_banding=OAK_ANTI_BANDING,
    manual_exposure_us=OAK_MANUAL_EXPOSURE_US,
    manual_iso=OAK_MANUAL_ISO,
    luma_denoise=OAK_LUMA_DENOISE,
    chroma_denoise=OAK_CHROMA_DENOISE,
    sharpness=OAK_SHARPNESS,
)
oak_left_seg   = SegmentCamera(
    base_cam=oak_left_raw,
    engine_path=left_engine_path,
    model_input_size=YOLO_MODEL_INPUT_SIZE,
    conf_thresh=YOLO_CONF_THRESH,
    iou_thresh=YOLO_IOU_THRESH,
    max_det=YOLO_MAX_DET,
    device=YOLO_DEVICE,
    skip_frames=SKIP_FRAMES,
    batch_size=BATCH_SIZE,
    rotation="left_90",  # Linke Kamera: 90° im Uhrzeigersinn
)

oak_right_raw  = OAK1MaxCamera(
    device_id=right_oak_serial,
    width=OAK_FRAME_WIDTH,
    height=OAK_FRAME_HEIGHT,
    fps=OAK_FPS,
    use_macro_focus=True,
    manual_focus=220,
    anti_banding=OAK_ANTI_BANDING,
    manual_exposure_us=OAK_MANUAL_EXPOSURE_US,
    manual_iso=OAK_MANUAL_ISO,
    luma_denoise=OAK_LUMA_DENOISE,
    chroma_denoise=OAK_CHROMA_DENOISE,
    sharpness=OAK_SHARPNESS,
)
oak_right_seg  = SegmentCamera(
    base_cam=oak_right_raw,
    engine_path=right_engine_path,
    model_input_size=YOLO_MODEL_INPUT_SIZE,
    conf_thresh=YOLO_CONF_THRESH,
    iou_thresh=YOLO_IOU_THRESH,
    max_det=YOLO_MAX_DET,
    device=YOLO_DEVICE,
    skip_frames=SKIP_FRAMES,
    batch_size=BATCH_SIZE,
    rotation="right_90",  # Rechte Kamera: 90° gegen Uhrzeigersinn
)

usb_center_raw    = USBWebcamCamera(device_index=USB_DEVICE_INDEX, width=USB_CAPTURE_WIDTH, height=USB_CAPTURE_HEIGHT, fps=USB_CAPTURE_FPS)
usb_center_detect = DetectCamera(
    base_cam=usb_center_raw,
    engine_path=center_engine_path,
    model_input_size=YOLO_MODEL_INPUT_SIZE,
    conf_thresh=YOLO_CONF_THRESH,
    iou_thresh=YOLO_IOU_THRESH,
    max_det=YOLO_MAX_DET,
    device=YOLO_DEVICE,
    skip_frames=SKIP_FRAMES,
    batch_size=BATCH_SIZE,
)

oak_left_raw.start()
oak_left_seg.start()
oak_right_raw.start()
oak_right_seg.start()
usb_center_raw.start()
usb_center_detect.start()


class ROIUpdate(BaseModel):
    roi: Optional[List[float]]


class CameraMappingUpdate(BaseModel):
    left_serial: str
    right_serial: str


class PrintRequest(BaseModel):
    name: str
    zustand: str = "Gut"
    length_mm: Optional[float] = None
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None


# ============================================
# FASTAPI-Routen
# ============================================
@app.get("/", response_class=Response)
def index(request: Request) -> Response:
    return templates.TemplateResponse("normal_index.html", {"request": request})


@app.get("/api/erfassung")
def capture_components() -> Dict[str, object]:
    detections = usb_center_detect.get_current_detections()
    enriched: List[Dict[str, object]] = []
    for detection in detections:
        qr_payload = _build_qr_payload(detection, zustand="Gut")
        qr_image = _build_qr_image(qr_payload)
        label_image = _build_label_image(detection, qr_image, zustand="Gut")
        enriched.append(
            {
                **detection,
                "qr_payload": qr_payload,
                "qr_code_data_url": _image_to_data_url(qr_image),
                "etikett_data_url": _image_to_data_url(label_image),
            }
        )

    return {
        "komponenten": enriched,
        "anzahl": len(enriched),
        "erfasst_am": datetime.now().isoformat(timespec="seconds"),
    }


@app.get("/api/etikett-vorschau")
def label_preview(
    name: str,
    zustand: str = "Gut",
    length_mm: Optional[float] = None,
    width_mm: Optional[float] = None,
) -> Dict[str, str]:
    detection = {
        "name": name,
        "length_mm": length_mm,
        "width_mm": width_mm,
        "height_mm": None,
    }
    qr_payload = _build_qr_payload(detection, zustand=zustand)
    qr_image = _build_qr_image(qr_payload)
    label_image = _build_label_image(detection, qr_image, zustand=zustand)
    return {
        "etikett_data_url": _image_to_data_url(label_image),
    }


@app.post("/api/drucken")
def print_label(request: PrintRequest) -> Dict[str, object]:
    detection = {
        "name": request.name,
        "length_mm": request.length_mm,
        "width_mm": request.width_mm,
        "height_mm": request.height_mm,
    }
    try:
        _print_label_via_usb(detection, zustand=request.zustand)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Druck fehlgeschlagen: {exc}") from exc
    return {
        "success": True,
        "message": f"Etikett fuer {request.name} wurde an den Drucker gesendet.",
    }


@app.get("/video_left")
def video_left() -> StreamingResponse:
    # Linke OAK: Segment (Masken-Overlay + FPS) - Rotation in Kamera-Klasse
    return StreamingResponse(
        mjpeg_stream_generator(oak_left_seg),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.get("/video_center")
def video_center() -> StreamingResponse:
    # USB-Webcam: Ensemble-Detect (Bounding-Boxes + reale Maße + FPS)
    return StreamingResponse(
        mjpeg_stream_generator(usb_center_detect),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.get("/video_center_full")
def video_center_full() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_stream_generator(usb_center_raw),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.get("/video_right")
def video_right() -> StreamingResponse:
    # Rechte OAK: Segment (Masken-Overlay + FPS) - Rotation in Kamera-Klasse
    return StreamingResponse(
        mjpeg_stream_generator(oak_right_seg),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.get("/roi", response_class=Response)
def roi_calibration_page(request: Request) -> Response:
    return templates.TemplateResponse(
        "roi_calibration.html",
        {
            "request": request,
            "is_admin": True,
            "auth_required": False,
            "center_stream_src": "/video_center_full",
        },
    )


@app.get("/roi/config")
def roi_config() -> Dict[str, object]:
    roi = usb_center_detect.get_roi()
    return {
        "roi": list(roi) if roi else None,
        "image_size": [YOLO_MODEL_INPUT_SIZE, YOLO_MODEL_INPUT_SIZE],
    }


@app.post("/roi/config")
def roi_update(config: ROIUpdate) -> Dict[str, object]:
    global CENTER_CAMERA_ROI_REL, CURRENT_CENTER_CAMERA_ROI_REL
    roi_list = config.roi
    roi_tuple: Optional[Tuple[float, float, float, float]] = None
    if roi_list is not None:
        if len(roi_list) != 4:
            raise HTTPException(status_code=400, detail="ROI benötigt genau vier Werte (x1, y1, x2, y2).")
        roi_tuple = normalise_roi_tuple(roi_list)
        if roi_tuple is None:
            raise HTTPException(status_code=400, detail="Ungültige ROI-Koordinaten.")
    usb_center_detect.set_roi(roi_tuple)
    CENTER_CAMERA_ROI_REL = roi_tuple
    CURRENT_CENTER_CAMERA_ROI_REL = usb_center_detect.get_roi()
    persist_center_roi(CURRENT_CENTER_CAMERA_ROI_REL)
    return {
        "roi": list(CURRENT_CENTER_CAMERA_ROI_REL) if CURRENT_CENTER_CAMERA_ROI_REL else None,
        "image_size": [YOLO_MODEL_INPUT_SIZE, YOLO_MODEL_INPUT_SIZE],
    }


@app.post("/roi/reset")
def roi_reset() -> Dict[str, object]:
    global CENTER_CAMERA_ROI_REL, CURRENT_CENTER_CAMERA_ROI_REL
    usb_center_detect.set_roi(None)
    CENTER_CAMERA_ROI_REL = None
    CURRENT_CENTER_CAMERA_ROI_REL = None
    persist_center_roi(None)
    return {
        "roi": None,
        "image_size": [YOLO_MODEL_INPUT_SIZE, YOLO_MODEL_INPUT_SIZE],
    }


@app.get("/camera/config")
def camera_config() -> Dict[str, object]:
    saved_left_serial, saved_right_serial = get_persisted_side_camera_mapping()
    active_mapping = {
        "left": left_oak_serial,
        "right": right_oak_serial,
    }
    saved_mapping = {
        "left": saved_left_serial or left_oak_serial,
        "right": saved_right_serial or right_oak_serial,
    }
    return {
        "available_serials": oak_serials,
        "active_mapping": active_mapping,
        "saved_mapping": saved_mapping,
        "restart_required": active_mapping != saved_mapping,
    }


@app.post("/camera/config")
def camera_update(config: CameraMappingUpdate) -> Dict[str, object]:
    left_serial = config.left_serial.strip()
    right_serial = config.right_serial.strip()
    if not left_serial or not right_serial:
        raise HTTPException(status_code=400, detail="Bitte beide Kameras zuordnen.")
    if left_serial == right_serial:
        raise HTTPException(status_code=400, detail="Links und rechts müssen unterschiedliche Kameras sein.")
    if left_serial not in oak_serials or right_serial not in oak_serials:
        raise HTTPException(status_code=400, detail="Die ausgewählten Kameras sind aktuell nicht verfügbar.")

    persist_side_camera_mapping(left_serial, right_serial)
    return {
        "available_serials": oak_serials,
        "active_mapping": {
            "left": left_oak_serial,
            "right": right_oak_serial,
        },
        "saved_mapping": {
            "left": left_serial,
            "right": right_serial,
        },
        "restart_required": left_serial != left_oak_serial or right_serial != right_oak_serial,
    }


@app.on_event("shutdown")
def cleanup_cameras() -> None:
    usb_center_detect.stop()
    oak_left_seg.stop()
    oak_right_seg.stop()
    time.sleep(0.5)


# ============================================
# ENTRYPOINT
# ============================================
def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
