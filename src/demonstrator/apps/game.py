"""
Gamified sorting scenario for the System 180 demonstrator.

This script reuses the three-camera setup (two OAK-1 Max cameras and one USB webcam)
and adds a lightweight game layer on top of the YOLO-based object recognition.

The goal: untrained users pull furniture parts from the storage (left), hold them in
front of the demonstrator (center) and sort them into bins (right). The game tracks
which part should be presented next and gives immediate feedback on the center camera
stream. Additional REST endpoints expose the current game state so the UI can react.
"""

from __future__ import annotations

import secrets
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO

from demonstrator.apps.common import build_app, mjpeg_stream_generator, register_video_route
from demonstrator.vision.aruco import calculate_dimensions, detect_aruco_markers_with_tracking
from demonstrator.vision.camera import (
    FrameGrabber,
    OAK1MaxCamera,
    USBWebcamCamera,
    list_oak_devices,
)
from demonstrator.config.camera_ids import LEFT_CAMERA_SERIAL, RIGHT_CAMERA_SERIAL
from demonstrator.config.settings import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    ARUCO_ENABLED,
    ARUCO_SKIP_FRAMES,
    BATCH_SIZE,
    DEFAULT_CENTER_CAMERA_ROI_REL,
    INVENTORY_GERADE_BINS_MM,
    INVENTORY_GERADE_BIN_TOLERANCE_MM,
    INVENTORY_SORT_DURATION_S,
    OAK_FPS,
    OAK_FRAME_HEIGHT,
    OAK_FRAME_WIDTH,
    OAK_ANTI_BANDING,
    OAK_CHROMA_DENOISE,
    OAK_LUMA_DENOISE,
    OAK_MANUAL_EXPOSURE_US,
    OAK_MANUAL_ISO,
    OAK_SHARPNESS,
    PART_CATALOG,
    PART_LOOKUP,
    SESSION_COOKIE_NAME,
    SESSION_TTL_SECONDS,
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
    normalise_roi_tuple,
    persist_center_roi,
    resolve_runtime_engine_path,
    resolve_side_camera_serials,
)

# ============================================================
# FASTAPI BOOTSTRAP
# ============================================================
app, templates, video_dir = build_app(
    title="System180 Gamified Sorting",
    description="Adds a gamified feedback layer to the three-camera demonstrator.",
    log_prefix="Gamified Sorting",
)
register_video_route(app, video_dir, log_prefix="gamified")

def _initial_center_roi() -> Optional[Tuple[float, float, float, float]]:
    persisted = get_persisted_center_roi()
    if persisted is not None:
        return persisted
    return normalise_roi_tuple(DEFAULT_CENTER_CAMERA_ROI_REL)


CENTER_CAMERA_ROI_REL = _initial_center_roi()
CURRENT_CENTER_CAMERA_ROI_REL = CENTER_CAMERA_ROI_REL

ADMIN_SESSIONS: Dict[str, float] = {}

class SegmentCamera(FrameGrabber):
    """
    Wraps a FrameGrabber with a YOLO detect model and keeps annotated frames.
    Copy of the implementation in main.py so we can run standalone.
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
        rotation: str = "none",
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
        self.rotation = rotation

        self.model = YOLO(engine_path, task="detect")
        self.model.overrides["imgsz"] = (model_input_size, model_input_size)
        self.model.overrides["conf"] = conf_thresh
        self.model.overrides["iou"] = iou_thresh
        self.model.overrides["max_det"] = max_det

        self._lock = threading.Lock()
        self._latest_annotated_320: Optional[np.ndarray] = None
        self._latest_labels: List[str] = []
        self._is_running = False
        self._frame_counter = 0
        self._fps_start_time = time.time()
        self._fps_frame_count = 0
        self._fps: float = 0.0

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
            if (self._frame_counter % self.skip_frames) != 0:
                time.sleep(0.001)
                continue

            frame_320 = cv2.resize(
                raw_full,
                (self.model_input_size, self.model_input_size),
                interpolation=cv2.INTER_LINEAR,
            )
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
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            det_labels: List[str] = []
            if results and results[0].boxes.cls is not None:
                for cls_id in classes:
                    try:
                        raw_name = self.model.names[int(cls_id)]
                    except (KeyError, IndexError, TypeError):
                        continue
                    normalised = str(raw_name).lower().replace("-", " ").replace("/", " ")
                    normalised = "_".join(normalised.split())
                    det_labels.append(normalised)

            annotated_full = raw_full.copy()
            frame_h, frame_w = annotated_full.shape[:2]
            scale_x = frame_w / float(self.model_input_size)
            scale_y = frame_h / float(self.model_input_size)
            for (x1, y1, x2, y2), conf, cls_id in zip(boxes.astype(int), scores, classes):
                draw_x1 = int(round(x1 * scale_x))
                draw_y1 = int(round(y1 * scale_y))
                draw_x2 = int(round(x2 * scale_x))
                draw_y2 = int(round(y2 * scale_y))
                try:
                    class_name = self.model.names[int(cls_id)]
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

            if self.rotation == "left_90":
                annotated_full = cv2.rotate(annotated_full, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation == "right_90":
                annotated_full = cv2.rotate(annotated_full, cv2.ROTATE_90_COUNTERCLOCKWISE)

            fps_text = f"FPS: {self._fps:.1f}"
            cv2.putText(
                annotated_full,
                fps_text,
                (5, annotated_full.shape[0] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                thickness=1,
            )

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
                self._latest_labels = det_labels

            # FPS based on completed inference passes, not raw capture loop.
            self._fps_frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self._fps_start_time
            if self._fps_frame_count >= 30 or elapsed_time >= 2.0:
                if elapsed_time > 0:
                    self._fps = self._fps_frame_count / elapsed_time
                self._fps_start_time = current_time
                self._fps_frame_count = 0

            time.sleep(0.001)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_annotated_320 is None:
                return None
            return self._latest_annotated_320.copy()

    def get_latest_labels(self) -> List[str]:
        with self._lock:
            return list(self._latest_labels)

    def clear_latest_frame(self) -> None:
        with self._lock:
            self._latest_annotated_320 = None
            self._latest_labels = []


# ============================================================
# GAMIFICATION CORE
# ============================================================
def _normalise_label(label: str) -> str:
    label = label.lower().replace("-", " ").replace("/", " ").replace(".", " ")
    label = "_".join(label.split())
    return label


class SortingGame:
    """Tracks missions, score and feedback for the sorting challenge."""

    mode_id: str = "missions"
    mode_name: str = "Finde gesuchtes Bauteil"

    def __init__(
        self,
        parts_catalog: Iterable[Dict[str, object]],
        success_threshold: float = 0.55,
        cooldown_seconds: float = 2.0,
    ) -> None:
        self.success_threshold = success_threshold
        self.cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()

        self._parts: Dict[str, Dict[str, object]] = {}
        self._alias_lookup: Dict[str, str] = {}
        for part in parts_catalog:
            part_id = part["id"]
            self._parts[part_id] = part
            for alias in part.get("aliases", []):
                self._alias_lookup[_normalise_label(str(alias))] = part_id
            self._alias_lookup[_normalise_label(str(part["display_name"]))] = part_id
            self._alias_lookup[_normalise_label(str(part_id))] = part_id

        name_pool = [
            "Alex", "Noah", "Mila", "Finn", "Lina", "Ben", "Sofia", "Jonas", "Maja", "Liam",
            "Emil", "Nora", "Mara", "Luca", "Tessa", "Oskar", "Levi", "Romy", "Mika", "Leni",
        ]
        random.shuffle(name_pool)
        score_candidates = sorted(
            [random.randint(380, 940) for _ in range(5)], reverse=True
        )
        self._leaderboard_template: List[Dict[str, object]] = [
            {"name": name_pool[i], "score": score_candidates[i]}
            for i in range(5)
        ]

        self._all_part_ids: List[str] = list(self._parts.keys())
        self._mission_queue: Deque[str] = deque()

        self.round_limit = 5
        self.wins_to_win = 5
        self.max_losses = self.round_limit - self.wins_to_win + 1

        self.game_enabled = False
        self.round_active = False
        self.current_target_id: Optional[str] = None
        self.current_target_display: str = ""
        self.round_started_at: float = 0.0
        self.next_round_time: float = 0.0

        self.score = 0
        self.total_rounds = 0
        self.attempts = 0
        self.streak = 0
        self.best_streak = 0
        self.round_number = 0
        self.wins = 0
        self.losses = 0

        self.feedback_message: str = "Spiel bereit. Drücke Start, um ein Bauteil zu erhalten."
        self.feedback_level: str = "info"
        self.feedback_expires: float = time.time() + 3600.0
        self.last_detected: Optional[Dict[str, object]] = None
        self._last_feedback_key: Optional[Tuple[Optional[str], Optional[str]]] = None
        self._last_feedback_timestamp: float = 0.0
        self._instruction_text: str = ""
        self._instruction_until: float = 0.0
        self._round_pause_total: float = 0.0
        self._instruction_pause_active: bool = False
        self._instruction_pause_start: float = 0.0
        self._instruction_pause_end: float = 0.0

        now = time.time()
        with self._lock:
            self._reset_counters_locked(now)

    # ------------------------
    # INTERNAL HELPERS
    # ------------------------
    def _update_instruction_state_locked(self, now: float) -> None:
        if self._instruction_pause_active and now >= self._instruction_pause_end:
            self._round_pause_total += max(0.0, self._instruction_pause_end - self._instruction_pause_start)
            self._instruction_pause_active = False

    def _refill_mission_queue_locked(self, previous_target: Optional[str]) -> None:
        shuffled = self._all_part_ids.copy()
        random.shuffle(shuffled)
        if previous_target and len(shuffled) > 1 and shuffled[0] == previous_target:
            shuffled.append(shuffled.pop(0))
        self._mission_queue.extend(shuffled)

    def _reset_counters_locked(self, now: float) -> None:
        self.round_active = False
        self.current_target_id = None
        self.current_target_display = ""
        self.round_started_at = now
        self.next_round_time = now
        self.score = 0
        self.total_rounds = 0
        self.attempts = 0
        self.streak = 0
        self.best_streak = 0
        self.round_number = 0
        self.wins = 0
        self.losses = 0
        self.last_detected = None
        self._last_feedback_key = None
        self._last_feedback_timestamp = now
        self._instruction_text = ""
        self._instruction_until = 0.0
        self._round_pause_total = 0.0
        self._instruction_pause_active = False
        self._instruction_pause_start = 0.0
        self._instruction_pause_end = 0.0
        self._mission_queue.clear()

    def _finish_match_locked(
        self,
        now: float,
        victory: Optional[bool] = None,
        reason: str = "",
    ) -> None:
        self._update_instruction_state_locked(now)
        if victory is None:
            victory = self.wins >= self.wins_to_win

        self.game_enabled = False
        self.round_active = False
        self.current_target_id = None
        self.current_target_display = ""
        self.next_round_time = now
        outcome_text = "Du hast gewonnen!" if victory else "Du hast verloren."
        if not reason:
            reason = f"(Endstand {self.wins}:{self.losses})"
        if reason:
            outcome_text += f" {reason}"
        self.feedback_message = outcome_text + " Starte ein neues Spiel, um weiterzumachen."
        self.feedback_level = "success" if victory else "warn"
        self.feedback_expires = now + 3600.0

    def _start_new_round_locked(self, now: float, announce: bool = True) -> None:
        self._update_instruction_state_locked(now)
        if (
            self.wins >= self.wins_to_win
            or self.losses >= self.max_losses
            or self.round_number >= self.round_limit
        ):
            self._finish_match_locked(now)
            return

        previous_target = self.current_target_id
        if not self._mission_queue:
            self._refill_mission_queue_locked(previous_target)

        if self._mission_queue:
            next_part_id = self._mission_queue.popleft()
            if previous_target and next_part_id == previous_target:
                if self._mission_queue:
                    self._mission_queue.append(next_part_id)
                    next_part_id = self._mission_queue.popleft()
                else:
                    alternatives = [pid for pid in self._all_part_ids if pid != previous_target]
                    next_part_id = random.choice(alternatives or self._all_part_ids)
        else:
            alternatives = [pid for pid in self._all_part_ids if pid != previous_target]
            next_part_id = random.choice(alternatives or self._all_part_ids)

        self.current_target_id = str(next_part_id)
        part = self._parts[self.current_target_id]
        self.current_target_display = str(part["display_name"])
        self.round_number += 1
        self.round_active = True
        self.round_started_at = now
        self.next_round_time = now
        self.total_rounds += 1
        self.last_detected = None
        self._last_feedback_key = None
        self._last_feedback_timestamp = now
        self._round_pause_total = 0.0
        self._instruction_pause_active = False
        if announce:
            mission_desc = self.current_target_display
            self.feedback_message = (
                f"Runde {self.round_number}/{self.round_limit}: Zeige eine {mission_desc}!"
            )
            self.feedback_level = "info"
            self.feedback_expires = now + 4.0

    def _ensure_round_active_locked(self, now: float) -> None:
        if not self.game_enabled or self.round_active:
            return

        if (
            self.wins >= self.wins_to_win
            or self.losses >= self.max_losses
            or self.round_number >= self.round_limit
        ):
            self._finish_match_locked(now)
            return

        if now >= self.next_round_time:
            self._start_new_round_locked(now)

    # ------------------------
    # PUBLIC API
    # ------------------------
    def ensure_round_active(self) -> None:
        with self._lock:
            now = time.time()
            self._update_instruction_state_locked(now)
            if not self.game_enabled:
                return
            self._ensure_round_active_locked(now)

    def start_game(self) -> None:
        with self._lock:
            now = time.time()
            self.game_enabled = True
            self._reset_counters_locked(now)
            self.feedback_message = "Spiel gestartet! Zeige die passenden Bauteile."
            self.feedback_level = "info"
            self.feedback_expires = now + 3.0
            self._start_new_round_locked(now)

    def force_new_round(self) -> None:
        with self._lock:
            now = time.time()
            self._update_instruction_state_locked(now)
            if not self.game_enabled:
                self.game_enabled = True
                self._reset_counters_locked(now)
                self._start_new_round_locked(now)
                return

            if self.round_active:
                self.round_active = False
                self.feedback_message = "Bauteil übersprungen. Neues Bauteil wird geladen."
                self.feedback_level = "info"
                self.feedback_expires = now + 2.5

            if self.wins >= self.wins_to_win:
                self._finish_match_locked(
                    now,
                    victory=True,
                    reason=f"(Endstand {self.wins}:{self.losses})",
                )
                return

            if self.round_number >= self.round_limit:
                self._finish_match_locked(
                    now,
                    victory=self.wins >= self.wins_to_win,
                    reason=f"(Endstand {self.wins}:{self.losses})",
                )
                return

            self.next_round_time = now
            self._start_new_round_locked(now)

    def reset(self) -> None:
        with self._lock:
            now = time.time()
            self.game_enabled = False
            self._reset_counters_locked(now)
            self.feedback_message = "Spiel zurückgesetzt. Drücke Start, um weiterzumachen."
            self.feedback_level = "info"
            self.feedback_expires = now + 3600.0

    def map_label_to_part_id(self, raw_label: str) -> Optional[str]:
        return self._alias_lookup.get(_normalise_label(raw_label))

    def _build_leaderboard_locked(self) -> List[Dict[str, object]]:
        entries = [dict(entry) for entry in self._leaderboard_template]
        player_entry = {"name": "Du", "score": self.score, "is_player": True}
        entries.append(player_entry)
        entries.sort(key=lambda entry: entry.get("score", 0), reverse=True)

        leaderboard: List[Dict[str, object]] = []
        player_included = False
        for entry in entries:
            leaderboard.append(entry)
            if entry.get("is_player"):
                player_included = True
            if len(leaderboard) >= 5:
                break

        if not player_included:
            player_entry["rank"] = entries.index(player_entry) + 1
            leaderboard.append(player_entry)

        for idx, entry in enumerate(leaderboard, start=1):
            entry.setdefault("rank", idx)
            entry.setdefault("is_player", False)

        return leaderboard

    def show_instruction(self, text: str, duration: float = 4.0, pause_timer: bool = False) -> None:
        with self._lock:
            now = time.time()
            self._update_instruction_state_locked(now)
            if self.round_active and pause_timer:
                if self._instruction_pause_active:
                    self._round_pause_total += max(0.0, now - self._instruction_pause_start)
                self._instruction_pause_active = True
                self._instruction_pause_start = now
                self._instruction_pause_end = now + duration
            self._instruction_text = text
            self._instruction_until = now + duration
            self.feedback_message = text
            self.feedback_level = "info"
            self.feedback_expires = now + duration
            self.last_instruction_message = text

    def update_with_detections(self, detections: Iterable[Dict[str, object]]) -> None:
        now = time.time()
        with self._lock:
            self._update_instruction_state_locked(now)
            if not self.game_enabled:
                return
            self._ensure_round_active_locked(now)
            if not self.round_active:
                return

        detections = list(detections)
        if not detections:
            return

        best_detection: Optional[Tuple[str, float, Optional[str], str, Optional[float]]] = None
        for det in detections:
            raw_label = str(det.get("raw_label", ""))
            confidence = float(det.get("confidence", 0.0))
            part_id = self.map_label_to_part_id(raw_label)
            display_name = (
                str(self._parts[part_id]["display_name"]) if part_id else raw_label
            )
            length_mm = det.get("length_mm")
            if best_detection is None or confidence > best_detection[1]:
                best_detection = (raw_label, confidence, part_id, display_name, length_mm)

        if best_detection is None:
            return

        raw_label, confidence, part_id, display_name, length_mm = best_detection
        feedback_key = (self.current_target_id, part_id)
        self.last_detected = {
            "id": part_id,
            "display": display_name,
            "confidence": confidence,
            "length_mm": length_mm,
        }

        if part_id is None:
            if (
                feedback_key == self._last_feedback_key
                and (now - self._last_feedback_timestamp) < 1.0
            ):
                return
            self.feedback_message = f"Unbekanntes Teil erkannt ({raw_label})."
            self.feedback_level = "warn"
            self.feedback_expires = now + 2.0
            self._last_feedback_key = feedback_key
            self._last_feedback_timestamp = now
            return

        if (
            part_id != self.current_target_id
            and feedback_key == self._last_feedback_key
            and (now - self._last_feedback_timestamp) < 1.0
        ):
            # Avoid spamming the same warning multiple times per second.
            return

        if part_id == self.current_target_id and confidence >= self.success_threshold:
            points = 100 + 20 * self.streak
            self.score += points
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
            self.attempts += 1
            self.wins += 1
            self.round_active = False
            self.next_round_time = now + self.cooldown_seconds
            current_score = f"{self.wins}:{self.losses}"
            if (
                self.wins >= self.wins_to_win
                or self.round_number >= self.round_limit
            ):
                self._last_feedback_key = feedback_key
                self._last_feedback_timestamp = now
                self._finish_match_locked(
                    now,
                    victory=self.wins >= self.wins_to_win,
                    reason=f"(Endstand {current_score})",
                )
                return
            else:
                self.feedback_message = (
                    f"Treffer! {self.current_target_display} erkannt (+{points} Punkte). "
                    f"Zwischenstand {current_score}."
                )
                self.feedback_level = "success"
                self.feedback_expires = now + 3.0
        elif part_id == self.current_target_id:
            self.streak = 0
            self.attempts += 1
            self.feedback_message = (
                f"Fast! {display_name} erkannt, aber Sicherheit zu gering "
                f"({confidence * 100:.0f}%). Halte das Teil ruhiger."
            )
            self.feedback_level = "warn"
            self.feedback_expires = now + 3.0
        else:
            self.streak = 0
            self.attempts += 1
            target = self.current_target_display
            self.feedback_message = f"Das ist {display_name}. Gesucht: {target}."
            self.feedback_level = "warn"
            self.feedback_expires = now + 3.0

        self._last_feedback_key = feedback_key
        self._last_feedback_timestamp = now

    def get_state(self) -> Dict[str, object]:
        now = time.time()
        with self._lock:
            self._update_instruction_state_locked(now)
            if self.game_enabled:
                self._ensure_round_active_locked(now)
            cooldown_remaining = max(0.0, self.next_round_time - now)
            if self.round_active:
                effective_pause = self._round_pause_total
                if self._instruction_pause_active:
                    effective_pause += max(0.0, now - self._instruction_pause_start)
                round_time = max(0.0, (now - self.round_started_at) - effective_pause)
            else:
                round_time = 0.0
            if not self.game_enabled:
                feedback = self.feedback_message
            else:
                feedback = self.feedback_message if now <= self.feedback_expires else ""
        state = {
            "game_enabled": self.game_enabled,
            "round_active": self.round_active,
            "target": None,
            "score": self.score,
            "rounds": self.total_rounds,
            "attempts": self.attempts,
            "streak": self.streak,
            "best_streak": self.best_streak,
            "round_number": self.round_number,
            "round_limit": self.round_limit,
            "wins_to_win": self.wins_to_win,
            "wins": self.wins,
            "losses": self.losses,
            "max_losses": self.max_losses,
            "round_time": round_time,
            "cooldown_remaining": cooldown_remaining,
            "feedback": feedback,
            "feedback_level": self.feedback_level,
            "last_detected": self.last_detected,
            "leaderboard": self._build_leaderboard_locked(),
            "instruction": self._instruction_text if now <= self._instruction_until else "",
            "mode": self.mode_id,
            "mode_name": self.mode_name,
        }
        if self.current_target_id:
            state["target"] = {
                "id": self.current_target_id,
                "display": self.current_target_display,
            }
        return state

    def peek_target_id(self) -> Optional[str]:
        with self._lock:
            if not self.game_enabled:
                return None
            return self.current_target_id


class InventorySortGame:
    mode_id: str = "inventory"
    mode_name: str = "Lagere Bauteile ein"

    def __init__(
        self,
        parts_catalog: Iterable[Dict[str, object]],
        duration_seconds: float = 90.0,
        gerade_bins_mm: Optional[List[float]] = None,
        gerade_tolerance_mm: float = 15.0,
    ) -> None:
        self.round_duration = duration_seconds
        self.gerade_bins_mm = list(
            gerade_bins_mm if gerade_bins_mm is not None else [180.0, 270.0, 360.0, 450.0]
        )
        self.gerade_bins_mm.sort()
        self.gerade_tolerance_mm = gerade_tolerance_mm

        self.num_length_bins: int = len(self.gerade_bins_mm)
        self._length_labels: Dict[int, str] = {
            idx + 1: f"ca. {length:.0f} mm" for idx, length in enumerate(self.gerade_bins_mm)
        }
        self._column_mapping = {
            "gerade": {"column": 1, "label": "Geraden"},
            "diagonale": {"column": 2, "label": "Diagonalen"},
            "mutternstab": {"column": 3, "label": "Mutternstäbe"},
            "versetzte_gerade": {"column": 4, "label": "Versetzte Geraden"},
        }

        self._lock = threading.Lock()
        self._parts: Dict[str, Dict[str, object]] = {}
        self._alias_lookup: Dict[str, str] = {}
        for part in parts_catalog:
            part_id = part["id"]
            self._parts[part_id] = part
            for alias in part.get("aliases", []):
                self._alias_lookup[_normalise_label(str(alias))] = part_id
            self._alias_lookup[_normalise_label(str(part["display_name"]))] = part_id
            self._alias_lookup[_normalise_label(str(part_id))] = part_id

        self.storage_counts: Dict[str, int] = {pid: 0 for pid in self._parts.keys()}

        self.game_enabled = False
        self.round_active = False
        self.round_started_at: float = 0.0
        self._round_pause_total: float = 0.0
        self._instruction_pause_active = False
        self._instruction_pause_start: float = 0.0
        self._instruction_pause_end: float = 0.0

        self.score: int = 0
        self.sorted_count: int = 0
        self.missed_count: int = 0
        self.attempts: int = 0

        self.feedback_message: str = "Bereit. Starte das Spiel, um Bauteile zu sortieren."
        self.feedback_level: str = "info"
        self.feedback_expires: float = time.time() + 3600.0
        self._instruction_text: str = ""
        self._instruction_until: float = 0.0

        self.last_detected: Optional[Dict[str, object]] = None
        self._last_feedback_key: Optional[Tuple[Optional[str], Optional[str]]] = None
        self._last_feedback_timestamp: float = 0.0

        self.last_instruction_message: str = "Halte ein Bauteil in die Kamera."
        self.highlight_row: Optional[int] = None
        self.highlight_column: Optional[int] = None
        self.highlight_area: Optional[str] = None
        self.awaiting_clear_frame: bool = False
        self._awaiting_check_counter: int = 0
        self._awaiting_expected: Optional[Dict[str, object]] = None
        self.multiple_detected: bool = False
        self.last_success_time: float = time.time()

    # ------------------------
    # Instruction helpers
    # ------------------------
    def _update_instruction_state_locked(self, now: float) -> None:
        if self._instruction_pause_active and now >= self._instruction_pause_end:
            self._round_pause_total += max(0.0, self._instruction_pause_end - self._instruction_pause_start)
            self._instruction_pause_active = False

    def show_instruction(self, text: str, duration: float = 4.0, pause_timer: bool = False) -> None:
        with self._lock:
            now = time.time()
            self._update_instruction_state_locked(now)
            if self.round_active and pause_timer:
                if self._instruction_pause_active:
                    self._round_pause_total += max(0.0, now - self._instruction_pause_start)
                self._instruction_pause_active = True
                self._instruction_pause_start = now
                self._instruction_pause_end = now + duration
            self._instruction_text = text
            self._instruction_until = now + duration
            self.feedback_message = text
            self.feedback_level = "info"
            self.feedback_expires = now + duration
            self.last_instruction_message = text
            self.highlight_row = None
            self.highlight_column = None
            self.highlight_area = None
            self.awaiting_clear_frame = False
            self._awaiting_check_counter = 0
            self._awaiting_expected = None

    def _begin_placement_pause_locked(self, now: float, expected: Dict[str, object]) -> None:
        if self._instruction_pause_active:
            self._round_pause_total += max(0.0, now - self._instruction_pause_start)
            self._instruction_pause_active = False
            self._instruction_pause_start = 0.0
            self._instruction_pause_end = 0.0
        self.awaiting_clear_frame = True
        self._awaiting_check_counter = 0
        self._awaiting_expected = expected
        self.multiple_detected = False

    def _finish_placement_pause_locked(self, now: float) -> None:
        if not self.awaiting_clear_frame:
            return
        if self._instruction_pause_active:
            self._round_pause_total += max(0.0, now - self._instruction_pause_start)
        self._instruction_pause_active = False
        self._instruction_pause_start = 0.0
        self._instruction_pause_end = 0.0
        self.awaiting_clear_frame = False
        self._awaiting_expected = None
        self._awaiting_check_counter = 0
        self.feedback_message = "Bereit für das nächste Bauteil."
        self.feedback_level = "info"
        self.feedback_expires = now + 3.0
        self.last_instruction_message = "Halte ein Bauteil in die Kamera."
        self.highlight_row = None
        self.highlight_column = None
        self.highlight_area = None
        self.multiple_detected = False

    # ------------------------
    # Game lifecycle
    # ------------------------
    def start_game(self) -> None:
        with self._lock:
            now = time.time()
            self._update_instruction_state_locked(now)
            self.storage_counts = {pid: 0 for pid in self._parts.keys()}
            self.score = 0
            self.sorted_count = 0
            self.missed_count = 0
            self.attempts = 0
            self.last_detected = None
            self.game_enabled = True
            self.round_active = True
            self.round_started_at = now
            self._round_pause_total = 0.0
            self._instruction_pause_active = False
            self.feedback_message = "Lagerspiel gestartet! Halte ein Bauteil in die Kamera."
            self.feedback_level = "info"
            self.feedback_expires = now + 3.0
            self.last_instruction_message = "Halte ein Bauteil in die Kamera."
            self.highlight_row = None
            self.highlight_column = None
            self.highlight_area = None
            self.awaiting_clear_frame = False
            self._awaiting_check_counter = 0
            self._awaiting_expected = None
            self.multiple_detected = False
            self.last_success_time = now

    def ensure_round_active(self) -> None:
        with self._lock:
            now = time.time()
            self._update_instruction_state_locked(now)
            if not self.game_enabled or not self.round_active:
                return
            elapsed = now - self.round_started_at
            effective_pause = self._round_pause_total
            if self._instruction_pause_active:
                effective_pause += max(0.0, now - self._instruction_pause_start)
            if (elapsed - effective_pause) >= self.round_duration:
                self.round_active = False
                self.game_enabled = False
                self.feedback_message = "Zeit ist abgelaufen! Starte neu für die nächste Runde."
                self.feedback_level = "warn"
                self.feedback_expires = now + 6.0

    def force_new_round(self) -> None:
        with self._lock:
            if not self.game_enabled:
                self.start_game()
            else:
                self.start_game()

    def reset(self) -> None:
        with self._lock:
            self.game_enabled = False
            self.round_active = False
            self.round_started_at = time.time()
            self._round_pause_total = 0.0
            self._instruction_pause_active = False
            self._instruction_pause_start = 0.0
            self._instruction_pause_end = 0.0
            self.score = 0
            self.sorted_count = 0
            self.missed_count = 0
            self.attempts = 0
            self.last_detected = None
            self.feedback_message = "Lagerspiel bereit. Starte, um Bauteile zu sortieren."
            self.feedback_level = "info"
            self.feedback_expires = time.time() + 3600.0
            self._instruction_text = ""
            self._instruction_until = 0.0
            self.last_instruction_message = "Halte ein Bauteil in die Kamera."
            self.highlight_row = None
            self.highlight_column = None
            self.highlight_area = None
            self.awaiting_clear_frame = False
            self._awaiting_check_counter = 0
            self._awaiting_expected = None
            self.multiple_detected = False
            self.last_success_time = time.time()

    # ------------------------
    # Detection helpers
    # ------------------------
    def map_label_to_part_id(self, raw_label: str) -> Optional[str]:
        return self._alias_lookup.get(_normalise_label(raw_label))

    def peek_target_id(self) -> Optional[str]:
        return None

    def _time_remaining_locked(self, now: float) -> float:
        if not self.round_active:
            return 0.0
        effective_pause = self._round_pause_total
        if self._instruction_pause_active:
            effective_pause += max(0.0, now - self._instruction_pause_start)
        elapsed = max(0.0, now - self.round_started_at - effective_pause)
        return max(0.0, self.round_duration - elapsed)

    def _match_length_to_row(self, length_mm: Optional[float]) -> Optional[int]:
        if length_mm is None or not self.gerade_bins_mm:
            return None
        diffs = [abs(length_mm - target) for target in self.gerade_bins_mm]
        min_idx = int(min(range(len(diffs)), key=diffs.__getitem__))
        if diffs[min_idx] <= self.gerade_tolerance_mm:
            return min_idx + 1
        return None

    def _assign_storage_slot(
        self, part_id: str, length_mm: Optional[float] = None
    ) -> Tuple[str, Optional[int], Optional[int], str]:
        if part_id == "noppenstein":
            return ("BOX", None, None, "Lege die Noppensteine in die Box auf dem Lagerregal.")

        column_info = self._column_mapping.get(part_id)
        if not column_info:
            return ("UNBEKANNT", None, None, "Bauteil nicht im Lagerplan.")

        count = self.storage_counts.get(part_id, 0)
        row = self._match_length_to_row(length_mm)
        if row is None:
            if self.num_length_bins > 0:
                row = (count % self.num_length_bins) + 1
            else:
                row = 1

        column = column_info["column"]
        column_label = column_info["label"]
        row_label = self._length_labels.get(row, f"Reihe {row}")
        message = (
            f"Lege das Bauteil in Reihe {row} ({row_label}), "
            f"Spalte {column} ({column_label})."
        )
        return ("GRID", row, column, message)

    def _select_best_detection(
        self, detections: Iterable[Dict[str, object]]
    ) -> Optional[Tuple[str, float, Optional[str], str, Optional[float]]]:
        best_detection: Optional[Tuple[str, float, Optional[str], str, Optional[float]]] = None
        for det in detections:
            raw_label = str(det.get("raw_label", ""))
            confidence = float(det.get("confidence", 0.0))
            part_id = self.map_label_to_part_id(raw_label)
            display_name = (
                str(self._parts[part_id]["display_name"]) if part_id else raw_label
            )
            length_mm = det.get("length_mm")
            if best_detection is None or confidence > best_detection[1]:
                best_detection = (raw_label, confidence, part_id, display_name, length_mm)
        return best_detection

    def _apply_instruction_from_detection_locked(
        self,
        part_id: Optional[str],
        display_name: str,
        length_mm: Optional[float],
        now: float,
        *,
        feedback_level: str = "success",
        set_expected: bool = True,
    ) -> Optional[Dict[str, object]]:
        if part_id is None:
            message = f"Unbekanntes Teil erkannt ({display_name})."
            self.feedback_message = message
            self.feedback_level = "warn"
            self.feedback_expires = now + 3.0
            self.last_instruction_message = message
            self.highlight_row = None
            self.highlight_column = None
            self.highlight_area = None
            if set_expected:
                self._awaiting_expected = None
            self.multiple_detected = False
            return None

        if part_id == "gerade":
            if length_mm is None:
                message = "Gerade erkannt, aber Länge fehlt. Bitte Marker sichtbar halten."
                self.feedback_message = message
                self.feedback_level = "warn"
                self.feedback_expires = now + 3.0
                self.last_instruction_message = message
                self.highlight_row = None
                self.highlight_column = None
                self.highlight_area = None
                if set_expected:
                    self._awaiting_expected = None
                self.multiple_detected = False
                return None
            diffs = [abs(length_mm - bin_mm) for bin_mm in self.gerade_bins_mm]
            min_idx = int(min(range(len(diffs)), key=diffs.__getitem__))
            if diffs[min_idx] > self.gerade_tolerance_mm:
                message = (
                    f"Gerade mit {length_mm:.0f} mm passt in kein Lagerfach. "
                    "Bitte lege das Teil zurück in die Schublade."
                )
                self.feedback_message = message
                self.feedback_level = "warn"
                self.feedback_expires = now + 3.0
                self.last_instruction_message = message
                self.highlight_row = None
                self.highlight_column = None
                self.highlight_area = None
                if set_expected:
                    self._awaiting_expected = None
                self.multiple_detected = False
                return None
            target_length = self.gerade_bins_mm[min_idx]
            row = min_idx + 1
            column_info = self._column_mapping.get("gerade")
            column = column_info["column"] if column_info else 1
            column_label = column_info["label"] if column_info else "Geraden"
            row_label = self._length_labels.get(row, f"{target_length:.0f} mm")
            message = (
                f"Gerade {target_length:.0f} mm erkannt. Lege sie in Reihe {row} "
                f"({row_label}), Spalte {column} ({column_label})."
            )
            expiry = now + (4.0 if feedback_level == "success" else 3.0)
            self.feedback_message = message
            self.feedback_level = feedback_level
            self.feedback_expires = expiry
            self.last_instruction_message = message
            self.highlight_row = row
            self.highlight_column = column
            self.highlight_area = "grid"
            expected_info = {
                "part_id": part_id,
                "area": "grid",
                "row": row,
                "column": column,
                "length_mm": target_length,
            }
            if set_expected:
                self._awaiting_expected = expected_info
            self.multiple_detected = False
            return expected_info

        if part_id in {"noppenstein", "noppenscheiben", "schraube"}:
            if part_id == "schraube":
                message = "Schraube erkannt. Lege sie in die Box auf dem Lagerregal."
            elif part_id == "noppenscheiben":
                message = "Noppenscheiben erkannt. Lege sie in die Box auf dem Lagerregal."
            else:
                message = "Noppenstein erkannt. Lege ihn in die Box auf dem Lagerregal."
            expiry = now + (4.0 if feedback_level == "success" else 3.0)
            self.feedback_message = message
            self.feedback_level = feedback_level
            self.feedback_expires = expiry
            self.last_instruction_message = message
            self.highlight_row = None
            self.highlight_column = None
            self.highlight_area = "box"
            expected_info = {"part_id": part_id, "area": "box"}
            if set_expected:
                self._awaiting_expected = expected_info
            self.multiple_detected = False
            return expected_info

        area_type, row, column, message = self._assign_storage_slot(part_id, length_mm)
        if area_type == "UNBEKANNT":
            warn_message = f"{message} Bitte lege das Teil zurück in die Schublade."
            self.feedback_message = warn_message
            self.feedback_level = "warn"
            self.feedback_expires = now + 3.0
            self.last_instruction_message = warn_message
            self.highlight_row = None
            self.highlight_column = None
            self.highlight_area = None
            if set_expected:
                self._awaiting_expected = None
            self.multiple_detected = False
            return None

        area = str(area_type).lower()
        expiry = now + (4.0 if feedback_level == "success" else 3.0)
        self.feedback_message = message
        self.feedback_level = feedback_level
        self.feedback_expires = expiry
        self.last_instruction_message = message
        if area == "grid":
            self.highlight_row = row
            self.highlight_column = column
            self.highlight_area = "grid"
        else:
            self.highlight_row = None
            self.highlight_column = None
            self.highlight_area = "box"
        expected_info = {
            "part_id": part_id,
            "area": self.highlight_area,
            "row": self.highlight_row,
            "column": self.highlight_column,
        }
        if set_expected:
            self._awaiting_expected = expected_info
        self.multiple_detected = False
        return expected_info

    def _validate_awaiting_detection_locked(
        self, detections: Iterable[Dict[str, object]], now: float
    ) -> None:
        recognized_parts = {
            self.map_label_to_part_id(str(det.get("raw_label", "")))
            for det in detections
        }
        recognized_parts.discard(None)

        if len(recognized_parts) > 1:
            message = "Mehrere Bauteile erkannt. Bitte nur ein Bauteil im Demonstrator lassen."
            self.feedback_message = message
            self.feedback_level = "warn"
            self.feedback_expires = now + 3.0
            self.last_instruction_message = message
            self.highlight_row = None
            self.highlight_column = None
            self.highlight_area = None
            self._awaiting_expected = None
            self.multiple_detected = True
            return

        best_detection = self._select_best_detection(detections)
        if best_detection is None:
            message = "Bauteil konnte nicht stabil erkannt werden. Bitte ruhig halten."
            self.feedback_message = message
            self.feedback_level = "warn"
            self.feedback_expires = now + 2.0
            self.last_instruction_message = message
            self.highlight_row = None
            self.highlight_column = None
            self.highlight_area = None
            self._awaiting_expected = None
            self.multiple_detected = False
            return

        raw_label, confidence, part_id, display_name, length_mm = best_detection
        self.last_detected = {
            "id": part_id,
            "display": display_name,
            "confidence": confidence,
            "length_mm": length_mm,
        }

        prev_expected = self._awaiting_expected
        feedback_level = "success"
        if prev_expected is None or part_id != prev_expected.get("part_id"):
            feedback_level = "info"

        self._apply_instruction_from_detection_locked(
            part_id,
            display_name,
            length_mm,
            now,
            feedback_level=feedback_level,
            set_expected=True,
        )

    def update_with_detections(self, detections: Iterable[Dict[str, object]]) -> None:
        detections_list = list(detections)
        now = time.time()
        with self._lock:
            self._update_instruction_state_locked(now)
            if not self.game_enabled or not self.round_active:
                return

            remaining = self._time_remaining_locked(now)
            if remaining <= 0:
                self.round_active = False
                self.game_enabled = False
                self.feedback_message = "Zeit ist abgelaufen! Starte neu für die nächste Runde."
                self.feedback_level = "warn"
                self.feedback_expires = now + 6.0
                return

            if self.awaiting_clear_frame:
                if detections_list:
                    recognized_parts = {
                        self.map_label_to_part_id(str(det.get("raw_label", "")))
                        for det in detections_list
                    }
                    recognized_parts.discard(None)
                    if len(recognized_parts) > 1:
                        message = "Mehrere Bauteile erkannt. Bitte nur ein Bauteil im Demonstrator lassen."
                        self.feedback_message = message
                        self.feedback_level = "warn"
                        self.feedback_expires = now + 3.0
                        self.last_instruction_message = message
                        self.highlight_row = None
                        self.highlight_column = None
                        self.highlight_area = None
                        self._awaiting_expected = None
                        self.multiple_detected = True
                        self._awaiting_check_counter = 0
                        return
                    if self.multiple_detected:
                        self.multiple_detected = False
                    self._awaiting_check_counter += 1
                    if self._awaiting_check_counter >= 10:
                        self._awaiting_check_counter = 0
                        self._validate_awaiting_detection_locked(detections_list, now)
                    return
                self.multiple_detected = False
                self._finish_placement_pause_locked(now)
                return

            if not detections_list:
                return

            best_detection = self._select_best_detection(detections_list)

            if best_detection is None:
                return

            raw_label, confidence, part_id, display_name, length_mm = best_detection
            self.last_detected = {
                "id": part_id,
                "display": display_name,
                "confidence": confidence,
                "length_mm": length_mm,
            }

            self.attempts += 1

            if part_id is None:
                self.missed_count += 1
                self.feedback_message = f"Unbekanntes Teil erkannt ({raw_label})."
                self.feedback_level = "warn"
                self.feedback_expires = now + 3.0
                self.highlight_row = None
                self.highlight_column = None
                self.highlight_area = None
                self.multiple_detected = False
                return

            if part_id == "gerade":
                if length_mm is None:
                    self.missed_count += 1
                    self.feedback_message = (
                        "Gerade erkannt, aber Länge fehlt. Bitte Marker sichtbar halten."
                    )
                    self.feedback_level = "warn"
                    self.feedback_expires = now + 3.0
                    self.highlight_row = None
                    self.highlight_column = None
                    self.highlight_area = None
                    self.multiple_detected = False
                    return
                diffs = [abs(length_mm - bin_mm) for bin_mm in self.gerade_bins_mm]
                min_idx = int(min(range(len(diffs)), key=diffs.__getitem__))
                if diffs[min_idx] > self.gerade_tolerance_mm:
                    self.missed_count += 1
                    self.feedback_message = (
                        f"Gerade mit {length_mm:.0f} mm passt in kein Lagerfach. Bitte lege das Teil zurück in die Schublade."
                    )
                    self.feedback_level = "warn"
                    self.feedback_expires = now + 3.0
                    self.highlight_row = None
                    self.highlight_column = None
                    self.highlight_area = None
                    self.multiple_detected = False
                    return
                row, column = 1, min_idx + 1
                self.storage_counts[part_id] = self.storage_counts.get(part_id, 0) + 1
                self.sorted_count += 1
                elapsed_since_last = now - self.last_success_time if self.last_success_time else 0.0
                bonus_points = max(0, int(self.round_duration - elapsed_since_last))
                points_awarded = 100 + bonus_points
                expected_info = self._apply_instruction_from_detection_locked(
                    part_id,
                    display_name,
                    length_mm,
                    now,
                    feedback_level="success",
                    set_expected=False,
                )
                if expected_info is None:
                    expected_info = {
                        "part_id": part_id,
                        "area": "grid",
                        "row": row,
                        "column": column,
                        "length_mm": self.gerade_bins_mm[min_idx],
                    }
                self._begin_placement_pause_locked(now, expected_info)
                self.score += points_awarded
                instruction_text = self.last_instruction_message
                self.feedback_message = f"{instruction_text} (+{points_awarded} Punkte)"
                self.last_success_time = now
                return

            if part_id in {"noppenstein", "noppenscheiben", "schraube"}:
                self.sorted_count += 1
                elapsed_since_last = now - self.last_success_time if self.last_success_time else 0.0
                bonus_points = max(0, int(self.round_duration - elapsed_since_last))
                points_awarded = 100 + bonus_points
                expected_info = self._apply_instruction_from_detection_locked(
                    part_id,
                    display_name,
                    length_mm,
                    now,
                    feedback_level="success",
                    set_expected=False,
                )
                if expected_info is None:
                    expected_info = {"part_id": part_id, "area": "box"}
                self._begin_placement_pause_locked(now, expected_info)
                self.score += points_awarded
                instruction_text = self.last_instruction_message
                self.feedback_message = f"{instruction_text} (+{points_awarded} Punkte)"
                self.last_success_time = now
                return

            area_type, row, column, message = self._assign_storage_slot(part_id, length_mm)
            if area_type == "UNBEKANNT":
                self.missed_count += 1
                self.feedback_message = f"{message} Bitte lege das Teil zurück in die Schublade."
                self.feedback_level = "warn"
                self.feedback_expires = now + 3.0
                self.highlight_row = None
                self.highlight_column = None
                self.highlight_area = None
                self.multiple_detected = False
                return

            self.storage_counts[part_id] = self.storage_counts.get(part_id, 0) + 1
            self.sorted_count += 1
            elapsed_since_last = now - self.last_success_time if self.last_success_time else 0.0
            bonus_points = max(0, int(self.round_duration - elapsed_since_last))
            points_awarded = 100 + bonus_points
            expected_info = self._apply_instruction_from_detection_locked(
                part_id,
                display_name,
                length_mm,
                now,
                feedback_level="success",
                set_expected=False,
            )
            if expected_info is None:
                expected_info = {
                    "part_id": part_id,
                    "area": str(area_type).lower(),
                    "row": row,
                    "column": column,
                }
            self._begin_placement_pause_locked(now, expected_info)
            self.score += points_awarded
            instruction_text = self.last_instruction_message
            self.feedback_message = f"{instruction_text} (+{points_awarded} Punkte)"
            self.last_success_time = now
            return

    def get_state(self) -> Dict[str, object]:
        now = time.time()
        with self._lock:
            self._update_instruction_state_locked(now)
            time_remaining = self._time_remaining_locked(now)
            state = {
                "mode": self.mode_id,
                "mode_name": self.mode_name,
                "game_enabled": self.game_enabled,
                "round_active": self.round_active,
                "score": self.score,
                "rounds": self.sorted_count,
                "attempts": self.attempts,
                "streak": self.sorted_count,
                "best_streak": self.sorted_count,
                "round_time": self.round_duration - time_remaining,
                "time_remaining": time_remaining,
                "sorted_count": self.sorted_count,
                "missed_count": self.missed_count,
                "feedback": self.feedback_message,
                "feedback_level": self.feedback_level,
                "last_detected": self.last_detected,
                "instruction": self._instruction_text if now <= self._instruction_until else "",
                "storage_instruction": self.last_instruction_message,
                "storage_row": self.highlight_row,
                "storage_column": self.highlight_column,
                "storage_area": self.highlight_area,
                "awaiting_clear_frame": self.awaiting_clear_frame,
                "multiple_detected": self.multiple_detected,
            }
            return state


class GameManager:
    def __init__(self, games: Dict[str, object], default_mode: Optional[str] = None) -> None:
        self._games = games
        self._active_mode = default_mode if default_mode in games else None

    def get_active_game(self) -> Optional[object]:
        if self._active_mode is None:
            return None
        return self._games[self._active_mode]

    def get_available_modes(self) -> List[Dict[str, object]]:
        modes = []
        for mode_id, game in self._games.items():
            name = getattr(game, "mode_name", mode_id.title())
            modes.append({"id": mode_id, "name": name, "active": mode_id == self._active_mode})
        return modes

    def set_active_mode(self, mode_id: str) -> Dict[str, object]:
        if mode_id not in self._games:
            raise KeyError(mode_id)
        if mode_id != self._active_mode and self._active_mode is not None:
            previous = self._games[self._active_mode]
            if hasattr(previous, "reset"):
                previous.reset()
        self._active_mode = mode_id
        current = self._games[self._active_mode]
        if hasattr(current, "reset"):
            current.reset()
        return self.get_state()

    def start_active_game(self) -> Dict[str, object]:
        game = self.get_active_game()
        if game is None:
            raise RuntimeError("Kein Spielmodus ausgewählt")
        if hasattr(game, "start_game"):
            game.start_game()
        return self.get_state()

    def next_round(self) -> Dict[str, object]:
        game = self.get_active_game()
        if game is None:
            raise RuntimeError("Kein Spielmodus ausgewählt")
        if hasattr(game, "force_new_round"):
            game.force_new_round()
        return self.get_state()

    def reset_active_game(self) -> Dict[str, object]:
        game = self.get_active_game()
        if game is None:
            raise RuntimeError("Kein Spielmodus ausgewählt")
        if hasattr(game, "reset"):
            game.reset()
        return self.get_state()

    def get_state(self) -> Dict[str, object]:
        game = self.get_active_game()
        if game is None:
            return {
                "mode": None,
                "mode_name": "",
                "game_enabled": False,
                "round_active": False,
                "score": 0,
                "rounds": 0,
                "attempts": 0,
                "streak": 0,
                "best_streak": 0,
                "round_time": 0.0,
                "time_remaining": 0.0,
                "sorted_count": 0,
                "missed_count": 0,
                "feedback": "Bitte Spielmodus auswählen.",
                "feedback_level": "info",
                "last_detected": None,
                "instruction": "",
                "storage_instruction": "",
                "leaderboard": [],
                "available_modes": self.get_available_modes(),
            }

        state = game.get_state()
        state.setdefault("mode", getattr(game, "mode_id", self._active_mode))
        state.setdefault("mode_name", getattr(game, "mode_name", state["mode"]))
        state["available_modes"] = self.get_available_modes()
        return state


missions_game = SortingGame(PART_CATALOG)
inventory_game = InventorySortGame(
    PART_CATALOG,
    duration_seconds=INVENTORY_SORT_DURATION_S,
    gerade_bins_mm=INVENTORY_GERADE_BINS_MM,
    gerade_tolerance_mm=INVENTORY_GERADE_BIN_TOLERANCE_MM,
)

game_manager = GameManager(
    {
        "missions": missions_game,
        "inventory": inventory_game,
    },
    default_mode=None,
)



# ============================================================
# DETECTION CAMERA WITH GAME OVERLAY
# ============================================================
class GameDetectCamera(FrameGrabber):
    """Detect camera that feeds detections into the SortingGame and draws feedback."""

    def __init__(
        self,
        base_cam: FrameGrabber,
        engine_path: str,
        game_manager: GameManager,
        model_input_size: int = YOLO_MODEL_INPUT_SIZE,
        conf_thresh: float = YOLO_CONF_THRESH,
        iou_thresh: float = YOLO_IOU_THRESH,
        max_det: int = YOLO_MAX_DET,
        device: str = YOLO_DEVICE,
        skip_frames: int = SKIP_FRAMES,
        batch_size: int = BATCH_SIZE,
        left_segment: Optional[SegmentCamera] = None,
        right_segment: Optional[SegmentCamera] = None,
    ) -> None:
        super().__init__()
        self.base_cam = base_cam
        self.game_manager = game_manager
        self.model_input_size = model_input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det
        self.device = device
        self.skip_frames = skip_frames
        self.batch_size = batch_size
        self.left_segment = left_segment
        self.right_segment = right_segment

        self.model = YOLO(engine_path, task="detect")
        self.model.overrides["imgsz"] = (model_input_size, model_input_size)
        self.model.overrides["conf"] = conf_thresh
        self.model.overrides["iou"] = iou_thresh
        self.model.overrides["max_det"] = max_det

        self._lock = threading.Lock()
        self._latest_annotated: Optional[np.ndarray] = None
        self._is_running = False
        self._frame_counter = 0
        self.last_known_ratio: Optional[float] = None
        self._fps_start_time = time.time()
        self._fps_frame_count = 0
        self._fps: float = 0.0
        self._roi_rel: Optional[Tuple[float, float, float, float]] = None
        self._roi_pixels: Optional[Tuple[int, int, int, int]] = None
        self._apply_roi_locked(CENTER_CAMERA_ROI_REL)

    def _classify_side(self, labels: List[str]) -> Optional[str]:
        if not labels:
            return None
        has_up = any(label == "nubsup" for label in labels)
        has_down = any(label == "nubsdown" for label in labels)
        if has_up and not has_down:
            return "nubsup"
        if has_down and not has_up:
            return "nubsdown"
        if has_up and has_down:
            return "mixed"
        return None

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

    def reset_calibration(self) -> None:
        with self._lock:
            self.last_known_ratio = None
            self._frame_counter = 0
        active_game = self.game_manager.get_active_game()
        if active_game and hasattr(active_game, "show_instruction"):
            active_game.show_instruction(
                "Kalibrierung gestartet – halte den ArUco Marker mittig in die Kamera.",
                duration=5.0,
                pause_timer=True,
            )

    def _draw_game_panel(self, frame: np.ndarray, state: Dict[str, object]) -> None:
        lines: List[Tuple[str, Tuple[int, int, int]]] = []
        mode = state.get("mode")

        if mode == "inventory":
            time_remaining = state.get("time_remaining")
            if state.get("round_active"):
                lines.append((f"Zeit verbleibend: {time_remaining:.1f}s" if isinstance(time_remaining, (int, float)) else "Sortierung aktiv", (255, 255, 255)))
            else:
                lines.append((state.get("feedback", "Sortierung bereit."), (0, 255, 255)))
            lines.append((f"Punkte: {state.get('score', 0)} | Sortiert: {state.get('sorted_count', 0)} / Verpasst: {state.get('missed_count', 0)}", (255, 255, 255)))
            instruction = state.get("storage_instruction")
            if instruction:
                lines.append((instruction, (0, 255, 255)))
            last_det = state.get("last_detected")
            if last_det and last_det.get("display"):
                det_line = f"Letztes Teil: {last_det['display']} ({last_det['confidence'] * 100:.0f}%)"
                lines.append((det_line, (180, 180, 255)))
        else:
            if state.get("round_active") and state.get("target"):
                target = state["target"]
                lines.append((f"Bauteil: {target['display']}", (0, 255, 255)))
                lines.append((f"Zeit: {state['round_time']:.1f}s", (255, 255, 255)))
            elif state.get("target"):
                cooldown = state.get("cooldown_remaining", 0.0)
                lines.append(
                    (f"Bauteile gefunden! Neue Runde in {cooldown:.1f}s", (0, 255, 0))
                )
            else:
                lines.append(("Spiel lädt...", (0, 255, 255)))

            score_line = f"Punkte: {state['score']} | Serie: {state['streak']} (Best: {state['best_streak']})"
            lines.append((score_line, (255, 255, 255)))

            last_det = state.get("last_detected")
            if last_det:
                det_line = f"Letztes Teil: {last_det['display']} ({last_det['confidence'] * 100:.0f}%)"
                lines.append((det_line, (180, 180, 255)))

            feedback = state.get("feedback")
            if feedback:
                level = state.get("feedback_level", "info")
                color = (255, 255, 255)
                if level == "success":
                    color = (0, 220, 0)
                elif level == "warn":
                    color = (0, 165, 255)
                lines.append((feedback, color))

        if not lines:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        line_height = 18
        padding = 8

        max_width = 0
        for text, _color in lines:
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            max_width = max(max_width, text_size[0])

        rect_width = max_width + padding * 2
        rect_height = line_height * len(lines) + padding * 2
        top_left = (8, 8)
        bottom_right = (top_left[0] + rect_width, top_left[1] + rect_height)
        cv2.rectangle(frame, top_left, bottom_right, (20, 20, 20), thickness=-1)

        y = top_left[1] + padding + 12
        for text, color in lines:
            cv2.putText(
                frame,
                text,
                (top_left[0] + padding, y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            y += line_height

    def _inference_loop(self) -> None:
        while self._is_running:
            raw_full = self.base_cam.get_latest_frame()
            if raw_full is None:
                time.sleep(0.001)
                continue
            game = self.game_manager.get_active_game()
            if game is not None and hasattr(game, "ensure_round_active"):
                game.ensure_round_active()

            self._frame_counter += 1

            frame_320 = cv2.resize(
                raw_full,
                (self.model_input_size, self.model_input_size),
                interpolation=cv2.INTER_LINEAR,
            )
            img_rgb = cv2.cvtColor(frame_320, cv2.COLOR_BGR2RGB)

            if ARUCO_ENABLED and (self._frame_counter % ARUCO_SKIP_FRAMES) == 0:
                ratio, _ = detect_aruco_markers_with_tracking(frame_320)
                if ratio is not None:
                    self.last_known_ratio = ratio

            if (self._frame_counter % self.skip_frames) != 0:
                time.sleep(0.001)
                continue

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
            final_boxes = results[0].boxes.xyxy.cpu().numpy()
            final_scores = results[0].boxes.conf.cpu().numpy()
            final_classes = results[0].boxes.cls.cpu().numpy().astype(int)

            annotated_full = raw_full.copy()
            frame_h, frame_w = annotated_full.shape[:2]
            scale_x = frame_w / float(self.model_input_size)
            scale_y = frame_h / float(self.model_input_size)
            detections_for_game: List[Dict[str, object]] = []
            target_id = game.peek_target_id() if game and hasattr(game, "peek_target_id") else None
            roi_pixels = self._roi_pixels

            for (x1, y1, x2, y2), conf, cls_id in zip(
                final_boxes.astype(int), final_scores, final_classes
            ):
                try:
                    class_name = self.model.names[cls_id]
                except (KeyError, IndexError):
                    class_name = f"class{cls_id}"
                normalised_class = _normalise_label(str(class_name))
                is_nubs_helper = normalised_class in {"nubsup", "nubsdown"}

                if roi_pixels:
                    rx1, ry1, rx2, ry2 = roi_pixels
                    cx = 0.5 * (x1 + x2)
                    cy = 0.5 * (y1 + y2)
                    if cx < rx1 or cx > rx2 or cy < ry1 or cy > ry2:
                        continue

                part_id = (
                    game.map_label_to_part_id(class_name)
                    if game and hasattr(game, "map_label_to_part_id")
                    else None
                )
                if game is not None and part_id is None:
                    continue

                is_target = (
                    game is not None and target_id is not None and part_id == target_id
                )
                if is_nubs_helper:
                    draw_color = (255, 180, 0)
                else:
                    draw_color = (0, 220, 0) if is_target else (0, 165, 255)
                draw_thickness = 2
                draw_x1, draw_y1, draw_x2, draw_y2 = x1, y1, x2, y2
                if part_id == "versetzte_gerade":
                    horiz_pad = max(4, int(0.06 * (x2 - x1)))
                    vert_pad = max(2, int(0.04 * (y2 - y1)))
                    draw_x1 = max(0, x1 - horiz_pad)
                    draw_x2 = min(self.model_input_size - 1, x2 + horiz_pad)
                    draw_y1 = max(0, y1 - vert_pad)
                    draw_y2 = min(self.model_input_size - 1, y2 + vert_pad)
                    draw_thickness = 3
                    draw_color = (64, 224, 208) if is_target else (180, 105, 255)

                if (not is_nubs_helper) and self.last_known_ratio is not None:
                    w_mm, h_mm = calculate_dimensions([x1, y1, x2, y2], self.last_known_ratio)
                else:
                    w_mm, h_mm = (None, None)

                length_mm = None
                dims = [d for d in (w_mm, h_mm) if d is not None]
                if dims:
                    length_mm = max(dims)

                if game is not None and part_id is not None:
                    detections_for_game.append(
                        {
                            "raw_label": class_name,
                            "confidence": float(conf),
                            "length_mm": length_mm,
                        }
                    )

                display_name = (
                    str(PART_LOOKUP.get(part_id, {}).get("display_name", class_name))
                    if part_id is not None
                    else class_name
                )

                label = f"{display_name} {conf:.2f}"
                if w_mm is not None and h_mm is not None:
                    label += f" ({w_mm:.1f}x{h_mm:.1f}mm)"
                elif length_mm is not None:
                    label += f" (L≈{length_mm:.1f}mm)"

                draw_x1_px = int(round(draw_x1 * scale_x))
                draw_y1_px = int(round(draw_y1 * scale_y))
                draw_x2_px = int(round(draw_x2 * scale_x))
                draw_y2_px = int(round(draw_y2 * scale_y))
                cv2.rectangle(
                    annotated_full,
                    (draw_x1_px, draw_y1_px),
                    (draw_x2_px, draw_y2_px),
                    draw_color,
                    draw_thickness,
                )
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_width = t_size[0] + 6
                label_height = t_size[1] + 6
                label_x1 = max(0, draw_x1_px)
                label_x2 = min(frame_w - 1, label_x1 + label_width)
                if is_nubs_helper:
                    label_y1 = min(frame_h - label_height - 1, draw_y2_px + 4)
                    label_y1 = max(0, label_y1)
                    label_y2 = min(frame_h - 1, label_y1 + label_height)
                else:
                    label_y2 = draw_y1_px
                    label_y1 = draw_y1_px - label_height
                    if label_y1 < 0:
                        label_y1 = min(frame_h - label_height - 1, draw_y2_px + 4)
                        label_y2 = min(frame_h - 1, label_y1 + label_height)
                label_baseline = max(2, min(frame_h - 2, label_y2 - 4))
                cv2.rectangle(
                    annotated_full,
                    (label_x1, max(0, label_y1)),
                    (label_x2, label_y2),
                    draw_color,
                    -1,
                )
                cv2.putText(
                    annotated_full,
                    label,
                    (label_x1 + 3, label_baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    thickness=1,
                )

            if (
                detections_for_game
                and self.left_segment is not None
                and self.right_segment is not None
            ):
                left_labels = self.left_segment.get_latest_labels()
                right_labels = self.right_segment.get_latest_labels()
                left_orientation = self._classify_side(left_labels)
                right_orientation = self._classify_side(right_labels)
                orientation_consensus: Optional[str] = None
                if (
                    left_orientation in {"nubsup", "nubsdown"}
                    and right_orientation in {"nubsup", "nubsdown"}
                ):
                    orientation_consensus = (
                        "same" if left_orientation == right_orientation else "opposite"
                    )
                for det in detections_for_game:
                    det["left_orientation"] = left_orientation
                    det["right_orientation"] = right_orientation
                    det["orientation_consensus"] = orientation_consensus
                    raw_label = det.get("raw_label", "")
                    normalised = _normalise_label(str(raw_label))
                    if orientation_consensus == "opposite" and normalised in {"diagonale", "versetzte_gerade"}:
                        det["raw_label"] = "versetzte_gerade"
                    elif orientation_consensus == "same" and normalised in {"diagonale", "versetzte_gerade"}:
                        det["raw_label"] = "diagonale"

            if game is not None and hasattr(game, "update_with_detections"):
                game.update_with_detections(detections_for_game)

            display_frame = annotated_full
            if roi_pixels:
                rx1, ry1, rx2, ry2 = roi_pixels
                crop_x1 = max(0, min(frame_w - 1, int(round(rx1 * scale_x))))
                crop_y1 = max(0, min(frame_h - 1, int(round(ry1 * scale_y))))
                crop_x2 = max(1, min(frame_w, int(round(rx2 * scale_x))))
                crop_y2 = max(1, min(frame_h, int(round(ry2 * scale_y))))
                if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                    display_frame = annotated_full[crop_y1:crop_y2, crop_x1:crop_x2].copy()

            fps_text = f"FPS: {self._fps:.1f}"
            display_h = display_frame.shape[0]
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

            with self._lock:
                self._latest_annotated = display_frame

            # FPS based on completed inference passes, not raw capture loop.
            self._fps_frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self._fps_start_time
            if self._fps_frame_count >= 30 or elapsed_time >= 2.0:
                if elapsed_time > 0:
                    self._fps = self._fps_frame_count / elapsed_time
                self._fps_start_time = current_time
                self._fps_frame_count = 0

            time.sleep(0.001)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_annotated is None:
                return None
            return self._latest_annotated.copy()

    def clear_latest_frame(self) -> None:
        with self._lock:
            self._latest_annotated = None

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


# ============================================================
# CAMERA INSTANTIATION
# ============================================================
oak_serials = list_oak_devices()
left_oak_serial, right_oak_serial = resolve_side_camera_serials(
    oak_serials,
    fallback_left_serial=LEFT_CAMERA_SERIAL,
    fallback_right_serial=RIGHT_CAMERA_SERIAL,
)
print(f"[INFO] Game camera mapping: left={left_oak_serial}, right={right_oak_serial}")

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

oak_left_raw = OAK1MaxCamera(
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
oak_left_seg = SegmentCamera(
    base_cam=oak_left_raw,
    engine_path=left_engine_path,
    model_input_size=YOLO_MODEL_INPUT_SIZE,
    conf_thresh=YOLO_CONF_THRESH,
    iou_thresh=YOLO_IOU_THRESH,
    max_det=YOLO_MAX_DET,
    device=YOLO_DEVICE,
    skip_frames=SKIP_FRAMES,
    batch_size=BATCH_SIZE,
    rotation="left_90",
)

oak_right_raw = OAK1MaxCamera(
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
oak_right_seg = SegmentCamera(
    base_cam=oak_right_raw,
    engine_path=right_engine_path,
    model_input_size=YOLO_MODEL_INPUT_SIZE,
    conf_thresh=YOLO_CONF_THRESH,
    iou_thresh=YOLO_IOU_THRESH,
    max_det=YOLO_MAX_DET,
    device=YOLO_DEVICE,
    skip_frames=SKIP_FRAMES,
    batch_size=BATCH_SIZE,
    rotation="right_90",
)

usb_center_raw = USBWebcamCamera(
    device_index=USB_DEVICE_INDEX,
    width=USB_CAPTURE_WIDTH,
    height=USB_CAPTURE_HEIGHT,
    fps=USB_CAPTURE_FPS,
)
usb_center_game = GameDetectCamera(
    base_cam=usb_center_raw,
    engine_path=center_engine_path,
    game_manager=game_manager,
    model_input_size=YOLO_MODEL_INPUT_SIZE,
    conf_thresh=YOLO_CONF_THRESH,
    iou_thresh=YOLO_IOU_THRESH,
    max_det=YOLO_MAX_DET,
    device=YOLO_DEVICE,
    skip_frames=SKIP_FRAMES,
    batch_size=BATCH_SIZE,
    left_segment=oak_left_seg,
    right_segment=oak_right_seg,
)

_CAMERA_RESTART_LOCK = threading.Lock()


def restart_all_cameras() -> None:
    """Stop and restart all camera pipelines (used when a device hangs)."""
    global CURRENT_CENTER_CAMERA_ROI_REL
    with _CAMERA_RESTART_LOCK:
        usb_center_game.stop()
        usb_center_raw.stop()
        oak_left_seg.stop()
        oak_left_raw.stop()
        oak_right_seg.stop()
        oak_right_raw.stop()

        for cam in (
            usb_center_game,
            usb_center_raw,
            oak_left_seg,
            oak_left_raw,
            oak_right_seg,
            oak_right_raw,
        ):
            if hasattr(cam, "clear_latest_frame"):
                cam.clear_latest_frame()

        time.sleep(0.4)

        oak_left_raw.start()
        oak_left_seg.start()
        oak_right_raw.start()
        oak_right_seg.start()
        usb_center_raw.start()
        usb_center_game.start()

        time.sleep(0.2)
        CURRENT_CENTER_CAMERA_ROI_REL = usb_center_game.get_roi()

oak_left_raw.start()
oak_left_seg.start()
oak_right_raw.start()
oak_right_seg.start()
usb_center_raw.start()
usb_center_game.start()

CURRENT_CENTER_CAMERA_ROI_REL = usb_center_game.get_roi()

# ============================================================
# AUTHENTICATION HELPERS
# ============================================================
def _register_session() -> str:
    token = secrets.token_urlsafe(32)
    ADMIN_SESSIONS[token] = time.time()
    return token


def _remove_session(token: str) -> None:
    ADMIN_SESSIONS.pop(token, None)


def _get_session_token(request: Request) -> Optional[str]:
    return request.cookies.get(SESSION_COOKIE_NAME)


def is_admin_request(request: Request) -> bool:
    token = _get_session_token(request)
    if not token:
        return False
    created = ADMIN_SESSIONS.get(token)
    if created is None:
        return False
    if SESSION_TTL_SECONDS and (time.time() - created) > SESSION_TTL_SECONDS:
        _remove_session(token)
        return False
    ADMIN_SESSIONS[token] = time.time()
    return True


def ensure_admin(request: Request) -> None:
    if not is_admin_request(request):
        raise HTTPException(status_code=401, detail="Admin-Rechte erforderlich.")


# ============================================================
# ROUTES
# ============================================================
@app.get("/", response_class=Response)
def index(request: Request) -> Response:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_left")
def video_left() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_stream_generator(oak_left_seg),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.get("/video_center")
def video_center() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_stream_generator(usb_center_game),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.get("/video_right")
def video_right() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_stream_generator(oak_right_seg),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


class ModeSelection(BaseModel):
    mode: str


class ROIUpdate(BaseModel):
    roi: Optional[List[float]]


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/auth/login")
def auth_login(request: Request, credentials: LoginRequest) -> JSONResponse:
    username = credentials.username.strip()
    password = credentials.password
    if username != ADMIN_USERNAME or password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Ungültige Anmeldedaten.")
    token = _register_session()
    response = JSONResponse({"success": True, "username": ADMIN_USERNAME})
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=int(SESSION_TTL_SECONDS),
        httponly=True,
        path="/",
        samesite="lax",
    )
    return response


@app.post("/auth/logout")
def auth_logout(request: Request) -> JSONResponse:
    token = _get_session_token(request)
    if token:
        _remove_session(token)
    response = JSONResponse({"success": True})
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return response


@app.get("/auth/status")
def auth_status(request: Request) -> Dict[str, object]:
    is_admin = is_admin_request(request)
    return {
        "is_admin": is_admin,
        "username": ADMIN_USERNAME if is_admin else "",
    }


@app.get("/game/state")
def get_game_state(request: Request) -> Dict[str, object]:
    state = dict(game_manager.get_state())
    is_admin = is_admin_request(request)
    state["is_admin"] = is_admin
    state["admin_username"] = ADMIN_USERNAME if is_admin else ""
    return state


@app.post("/game/next")
def force_next_round(request: Request) -> Dict[str, object]:
    try:
        game_manager.next_round()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return get_game_state(request)


@app.post("/game/start")
def start_game(request: Request) -> Dict[str, object]:
    try:
        game_manager.start_active_game()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return get_game_state(request)


@app.post("/game/reset")
def reset_game(request: Request) -> Dict[str, object]:
    try:
        game_manager.reset_active_game()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return get_game_state(request)


@app.post("/game/select")
def select_game(request: Request, selection: ModeSelection) -> Dict[str, object]:
    try:
        game_manager.set_active_mode(selection.mode)
    except KeyError as exc:  # noqa: B902
        raise HTTPException(status_code=400, detail=f"Unbekannter Spielmodus: {selection.mode}") from exc
    return get_game_state(request)


@app.get("/roi", response_class=Response)
def roi_calibration_page(request: Request) -> Response:
    return templates.TemplateResponse(
        "roi_calibration.html",
        {
            "request": request,
            "is_admin": is_admin_request(request),
            "auth_required": True,
        },
    )


@app.get("/roi/config")
def roi_config(request: Request) -> Dict[str, object]:
    ensure_admin(request)
    roi = usb_center_game.get_roi()
    return {
        "roi": list(roi) if roi else None,
        "image_size": [YOLO_MODEL_INPUT_SIZE, YOLO_MODEL_INPUT_SIZE],
    }


@app.post("/roi/config")
def roi_update(request: Request, config: ROIUpdate) -> Dict[str, object]:
    ensure_admin(request)
    global CENTER_CAMERA_ROI_REL, CURRENT_CENTER_CAMERA_ROI_REL
    roi_list = config.roi
    roi_tuple: Optional[Tuple[float, float, float, float]] = None
    if roi_list is not None:
        if len(roi_list) != 4:
            raise HTTPException(status_code=400, detail="ROI benötigt genau vier Werte (x1, y1, x2, y2).")
        try:
            roi_tuple = normalise_roi_tuple(
                (
                    float(roi_list[0]),
                    float(roi_list[1]),
                    float(roi_list[2]),
                    float(roi_list[3]),
                )
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="ROI-Werte müssen Zahlen sein.") from exc
        if roi_tuple is None:
            raise HTTPException(status_code=400, detail="Ungültige ROI-Koordinaten.")
    usb_center_game.set_roi(roi_tuple)
    CENTER_CAMERA_ROI_REL = roi_tuple
    CURRENT_CENTER_CAMERA_ROI_REL = usb_center_game.get_roi()
    persist_center_roi(CURRENT_CENTER_CAMERA_ROI_REL)
    return {
        "roi": list(CURRENT_CENTER_CAMERA_ROI_REL) if CURRENT_CENTER_CAMERA_ROI_REL else None,
        "image_size": [YOLO_MODEL_INPUT_SIZE, YOLO_MODEL_INPUT_SIZE],
    }


@app.post("/roi/reset")
def roi_reset(request: Request) -> Dict[str, object]:
    ensure_admin(request)
    global CENTER_CAMERA_ROI_REL, CURRENT_CENTER_CAMERA_ROI_REL
    usb_center_game.set_roi(None)
    CENTER_CAMERA_ROI_REL = None
    CURRENT_CENTER_CAMERA_ROI_REL = None
    persist_center_roi(None)
    return {
        "roi": None,
        "image_size": [YOLO_MODEL_INPUT_SIZE, YOLO_MODEL_INPUT_SIZE],
    }


@app.post("/cameras/restart")
def cameras_restart(request: Request) -> Dict[str, object]:
    ensure_admin(request)
    try:
        restart_all_cameras()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Kameras konnten nicht neu gestartet werden: {exc}",
        ) from exc
    state = get_game_state(request)
    state["camera_status"] = "restarted"
    state["camera_status_timestamp"] = time.time()
    return state


@app.post("/calibration/reset")
def calibration_reset(request: Request) -> Dict[str, object]:
    ensure_admin(request)
    usb_center_game.reset_calibration()
    return get_game_state(request)


@app.on_event("shutdown")
def cleanup_cameras() -> None:
    usb_center_game.stop()
    usb_center_raw.stop()
    oak_left_seg.stop()
    oak_left_raw.stop()
    oak_right_seg.stop()
    oak_right_raw.stop()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
