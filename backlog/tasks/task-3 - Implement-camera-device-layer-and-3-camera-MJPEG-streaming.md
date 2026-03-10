---
id: TASK-3
title: Implement camera device layer and 3-camera MJPEG streaming
status: To Do
assignee: []
created_date: '2026-03-10 21:31'
labels:
  - backend
  - streaming
  - devices
milestone: m-0
dependencies:
  - TASK-1
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build the device abstraction layer and the minimal pipeline that captures frames from all three cameras and streams them as MJPEG to the web UI. This is the core of the system — everything else builds on top.

**Device layer (`src/demonstrator/devices/`):**
- `base.py`: `FrameSource` ABC with `start()`, `get_latest() -> Frame | None`, `stop()`
- `opencv.py`: USB webcam via `cv2.VideoCapture`. Uses a background thread; `get_latest()` returns the most recent frame without blocking. Never queues stale frames.
- `oak.py`: OAK-1 Max via `depthai`. Same contract as above. Handles reconnect gracefully.
- `mock.py`: Reads a static image or loops a video file. Useful for development without physical hardware.

**Minimal pipeline (`src/demonstrator/pipeline/`):**
- `capture.py`: Calls `device.get_latest()` at `render_fps` rate, skips if `None`
- `render.py`: Stamps a timestamp and camera label on the frame, encodes to JPEG once per cycle
- `stream.py`: `SharedJPEGBuffer` — a thread-safe store holding the latest JPEG bytes per camera. All MJPEG clients read from the same buffer; encoding happens exactly once per frame regardless of client count.

**MJPEG API endpoints (`src/demonstrator/api/stream.py`):**
- `GET /stream/{camera_id}` — standard multipart MJPEG response. Reads from `SharedJPEGBuffer`.
- Camera IDs: `left`, `center`, `right`

**UI integration:**
- Add a camera view page (extends the base template from task-2) with three `<img>` tags pointing to the MJPEG endpoints
- Layout: center camera large, left and right cameras smaller alongside
- Show a placeholder/static image when a camera stream is unavailable

**Profile-driven startup:**
- The `dev_laptop_1cam.yaml` profile enables only `center` camera (OpenCV), left and right use `MockCamera` with a placeholder image
- The `demo_jetson_3cam.yaml` profile enables all three: left/right as OAK, center as OpenCV

Depends on: task-1 (package skeleton and profile system).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All three MJPEG streams are accessible at `/stream/left`, `/stream/center`, `/stream/right`
- [ ] #2 Opening `/stream/center` in a browser shows a live or mock video feed
- [ ] #3 Three simultaneous clients on the same stream do not cause additional JPEG encode work (single encode per cycle verified via logging)
- [ ] #4 Stopping and restarting the app with `dev_laptop_1cam.yaml` starts correctly with one real camera and two mock streams
- [ ] #5 If an OAK camera disconnects, the stream falls back to a placeholder image without crashing the app
- [ ] #6 `get_latest()` never returns a frame older than 2 render cycles (no stale frame queuing)
- [ ] #7 The camera view page in the UI shows all three streams in the correct layout
<!-- AC:END -->
