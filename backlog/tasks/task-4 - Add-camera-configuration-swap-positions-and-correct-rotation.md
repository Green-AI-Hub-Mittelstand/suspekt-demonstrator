---
id: TASK-4
title: 'Add camera configuration: swap positions and correct rotation'
status: To Do
assignee: []
created_date: '2026-03-10 21:32'
labels:
  - backend
  - frontend
  - config
milestone: m-0
dependencies:
  - TASK-3
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow operators to configure camera positions and orientation directly from the web UI, with changes persisted to the active profile. This solves the common physical setup issue where left/right OAK cameras are plugged in on the wrong side or mounted at a non-standard angle.

**Features:**
1. **Swap cameras**: a button to swap the `left` ↔ `right` logical assignment without physically touching the cables. Implemented as swapping the `device_index` (or USB path) in the running config.
2. **Per-camera rotation**: a control to set rotation to 0°, 90°, 180°, or 270° per camera. Applied in the render stage before encoding.
3. **Live preview**: changes take effect immediately in the MJPEG stream — no app restart required.
4. **Persist to profile**: a "Save" button writes the current camera config back to the active YAML profile file so it survives restarts.
5. **Reset to defaults**: a "Reset" button reverts to the last saved profile state.

**UI:**
- Camera config panel accessible from the camera view page (e.g. a settings icon → slide-out panel or modal)
- Each camera slot shows its stream thumbnail, current rotation, and swap/rotate controls
- Uses HTMX for all interactions (`hx-post` to config endpoints, `hx-swap="none"` — stream updates itself)

**API endpoints (`src/demonstrator/api/camera_config.py`):**
- `POST /api/cameras/swap` — swaps left ↔ right logical assignment
- `POST /api/cameras/{camera_id}/rotation` — body: `{"degrees": 90}`
- `POST /api/cameras/save` — writes current config to the active profile YAML
- `POST /api/cameras/reset` — reloads config from the active profile YAML

**Rotation implementation:**
- Applied in `render.py` using `cv2.rotate()` before JPEG encoding
- Rotation value is read from the camera's runtime config, not hardcoded

Depends on: task-3 (camera streaming must be working; runtime config object must exist).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Clicking swap in the UI immediately exchanges the left and right camera streams in the browser
- [ ] #2 Changing rotation on a camera updates the stream within 1 second, no restart required
- [ ] #3 Saving config writes the updated rotation and swap state to the active YAML profile file
- [ ] #4 After an app restart, the saved rotation and swap config is restored correctly
- [ ] #5 Reset reverts any unsaved changes and the stream reflects the original profile values
- [ ] #6 All config changes use HTMX POST requests; no full page reload occurs
- [ ] #7 Camera config panel works correctly with mock cameras (dev_laptop_1cam.yaml profile)
<!-- AC:END -->
