---
id: TASK-1
title: Set up Python package skeleton and profile system
status: To Do
assignee: []
created_date: '2026-03-10 21:31'
labels:
  - backend
  - infra
milestone: m-0
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bootstrap the `demonstrator` Python package that all app modes will build on. This includes the full directory structure, packaging config, shared data models, and a YAML-based profile system for selecting hardware targets at startup.

**Package structure to create:**
```
src/demonstrator/
  core/
    config.py       # Pydantic settings + YAML profile loader
    models.py       # Detection, Frame, PartClass, ROI dataclasses
  devices/          # Empty stubs only (implemented in later task)
  inference/        # Empty stubs only (implemented in later task)
  pipeline/         # Empty stubs only (implemented in later task)
  api/
    app.py          # Minimal FastAPI factory (no routes yet)
  apps/
    inventarization/
      __init__.py
      app.py        # Stub
    gamification/
      __init__.py
      app.py        # Stub
  profiles/
    demo_jetson_3cam.yaml
    dev_laptop_1cam.yaml
    replay.yaml
pyproject.toml
```

**Profile schema** (YAML, loaded via Pydantic):
- `hardware.cameras`: list of camera configs — `id` (left/center/right), `type` (oak/opencv/mock), `device_index`, `rotation` (0/90/180/270), `enabled`
- `pipeline.infer_fps`: max inference rate per camera
- `pipeline.render_fps`: max render/encode rate
- `app.mode`: `inventarization` or `gamification`

**pyproject.toml** must declare all dependencies (FastAPI, uvicorn, Pydantic, PyYAML, opencv-python, depthai, ultralytics) with optional groups for `[dev]` (pytest) and `[jetson]` (tensorrt extras).

No business logic, no routes, no UI in this task — just the skeleton that compiles and is importable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Package is installable via `pip install -e .` on a clean virtualenv
- [ ] #2 Both `dev_laptop_1cam.yaml` and `demo_jetson_3cam.yaml` profiles load without error via the profile loader
- [ ] #3 Profile loader raises a clear error when a required field is missing or an unknown camera type is specified
- [ ] #4 `from demonstrator.core.models import Detection, Frame, ROI` works without import errors
- [ ] #5 Stub app factories for both `inventarization` and `gamification` are importable
- [ ] #6 pyproject.toml includes all required dependencies in the correct optional groups
<!-- AC:END -->
