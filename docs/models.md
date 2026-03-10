# Models

The demonstrator currently has three active models in `models/`.

## Overview

| File                                    | Purpose                    | Task     | Base YOLO  | Classes |
| --------------------------------------- | -------------------------- | -------- | ---------- | ------- |
| `251101_components_Y11n_detect_6cls.pt` | Compact component detector | `detect` | `YOLO11n`  | 6       |
| `251104_real_Y12m_detect_29cls.pt`      | Broad component detector   | `detect` | `YOLOv12m` | 29      |
| `251105_nubs_Y12m_detect_2cls.pt`       | Nub orientation detector   | `detect` | `YOLOv12m` | 2       |

## `251101_components_Y11n_detect_6cls.pt`

This is the compact demonstrator component model. It detects the six core demonstrator classes:

- `Diagonale`
- `Gerade`
- `Mutternstab`
- `Noppenscheiben`
- `Schraube`
- `Versetzte_gerade`

Metadata from `tools/inspect_model.py`:

- Task: `detect`
- Saved: `2025-11-01T18:58:23.711625`
- Ultralytics version: `8.3.34`
- Base architecture: `YOLO11n`
- Parameters: `2,591,010`
- Compute: `6.4 GFLOPs`
- Training image size: `640`
- Target epochs: `1000`
- Batch size: `216`
- Optimizer: `auto`
- Initial learning rate: `0.01`
- Dataset config: `config.yaml`

This is the smallest current model and the most focused one for the main demonstrator part classes.

## `251104_real_Y12m_detect_29cls.pt`

This is the broader component detector. It predicts the following 29 classes:

- `Auszug`
- `Diagonale`
- `Doppeltuer`
- `Doppeltuerblatt`
- `Einzeltuer`
- `Fachboden`
- `Fachboden mit Verstaerkung`
- `Gerade`
- `Griff`
- `Magazin`
- `Mutternstab`
- `Noppenscheiben`
- `NubsDown`
- `NubsUp`
- `Regal`
- `Rolle`
- `Schraube`
- `Seitenverkleidung`
- `Seitenverkleidung---0-0`
- `Seitenverkleidung---0-IN`
- `Seitenverkleidung---IN-IN`
- `Seitenverkleidung-0-IN`
- `Sideboard`
- `Sockelfuss`
- `Systemboden`
- `Tresen`
- `Verbindungswinkel`
- `Verkleidung`
- `Winkelfuss`

Metadata from `tools/inspect_model.py`:

- Task: `detect`
- Saved: `2025-11-04T16:59:36.910480`
- Ultralytics version: `8.3.225`
- Base architecture: `YOLOv12m`
- Parameters: `20,159,847`
- Compute: `67.9 GFLOPs`
- Training image size: `640`
- Target epochs: `200`
- Batch size: `16`
- Optimizer: `auto`
- Initial learning rate: `0.01`
- Dataset config: `/content/GAIH System 180_42_KL_v4/data.yaml`

This model is much broader than the six-class model and covers a wider set of furniture and assembly classes.

## `251105_nubs_Y12m_detect_2cls.pt`

This is the dedicated nub orientation model for the side-camera views. It predicts:

- `NubsDown`
- `NubsUp`

Metadata from `tools/inspect_model.py`:

- Task: `detect`
- Saved: `2025-11-05T21:52:14.179849`
- Ultralytics version: `8.3.225`
- Base architecture: `YOLOv12m`
- Parameters: `20,139,030`
- Compute: `67.7 GFLOPs`
- Training image size: `640`
- Target epochs: `200`
- Batch size: `16`
- Optimizer: `auto`
- Initial learning rate: `0.01`
- Dataset config: `/content/NubsDetection.v4i.yolov11/data.yaml`

This model is intended for the side cameras, where the main task is to detect nub orientation rather than identify the full part class.

## Practical difference between the three models

- `251101_components_Y11n_detect_6cls.pt` is the narrow model for the six core demonstrator component classes.
- `251104_real_Y12m_detect_29cls.pt` is the broad multi-class model with many more furniture-related classes.
- `251105_nubs_Y12m_detect_2cls.pt` is the specialized side-camera model for `NubsDown` and `NubsUp`.

## Deployment

These `.pt` files are training artifacts. For Jetson deployment, they need to be exported to `.onnx` and then to TensorRT `.engine`.

All three inspected models were trained with `imgsz=640`. If the demonstrator runtime uses a smaller inference size such as `320`, that is a deployment optimization step rather than the training resolution.
