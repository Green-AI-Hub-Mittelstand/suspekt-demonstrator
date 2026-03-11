#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$ROOT_DIR/models"

CENTER_SOURCE="251104_real_Y12m_detect_29cls.pt"
NUBS_SOURCE="251105_nubs_Y12m_detect_2cls.pt"

CENTER_TARGET="251104_real_Y12m_detect_29cls_320_FP16.engine"
NUBS_TARGET="251105_nubs_Y12m_detect_2cls_320_FP16.engine"

echo "Converting YOLO models to TensorRT engines..."
echo "Project: $ROOT_DIR"
echo

if [ ! -d "$MODELS_DIR" ]; then
    echo "Models directory not found: $MODELS_DIR"
    exit 1
fi

if [ ! -f "$MODELS_DIR/$CENTER_SOURCE" ]; then
    echo "Missing source model: $MODELS_DIR/$CENTER_SOURCE"
    exit 1
fi

if [ ! -f "$MODELS_DIR/$NUBS_SOURCE" ]; then
    echo "Missing source model: $MODELS_DIR/$NUBS_SOURCE"
    exit 1
fi

cd "$ROOT_DIR"

echo "Converting $NUBS_SOURCE ..."
ultralytics export model="$MODELS_DIR/$NUBS_SOURCE" format=engine device=0 imgsz=320 half=True
mv -f "$MODELS_DIR/${NUBS_SOURCE%.pt}.engine" "$MODELS_DIR/$NUBS_TARGET"
echo "[OK] $NUBS_TARGET"
echo

echo "Converting $CENTER_SOURCE ..."
ultralytics export model="$MODELS_DIR/$CENTER_SOURCE" format=engine device=0 imgsz=320 half=True
mv -f "$MODELS_DIR/${CENTER_SOURCE%.pt}.engine" "$MODELS_DIR/$CENTER_TARGET"
echo "[OK] $CENTER_TARGET"
echo

echo "Model conversion complete."
