---
id: m-0
title: "M1: Core Foundation"
---

## Description

Shared Python package that both Inventarization and Gamification apps build on. Covers device abstraction (OAK, USB, mock), inference abstraction (TensorRT, Ultralytics, dummy), rate-limited pipeline stages, shared MJPEG streaming with single-encode buffer, YAML profile system, and ArUco scaffolding. Must run on Jetson with 3 cameras and on a dev laptop with a single webcam or mock source. No app-specific logic lives here.
