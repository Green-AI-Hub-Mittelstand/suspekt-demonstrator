"""Shared FastAPI and streaming helpers for demonstrator app modes."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generator, Optional

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from demonstrator.config.settings import MJPEG_BOUNDARY
from demonstrator.vision.camera import FrameGrabber


def build_app(
    *,
    title: Optional[str] = None,
    description: Optional[str] = None,
    log_prefix: str = "App",
) -> tuple[FastAPI, Jinja2Templates, Path]:
    """Create a FastAPI app with shared static/template setup."""
    app_kwargs = {}
    if title is not None:
        app_kwargs["title"] = title
    if description is not None:
        app_kwargs["description"] = description
    app = FastAPI(**app_kwargs)

    base_dir = Path(__file__).resolve().parents[3]
    static_dir = base_dir / "static"
    video_dir = base_dir / "videos"
    template_dir = base_dir / "templates"

    print(f"[INFO] {log_prefix} static root: {static_dir.resolve()}")
    print(f"[INFO] {log_prefix} video root: {video_dir.resolve()} (exists={video_dir.exists()})")

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    templates = Jinja2Templates(directory=str(template_dir))
    return app, templates, video_dir


def register_video_route(
    app: FastAPI,
    video_dir: Path,
    *,
    log_prefix: str = "app",
) -> None:
    """Register a shared static video endpoint on the given app."""

    @app.get("/videos/{video_name}")
    async def serve_video(video_name: str) -> FileResponse:
        safe_name = Path(video_name).name
        file_path = (video_dir / safe_name).resolve()
        print(f"[VIDEO] request ({log_prefix}): {video_name} -> {file_path} (exists={file_path.exists()})")
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail=f"Video {safe_name} nicht gefunden.")
        return FileResponse(str(file_path), media_type="video/mp4")


def mjpeg_stream_generator(camera: FrameGrabber) -> Generator[bytes, None, None]:
    """Yield frames as MJPEG multipart chunks."""
    boundary = MJPEG_BOUNDARY
    while True:
        frame = camera.get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        success, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not success:
            time.sleep(0.01)
            continue

        yield (
            boundary
            + b"\r\n"
            + b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8")
            + jpg.tobytes()
            + b"\r\n"
        )
        time.sleep(0.03)
