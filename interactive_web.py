from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Literal

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from src.interactive import AsyncAutosaveManager, AsyncSaveManager, MaskAnnotation, PrefetchQueue, SamEmbeddingCacheService, SaveTask


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SnapSeg interactive web annotator")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--image", type=Path, help="Single image path")
    src.add_argument("--input-dir", type=Path, help="Image directory")
    p.add_argument("--label", type=str, default="object", help="Default label")
    p.add_argument("--classes", type=str, default="", help="Comma-separated class list")
    p.add_argument("--out", type=Path, default=Path("outputs/interactive"), help="Output directory")
    p.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    p.add_argument("--port", type=int, default=7861, help="Bind port")
    p.add_argument(
        "--backend",
        type=str,
        default="sam",
        choices=["sam", "mobile_sam"],
        help="Segmentation backend",
    )
    p.add_argument(
        "--model-id",
        type=str,
        default="",
        help="Optional Hugging Face model id override for selected backend",
    )
    return p.parse_args()


def collect_images(image: Path | None, input_dir: Path | None) -> list[Path]:
    if image is not None:
        return [image]
    if input_dir is None or not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


def pick_path_dialog(mode: Literal["folder", "image"]) -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if mode == "folder":
            path = filedialog.askdirectory(title="Select Folder")
        else:
            path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All Files", "*.*")],
            )
    finally:
        root.destroy()
    return str(path or "")


class ClickIn(BaseModel):
    x: float
    y: float
    label: int


class BoxIn(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class ActionIn(BaseModel):
    action: str
    class_idx: int | None = None
    epsilon: float | None = None
    index: int | None = None


class ConfigIn(BaseModel):
    source_path: str
    classes: str = ""


@dataclass
class ImageSessionState:
    instances: list[tuple[str, np.ndarray, float]]
    is_dirty: bool = False


class AnnotatorSession:
    def __init__(
        self,
        images: list[Path],
        class_list: list[str],
        out_dir: Path,
        source_path: str = "",
        backend: Literal["sam", "mobile_sam"] = "sam",
        model_id: str | None = None,
    ) -> None:
        self.images = images
        self.class_list = class_list if class_list else ["object"]
        self.source_path = source_path
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.autosave_dir = self.out_dir / "autosave"
        self.autosave_dir.mkdir(parents=True, exist_ok=True)

        self.service = SamEmbeddingCacheService(backend=backend, model_id=model_id)
        self.prefetch = PrefetchQueue(device=self.service.device, min_free_gb=2.0)
        self.save_manager = AsyncSaveManager()
        self.autosave_manager = AsyncAutosaveManager()
        self.lock = threading.Lock()

        self.current_idx = 0
        self.class_idx = 0
        self.points: list[tuple[float, float]] = []
        self.point_labels: list[int] = []
        self.current_box: tuple[float, float, float, float] | None = None
        self.current_mask: np.ndarray | None = None
        self.last_latency_ms = 0.0
        self.last_score = 0.0
        self.base_bgr: np.ndarray | None = None
        self.polygon_epsilon_ratio = 0.005

        self.states: dict[str, ImageSessionState] = {
            str(p): ImageSessionState(instances=[], is_dirty=False) for p in images
        }
        if self.images:
            self._load_image(0)

    @property
    def current_image(self) -> Path:
        return self.images[self.current_idx]

    @property
    def has_images(self) -> bool:
        return len(self.images) > 0

    def _image_state(self) -> ImageSessionState:
        return self.states[str(self.current_image)]

    def _instances(self) -> list[tuple[str, np.ndarray, float]]:
        return self._image_state().instances

    def _restore_autosave_for_current_image(self) -> None:
        if not self.has_images:
            return
        st = self._image_state()
        if st.instances:
            return

        autosave_json = self.autosave_dir / f"{self.current_image.stem}_autosave.json"
        if not autosave_json.exists():
            return

        try:
            payload = json.loads(autosave_json.read_text(encoding="utf-8"))
        except Exception:
            return

        items = payload.get("instances", [])
        if not isinstance(items, list):
            return

        restored: list[tuple[str, np.ndarray, float]] = []
        h, w = self.service.image_rgb.shape[:2]
        for item in items:
            if not isinstance(item, dict):
                continue
            label_name = str(item.get("label", "object")).strip() or "object"
            try:
                score = float(item.get("score", 0.0))
            except Exception:
                score = 0.0
            mask_path_raw = item.get("mask_path")
            if not mask_path_raw:
                continue
            mask_path = Path(str(mask_path_raw))
            if not mask_path.is_absolute():
                mask_path = (self.out_dir / mask_path).resolve()
            if not mask_path.exists():
                continue
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                continue
            if mask_img.shape[:2] != (h, w):
                continue
            mask_bin = (mask_img > 0).astype(np.uint8)
            if int(mask_bin.sum()) == 0:
                continue
            restored.append((label_name, mask_bin, score))

        if restored:
            st.instances = restored
            st.is_dirty = False

    def _load_image(self, idx: int) -> None:
        if not self.has_images:
            return
        self.current_idx = max(0, min(idx, len(self.images) - 1))
        cache = self.prefetch.pop_ready(self.current_image)
        if cache is not None:
            self.service.load_cache(cache)
        else:
            self.service.set_image(self.current_image)
        self.base_bgr = cv2.cvtColor(self.service.image_rgb, cv2.COLOR_RGB2BGR)
        self.points.clear()
        self.point_labels.clear()
        self.current_box = None
        self.current_mask = None
        self.last_latency_ms = 0.0
        self.last_score = 0.0
        self._restore_autosave_for_current_image()
        if self.current_idx + 1 < len(self.images):
            self.prefetch.request(self.images[self.current_idx + 1])

    def configure(self, source_path: str, classes_csv: str) -> None:
        p = Path(source_path).expanduser()
        if p.is_file():
            images = [p]
        elif p.is_dir():
            images = sorted([x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in IMAGE_EXTS])
        else:
            raise FileNotFoundError(f"Path not found: {source_path}")
        if not images:
            raise RuntimeError("No images found in selected path.")

        classes = [c.strip() for c in classes_csv.split(",") if c.strip()]
        if not classes:
            classes = ["object"]

        self.source_path = str(p)
        self.images = images
        self.class_list = classes
        self.class_idx = 0
        self.current_idx = 0
        self.states = {str(img): ImageSessionState(instances=[], is_dirty=False) for img in images}
        self.points.clear()
        self.point_labels.clear()
        self.current_mask = None
        self.last_latency_ms = 0.0
        self.last_score = 0.0
        self._load_image(0)

    def _write_autosave_if_dirty(self) -> None:
        if not self.has_images:
            return
        st = self._image_state()
        autosave_json = self.autosave_dir / f"{self.current_image.stem}_autosave.json"
        if not st.is_dirty:
            return
        if not st.instances:
            self.autosave_manager.submit_delete(autosave_json)
            st.is_dirty = False
            return

        payload = {
            "image": str(self.current_image),
            "updated_unix": int(time()),
            "class_list": self.class_list,
            "instances": [],
        }
        for i, (label_name, m, score) in enumerate(self._instances()):
            mask_path = self.autosave_dir / f"{self.current_image.stem}_inst_{i}_{label_name}.png"
            cv2.imwrite(str(mask_path), (m.astype(np.uint8) * 255))
            ys, xs = np.where(m > 0)
            if len(xs) == 0 or len(ys) == 0:
                bbox = [0, 0, 0, 0]
            else:
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()), int(ys.max())
                bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
            payload["instances"].append(
                {
                    "index": i,
                    "label": label_name,
                    "score": float(score),
                    "bbox_xywh": bbox,
                    "mask_path": str(mask_path),
                }
            )
        self.autosave_manager.submit_write(autosave_json, payload)
        st.is_dirty = False

    def _run_predict(self) -> None:
        if not self.has_images:
            return
        if not self.points and self.current_box is None:
            self.current_mask = None
            self.last_latency_ms = 0.0
            self.last_score = 0.0
            return
        point_coords = [[float(x), float(y)] for x, y in self.points] if self.points else None
        point_labels = self.point_labels if self.points else None
        pred = self.service.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box_xyxy=list(self.current_box) if self.current_box is not None else None,
            multimask_output=False,
        )
        self.current_mask = pred.mask > 0
        self.last_latency_ms = pred.latency_ms
        self.last_score = pred.score

    def click(self, x: float, y: float, label: int) -> None:
        if not self.has_images:
            return
        self.points.append((x, y))
        self.point_labels.append(1 if label > 0 else 0)
        self._image_state().is_dirty = True
        self._run_predict()

    def set_box(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if not self.has_images:
            return
        h, w = self.service.image_rgb.shape[:2]
        lx = float(max(0.0, min(float(w - 1), min(x1, x2))))
        rx = float(max(0.0, min(float(w - 1), max(x1, x2))))
        ty = float(max(0.0, min(float(h - 1), min(y1, y2))))
        by = float(max(0.0, min(float(h - 1), max(y1, y2))))
        if (rx - lx) < 2.0 or (by - ty) < 2.0:
            return
        self.current_box = (lx, ty, rx, by)
        self._image_state().is_dirty = True
        self._run_predict()

    def confirm(self) -> bool:
        if not self.has_images:
            return False
        if self.current_mask is None:
            return False
        self._instances().append((self.class_list[self.class_idx], self.current_mask.astype(np.uint8), float(self.last_score)))
        self._image_state().is_dirty = True
        self.points.clear()
        self.point_labels.clear()
        self.current_box = None
        self.current_mask = None
        self.last_score = 0.0
        self.last_latency_ms = 0.0
        self._write_autosave_if_dirty()
        return True

    def remove_last_instance(self) -> bool:
        if not self.has_images:
            return False
        inst = self._instances()
        if not inst:
            return False
        inst.pop()
        st = self._image_state()
        st.is_dirty = True
        self._write_autosave_if_dirty()
        return True

    def remove_instance(self, instance_idx: int) -> bool:
        if not self.has_images:
            return False
        inst = self._instances()
        if instance_idx < 0 or instance_idx >= len(inst):
            return False
        inst.pop(instance_idx)
        st = self._image_state()
        st.is_dirty = True
        self._write_autosave_if_dirty()
        return True

    def save(self) -> bool:
        if not self.has_images:
            return False
        if self.current_mask is not None:
            self.confirm()
        inst = self._instances()
        if not inst:
            return False
        image_out = self.out_dir / self.current_image.stem
        anns: list[MaskAnnotation] = []
        for label_name, m, score in inst:
            anns.append(
                MaskAnnotation(
                    image_path=self.current_image,
                    category_name=label_name,
                    mask=m.astype(np.uint8).copy(),
                    score=score,
                )
            )
        self.save_manager.submit(
            SaveTask(
                image_path=self.current_image,
                image_out=image_out,
                annotations=anns,
                polygon_epsilon_ratio=self.polygon_epsilon_ratio,
            )
        )
        self._image_state().is_dirty = True
        self._write_autosave_if_dirty()
        return True

    def do_action(
        self,
        action: str,
        class_idx: int | None = None,
        epsilon: float | None = None,
        index: int | None = None,
    ) -> None:
        if not self.has_images and action not in {"set_epsilon"}:
            return
        if action == "undo":
            if self.points:
                self.points.pop()
                self.point_labels.pop()
                self._run_predict()
        elif action == "reset":
            self.points.clear()
            self.point_labels.clear()
            self.current_box = None
            self.current_mask = None
            self.last_score = 0.0
            self.last_latency_ms = 0.0
            # If there is no confirmed label, reset means clean state.
            if not self._instances():
                self._image_state().is_dirty = False
                self.autosave_manager.submit_delete(self.autosave_dir / f"{self.current_image.stem}_autosave.json")
        elif action == "confirm":
            self.confirm()
        elif action == "save":
            self.save()
        elif action == "undo_instance":
            self.remove_last_instance()
        elif action == "delete_instance" and index is not None:
            self.remove_instance(int(index))
        elif action == "next":
            if self.current_idx < len(self.images) - 1:
                self._write_autosave_if_dirty()
                self._load_image(self.current_idx + 1)
        elif action == "prev":
            if self.current_idx > 0:
                self._write_autosave_if_dirty()
                self._load_image(self.current_idx - 1)
        elif action == "goto" and index is not None:
            target_idx = max(0, min(int(index) - 1, len(self.images) - 1))
            if target_idx != self.current_idx:
                self._write_autosave_if_dirty()
                self._load_image(target_idx)
        elif action == "class_next":
            self.class_idx = (self.class_idx + 1) % len(self.class_list)
        elif action == "class_prev":
            self.class_idx = (self.class_idx - 1) % len(self.class_list)
        elif action == "set_class" and class_idx is not None:
            if 0 <= class_idx < len(self.class_list):
                self.class_idx = class_idx
        elif action == "set_epsilon" and epsilon is not None:
            self.polygon_epsilon_ratio = max(0.0, min(0.05, float(epsilon)))

    def render_frame(self) -> bytes:
        if self.base_bgr is None or not self.has_images:
            ph = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(ph, "Set source path and classes, then click Load", (70, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
            ok, enc = cv2.imencode(".jpg", ph, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                raise RuntimeError("Frame encode failed")
            return enc.tobytes()
        view = self.base_bgr.copy()
        for _, m, _ in self._instances():
            color = np.zeros_like(view)
            color[:, :, 1] = 120
            color[:, :, 2] = 255
            mm = m.astype(bool)
            view[mm] = (0.74 * view[mm] + 0.26 * color[mm]).astype(np.uint8)
        if self.current_mask is not None:
            color = np.zeros_like(view)
            color[:, :, 2] = 255
            mm = self.current_mask.astype(bool)
            view[mm] = (0.58 * view[mm] + 0.42 * color[mm]).astype(np.uint8)
        if self.current_box is not None:
            x1, y1, x2, y2 = [int(v) for v in self.current_box]
            cv2.rectangle(view, (x1, y1), (x2, y2), (80, 220, 255), 2, lineType=cv2.LINE_AA)
        ok, enc = cv2.imencode(".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("Frame encode failed")
        return enc.tobytes()

    def state(self) -> dict:
        if not self.has_images or self.base_bgr is None:
            return {
                "ready": False,
                "source_path": self.source_path,
                "image_name": "",
                "image_index": 0,
                "image_total": 0,
                "width": 1280,
                "height": 720,
                "class_list": self.class_list,
                "class_idx": 0,
                "instances": 0,
                "points": 0,
                "score": 0.0,
                "latency_ms": 0.0,
                "autosave": "",
                "polygon_epsilon_ratio": self.polygon_epsilon_ratio,
                "save_queue": self.save_manager.pending(),
                "autosave_queue": self.autosave_manager.pending(),
                "has_box_prompt": False,
                "backend_requested": self.service.requested_backend,
                "backend": self.service.backend,
                "model_id": self.service.model_id,
                "backend_warning": self.service.last_load_warning,
                "prefetch_free_gb": 0.0,
                "prefetch_paused_low_vram": False,
                "instances_detail": [],
            }
        h, w = self.service.image_rgb.shape[:2]
        pf = self.prefetch.status()
        inst_detail = [
            {"index": i, "label": label_name, "score": round(float(score), 4)}
            for i, (label_name, _, score) in enumerate(self._instances())
        ]
        return {
            "ready": True,
            "source_path": self.source_path,
            "image_name": self.current_image.name,
            "image_index": self.current_idx + 1,
            "image_total": len(self.images),
            "width": w,
            "height": h,
            "class_list": self.class_list,
            "class_idx": self.class_idx,
            "instances": len(self._instances()),
            "points": len(self.points),
            "score": round(float(self.last_score), 4),
            "latency_ms": round(float(self.last_latency_ms), 2),
            "autosave": f"{self.current_image.stem}_autosave.json",
            "polygon_epsilon_ratio": self.polygon_epsilon_ratio,
            "save_queue": self.save_manager.pending(),
            "autosave_queue": self.autosave_manager.pending(),
            "has_box_prompt": self.current_box is not None,
            "backend_requested": self.service.requested_backend,
            "backend": self.service.backend,
            "model_id": self.service.model_id,
            "backend_warning": self.service.last_load_warning,
            "prefetch_free_gb": round(float(pf["free_gb"]), 2),
            "prefetch_paused_low_vram": bool(pf["paused_low_vram"]),
            "instances_detail": inst_detail,
        }


def build_app(session: AnnotatorSession) -> FastAPI:
    app = FastAPI(title="SnapSeg Interactive Web")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>SnapSeg Interactive Annotator</title>
  <style>
    :root { --bg:#0f1720; --card:#1b2530; --line:#2d3a48; --text:#e6eef7; --muted:#9fb2c8; --ok:#27c26c; --ng:#ff9f40; }
    body { margin:0; background:linear-gradient(120deg,#0f1720,#121b24); color:var(--text); font-family:"Segoe UI","Helvetica Neue",Arial,sans-serif; }
    .wrap { display:grid; grid-template-columns: 280px 1fr; gap:12px; padding:12px; height:100vh; box-sizing:border-box; }
    .panel { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:12px; overflow:auto; }
    .btn { width:100%; padding:10px; margin:6px 0; border:1px solid #3a4d62; border-radius:10px; background:#223142; color:#e6eef7; cursor:pointer; font-weight:600; }
    .btn:hover { filter:brightness(1.08); }
    .btn.primary { background:#1f7a4a; border-color:#2ea162; }
    .btn.warn { background:#7a4a1f; border-color:#a16a2e; }
    .status { font-size:13px; line-height:1.6; color:var(--muted); white-space:pre-line; }
    .canvasWrap { background:#0d141b; border:1px solid var(--line); border-radius:14px; display:flex; align-items:center; justify-content:center; height:calc(100vh - 24px); }
    canvas { max-width: 100%; max-height: 100%; width:auto; height:auto; border-radius:10px; cursor:crosshair; }
    .title { font-weight:800; margin-bottom:8px; }
    .row { display:flex; gap:8px; }
    .row .btn { margin:6px 0; }
    select { width:100%; padding:8px; border-radius:8px; background:#182430; color:#e6eef7; border:1px solid #33485f; }
    .loadingOverlay {
      position: fixed;
      inset: 0;
      background: rgba(8, 12, 18, 0.66);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 9999;
      backdrop-filter: blur(2px);
    }
    .loadingCard {
      min-width: 280px;
      padding: 18px 20px;
      border-radius: 14px;
      border: 1px solid #3b4f62;
      background: #162331;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35);
      text-align: center;
    }
    .spinner {
      width: 28px;
      height: 28px;
      border: 3px solid #3f5569;
      border-top: 3px solid #4fbf84;
      border-radius: 50%;
      margin: 0 auto 12px auto;
      animation: spin 0.9s linear infinite;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
  <div id="loadingOverlay" class="loadingOverlay">
    <div class="loadingCard">
      <div class="spinner"></div>
      <div id="loadingText">Loading...</div>
    </div>
  </div>
  <div class="wrap">
    <div class="panel">
      <div class="title">SnapSeg Interactive Annotator</div>
      <div style="margin-bottom:6px;">Select a source folder or a single image:</div>
      <input id="sourcePath" type="text" readonly style="width:100%;padding:10px;border-radius:10px;background:#182430;color:#e6eef7;border:1px solid #33485f;" placeholder="Pick a folder or image path">
      <div class="row">
        <button class="btn" onclick="pickFolder()">Pick Folder</button>
        <button class="btn" onclick="pickImage()">Pick Image</button>
      </div>
      <div style="margin-top:8px;margin-bottom:6px;">Annotation Classes (comma-separated)</div>
      <input id="classInput" type="text" style="width:100%;padding:10px;border-radius:10px;background:#182430;color:#e6eef7;border:1px solid #33485f;" placeholder="scratch,particle,stain">
      <button class="btn primary" onclick="applyConfig()">Load Source</button>
      <button class="btn primary" onclick="act('save')">S Save Current Image</button>
      <button class="btn" onclick="act('confirm')">Enter Confirm Current Instance</button>
      <div class="row">
        <button class="btn" onclick="act('prev')">Previous</button>
        <button class="btn" onclick="act('next')">Next</button>
      </div>
      <div class="row">
        <input id="gotoIdx" type="number" min="1" step="1" style="flex:1;padding:10px;border-radius:10px;background:#182430;color:#e6eef7;border:1px solid #33485f;" placeholder="Image index">
        <button class="btn" style="width:120px" onclick="gotoImage()">Go</button>
      </div>
      <div class="row">
        <button class="btn" onclick="act('undo')">U Undo Point</button>
        <button class="btn warn" onclick="act('reset')">R Reset Current</button>
      </div>
      <button class="btn warn" onclick="act('undo_instance')">Backspace Undo Last Instance</button>
      <button id="boxModeBtn" class="btn" onclick="toggleBoxMode()">B Box Mode: Off</button>
      <div style="margin-top:8px;margin-bottom:6px;">Polygon Smoothing (epsilon)</div>
      <input id="eps" type="range" min="0" max="0.02" step="0.0005" value="0.005" style="width:100%;">
      <div class="row">
        <button class="btn" onclick="applyEpsilon()">Apply Epsilon</button>
      </div>
      <div class="status" id="epsText"></div>
      <div style="margin-top:8px;margin-bottom:6px;">Zoom</div>
      <div class="row">
        <button class="btn" onclick="zoomIn()">Zoom +</button>
        <button class="btn" onclick="zoomOut()">Zoom -</button>
      </div>
      <button class="btn" onclick="zoomReset()">Reset View</button>
      <div style="margin-top:8px;margin-bottom:6px;">Class</div>
      <select id="classSel" onchange="setClass(this.value)"></select>
      <div class="row">
        <button class="btn" onclick="act('class_prev')">Prev Class</button>
        <button class="btn" onclick="act('class_next')">Next Class</button>
      </div>
      <div style="margin-top:8px;margin-bottom:6px;">Confirmed Instances</div>
      <div id="instanceList" class="status"></div>
      <div class="status" id="status"></div>
      <div class="status" style="margin-top:12px;">
        Left click: positive point (+)<br/>
        Right click: negative point (-)<br/>
        Box mode: left drag to draw box prompt<br/>
        Mouse wheel: zoom in/out<br/>
        Shift + left drag: pan view<br/>
        Hotkeys: S / Enter / Space / Left / Right / U / R / Backspace / B / N / P / 1-9 / 0 (reset zoom)
      </div>
    </div>
    <div class="canvasWrap panel">
      <canvas id="cv"></canvas>
    </div>
  </div>
<script>
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const statusEl = document.getElementById('status');
const classSel = document.getElementById('classSel');
const epsEl = document.getElementById('eps');
const epsText = document.getElementById('epsText');
const gotoEl = document.getElementById('gotoIdx');
const sourceEl = document.getElementById('sourcePath');
const classInputEl = document.getElementById('classInput');
const instanceListEl = document.getElementById('instanceList');
const boxModeBtn = document.getElementById('boxModeBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');
let state = null;
let frameImg = new Image();
let hasFrame = false;
let currentImageName = "";
let viewScale = 1.0;
let viewTx = 0.0;
let viewTy = 0.0;
let dragPan = false;
let panLastX = 0.0;
let panLastY = 0.0;
let boxMode = false;
let dragBox = false;
let boxStartX = 0.0;
let boxStartY = 0.0;
let boxEndX = 0.0;
let boxEndY = 0.0;

function showLoading(msg) {
  loadingText.textContent = msg || 'Loading...';
  loadingOverlay.style.display = 'flex';
}

function hideLoading() {
  loadingOverlay.style.display = 'none';
}

function zoomReset() {
  viewScale = 1.0;
  viewTx = 0.0;
  viewTy = 0.0;
  renderCanvas();
}

function zoomAt(cx, cy, scaleFactor) {
  const oldScale = viewScale;
  const newScale = Math.max(1.0, Math.min(12.0, oldScale * scaleFactor));
  if (Math.abs(newScale - oldScale) < 1e-6) return;
  viewTx = cx - (cx - viewTx) * (newScale / oldScale);
  viewTy = cy - (cy - viewTy) * (newScale / oldScale);
  viewScale = newScale;
  renderCanvas();
}

function zoomIn() {
  zoomAt(cv.width / 2, cv.height / 2, 1.2);
}

function zoomOut() {
  zoomAt(cv.width / 2, cv.height / 2, 1.0 / 1.2);
}

function updateBoxModeButton() {
  boxModeBtn.textContent = `B Box Mode: ${boxMode ? 'On' : 'Off'}`;
  boxModeBtn.style.borderColor = boxMode ? '#2ea162' : '#3a4d62';
}

function toggleBoxMode() {
  boxMode = !boxMode;
  dragBox = false;
  updateBoxModeButton();
}

function canvasToImage(e) {
  const rect = cv.getBoundingClientRect();
  const cx = (e.clientX - rect.left) * (cv.width / rect.width);
  const cy = (e.clientY - rect.top) * (cv.height / rect.height);
  const imgX = (cx - viewTx) / viewScale;
  const imgY = (cy - viewTy) / viewScale;
  const x = Math.max(0, Math.min(state.width - 1, imgX));
  const y = Math.max(0, Math.min(state.height - 1, imgY));
  return { x, y };
}

function renderCanvas() {
  if (!hasFrame || !state) return;
  if (!state.ready) {
    cv.width = 1280;
    cv.height = 720;
    const g = ctx.createLinearGradient(0, 0, cv.width, cv.height);
    g.addColorStop(0, '#122131');
    g.addColorStop(1, '#0f1720');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = '#dbe9f7';
    ctx.font = '700 34px "Segoe UI","Helvetica Neue",Arial,sans-serif';
    ctx.fillText('SnapSeg Interactive Annotation', 72, 180);
    ctx.font = '500 20px "Segoe UI","Helvetica Neue",Arial,sans-serif';
    ctx.fillStyle = '#9fb2c8';
    ctx.fillText('1. Pick a source folder or image on the left panel', 72, 240);
    ctx.fillText('2. Set classes (comma-separated), then click "Load Source"', 72, 280);
    ctx.fillText('3. Click on defects/objects to start interactive annotation', 72, 320);
    ctx.strokeStyle = '#35506a';
    ctx.lineWidth = 2;
    ctx.strokeRect(60, 120, 760, 260);
    return;
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.setTransform(viewScale, 0, 0, viewScale, viewTx, viewTy);
  ctx.drawImage(frameImg, 0, 0, state.width, state.height);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (dragBox) {
    const x = Math.min(boxStartX, boxEndX);
    const y = Math.min(boxStartY, boxEndY);
    const w = Math.abs(boxEndX - boxStartX);
    const h = Math.abs(boxEndY - boxStartY);
    ctx.setTransform(viewScale, 0, 0, viewScale, viewTx, viewTy);
    ctx.lineWidth = 2.0 / Math.max(0.01, viewScale);
    ctx.strokeStyle = '#ffe066';
    ctx.strokeRect(x, y, w, h);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }
}

function renderInstanceList() {
  if (!state || !state.ready || !state.instances_detail || state.instances_detail.length === 0) {
    instanceListEl.innerHTML = 'No confirmed instances yet.';
    return;
  }
  const rows = state.instances_detail.map((it) => {
    const idx = Number(it.index) + 1;
    const label = String(it.label || 'object');
    const score = Number(it.score || 0).toFixed(4);
    return (
      `<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin:4px 0;">` +
      `<span>#${idx} ${label} (score ${score})</span>` +
      `<button class="btn warn" style="width:84px;margin:0;padding:6px 8px;" onclick="deleteInstance(${Number(it.index)})">Delete</button>` +
      `</div>`
    );
  });
  instanceListEl.innerHTML = rows.join('');
}

async function getState() {
  const res = await fetch('/api/state');
  state = await res.json();
  classSel.innerHTML = '';
  state.class_list.forEach((c, i) => {
    const op = document.createElement('option');
    op.value = i;
    op.textContent = `${i+1}. ${c}`;
    if (i === state.class_idx) op.selected = true;
    classSel.appendChild(op);
  });
  statusEl.textContent =
    `Image: ${state.image_index}/${state.image_total} (${state.image_name})\\n` +
    `Class: ${state.class_list[state.class_idx]}  |  Instances: ${state.instances}  |  Box Prompt: ${state.has_box_prompt}\\n` +
    `Backend(requested/effective): ${state.backend_requested} / ${state.backend}  |  Model: ${state.model_id}\\n` +
    (state.backend_warning ? `Backend warning: ${state.backend_warning}\\n` : '') +
    `Points: ${state.points}  |  Score: ${state.score}  |  Decoder: ${state.latency_ms} ms\\n` +
    `Save Queue: ${state.save_queue}  |  Autosave Queue: ${state.autosave_queue}\\n` +
    `Prefetch Free VRAM: ${state.prefetch_free_gb} GB\\n` +
    `Prefetch Paused(<2GB): ${state.prefetch_paused_low_vram}\\n` +
    `Autosave: ${state.autosave}\\n` +
    `Zoom: ${viewScale.toFixed(2)}x  |  Mode: ${boxMode ? 'Box' : 'Point'}`;
  epsEl.value = Number(state.polygon_epsilon_ratio || 0.005);
  epsText.textContent = `Current epsilon: ${Number(state.polygon_epsilon_ratio || 0.005).toFixed(4)}`;
  gotoEl.max = String(state.image_total);
  gotoEl.placeholder = `1 ~ ${state.image_total}`;
  if (state.source_path) sourceEl.value = state.source_path;
  const activeTag = (document.activeElement && document.activeElement.tagName) ? document.activeElement.tagName.toUpperCase() : "";
  const editingClassInput = (document.activeElement === classInputEl);
  if (!editingClassInput && state.class_list && state.class_list.length > 0) {
    classInputEl.value = state.class_list.join(',');
  }
  renderInstanceList();
}

async function drawFrame() {
  await getState();
  if (!state.ready) {
    hasFrame = true;
    renderCanvas();
    return;
  }
  if (currentImageName !== state.image_name) {
    currentImageName = state.image_name;
    zoomReset();
  }
  frameImg = new Image();
  frameImg.onload = () => {
    cv.width = state.width;
    cv.height = state.height;
    hasFrame = true;
    renderCanvas();
  };
  frameImg.src = `/api/frame?ts=${Date.now()}`;
}

async function act(action) {
  if (!state || !state.ready) return;
  if (action === 'next' || action === 'prev' || action === 'goto') {
    showLoading('Loading image...');
  }
  await fetch('/api/action', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action})});
  await drawFrame();
  hideLoading();
}

async function setClass(idx) {
  if (!state || !state.ready) return;
  await fetch('/api/action', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action:'set_class', class_idx: Number(idx)})});
  await drawFrame();
}

async function deleteInstance(idx) {
  if (!state || !state.ready) return;
  await fetch('/api/action', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'delete_instance', index: Number(idx)})
  });
  await drawFrame();
}

async function submitBox(x1, y1, x2, y2) {
  if (!state || !state.ready) return;
  await fetch('/api/box', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({x1, y1, x2, y2})
  });
  await drawFrame();
}

async function applyEpsilon() {
  if (!state || !state.ready) return;
  const v = Number(epsEl.value);
  await fetch('/api/action', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'set_epsilon', epsilon: v})
  });
  await drawFrame();
}

async function gotoImage() {
  if (!state || !state.ready) return;
  const idx = Number(gotoEl.value);
  if (!Number.isFinite(idx)) return;
  showLoading('Jumping to image...');
  await fetch('/api/action', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'goto', index: Math.round(idx)})
  });
  await drawFrame();
  hideLoading();
}

async function applyConfig() {
  const source_path = sourceEl.value.trim();
  const classes = classInputEl.value.trim();
  if (!source_path) {
    statusEl.textContent = "Please pick a folder/image path first.";
    return;
  }
  showLoading('Loading source and preparing SAM embeddings...');
  const res = await fetch('/api/config', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({source_path, classes})
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({detail:"unknown error"}));
    statusEl.textContent = `Error: ${err.detail || 'unknown error'}`;
    hideLoading();
    return;
  }
  await drawFrame();
  hideLoading();
}

async function pickFolder() {
  const res = await fetch('/api/pick-folder', { method: 'POST' });
  if (!res.ok) return;
  const data = await res.json();
  if (data.path) sourceEl.value = data.path;
}

async function pickImage() {
  const res = await fetch('/api/pick-image', { method: 'POST' });
  if (!res.ok) return;
  const data = await res.json();
  if (data.path) sourceEl.value = data.path;
}

cv.addEventListener('contextmenu', (e) => e.preventDefault());
cv.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (!state) return;
  const rect = cv.getBoundingClientRect();
  const cx = (e.clientX - rect.left) * (cv.width / rect.width);
  const cy = (e.clientY - rect.top) * (cv.height / rect.height);
  const factor = e.deltaY < 0 ? 1.15 : 1.0 / 1.15;
  zoomAt(cx, cy, factor);
}, { passive: false });

cv.addEventListener('mousedown', async (e) => {
  if (!state || !state.ready) return;
  if (e.button === 0 && e.shiftKey) {
    dragPan = true;
    panLastX = e.clientX;
    panLastY = e.clientY;
    cv.style.cursor = 'grabbing';
    return;
  }
  if (boxMode && e.button === 0) {
    const p = canvasToImage(e);
    dragBox = true;
    boxStartX = p.x;
    boxStartY = p.y;
    boxEndX = p.x;
    boxEndY = p.y;
    renderCanvas();
    return;
  }
  const p = canvasToImage(e);
  const x = p.x;
  const y = p.y;
  const label = (e.button === 2) ? 0 : 1;
  await fetch('/api/click', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({x, y, label})});
  await drawFrame();
});

cv.addEventListener('mousemove', async (e) => {
  if (dragPan) {
    const dx = (e.clientX - panLastX) * (cv.width / cv.getBoundingClientRect().width);
    const dy = (e.clientY - panLastY) * (cv.height / cv.getBoundingClientRect().height);
    panLastX = e.clientX;
    panLastY = e.clientY;
    viewTx += dx;
    viewTy += dy;
    renderCanvas();
    return;
  }
  if (dragBox) {
    const p = canvasToImage(e);
    boxEndX = p.x;
    boxEndY = p.y;
    renderCanvas();
  }
});

window.addEventListener('mouseup', async () => {
  if (dragPan) {
    dragPan = false;
    cv.style.cursor = 'crosshair';
    return;
  }
  if (dragBox) {
    dragBox = false;
    const x1 = Math.min(boxStartX, boxEndX);
    const y1 = Math.min(boxStartY, boxEndY);
    const x2 = Math.max(boxStartX, boxEndX);
    const y2 = Math.max(boxStartY, boxEndY);
    if ((x2 - x1) >= 2 && (y2 - y1) >= 2) {
      await submitBox(x1, y1, x2, y2);
    } else {
      renderCanvas();
    }
  }
});

window.addEventListener('keydown', async (e) => {
  const tag = (document.activeElement && document.activeElement.tagName) ? document.activeElement.tagName.toUpperCase() : "";
  const isTypingContext = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
  if (isTypingContext) return;
  if (e.key === 's' || e.key === 'S') return act('save');
  if (e.key === 'Enter') return act('confirm');
  if (e.key === 'u' || e.key === 'U') return act('undo');
  if (e.key === 'Backspace') { e.preventDefault(); return act('undo_instance'); }
  if (e.key === 'r' || e.key === 'R') return act('reset');
  if (e.key === ' ') { e.preventDefault(); return act('next'); }
  if (e.key === 'ArrowRight') return act('next');
  if (e.key === 'ArrowLeft') return act('prev');
  if (e.key === 'n' || e.key === 'N') return act('class_next');
  if (e.key === 'p' || e.key === 'P') return act('class_prev');
  if (e.key === 'b' || e.key === 'B') return toggleBoxMode();
  if (e.key === '+' || e.key === '=') return zoomIn();
  if (e.key === '-' || e.key === '_') return zoomOut();
  if (e.key === '0') return zoomReset();
  if (/^[1-9]$/.test(e.key)) return setClass(Number(e.key) - 1);
});

updateBoxModeButton();
drawFrame();
</script>
</body>
</html>
"""

    @app.get("/api/state")
    def api_state() -> JSONResponse:
        with session.lock:
            return JSONResponse(session.state())

    @app.get("/api/frame")
    def api_frame() -> Response:
        with session.lock:
            jpg = session.render_frame()
        return Response(content=jpg, media_type="image/jpeg")

    @app.post("/api/click")
    def api_click(data: ClickIn) -> JSONResponse:
        with session.lock:
            session.click(float(data.x), float(data.y), int(data.label))
            out = session.state()
        return JSONResponse(out)

    @app.post("/api/box")
    def api_box(data: BoxIn) -> JSONResponse:
        with session.lock:
            session.set_box(float(data.x1), float(data.y1), float(data.x2), float(data.y2))
            out = session.state()
        return JSONResponse(out)

    @app.post("/api/action")
    def api_action(data: ActionIn) -> JSONResponse:
        with session.lock:
            try:
                session.do_action(data.action, data.class_idx, data.epsilon, data.index)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            out = session.state()
        return JSONResponse(out)

    @app.post("/api/config")
    def api_config(data: ConfigIn) -> JSONResponse:
        with session.lock:
            try:
                session.configure(data.source_path, data.classes)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return JSONResponse(session.state())

    @app.post("/api/pick-folder")
    def api_pick_folder() -> JSONResponse:
        try:
            path = pick_path_dialog("folder")
            return JSONResponse({"path": path})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/pick-image")
    def api_pick_image() -> JSONResponse:
        try:
            path = pick_path_dialog("image")
            return JSONResponse({"path": path})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return app


def main() -> None:
    args = parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not classes:
        classes = [args.label]

    images: list[Path] = []
    source_path = ""
    if args.image is not None or args.input_dir is not None:
        images = [p for p in collect_images(args.image, args.input_dir) if p.exists()]
        source_path = str(args.image if args.image is not None else args.input_dir)
    model_id = args.model_id.strip() or None
    session = AnnotatorSession(
        images=images,
        class_list=classes,
        out_dir=args.out,
        source_path=source_path,
        backend=args.backend,
        model_id=model_id,
    )
    app = build_app(session)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

