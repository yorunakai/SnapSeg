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

    def render_frame(self, image_format: Literal["jpg", "png"] = "png") -> bytes:
        if self.base_bgr is None or not self.has_images:
            ph = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(ph, "Set source path and classes, then click Load", (70, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
            if image_format == "png":
                ok, enc = cv2.imencode(".png", ph, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
            else:
                ok, enc = cv2.imencode(".jpg", ph, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
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
        if image_format == "png":
            ok, enc = cv2.imencode(".png", view, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        else:
            ok, enc = cv2.imencode(".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
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
        html_path = Path(__file__).resolve().parent / "web" / "index.html"
        if not html_path.exists():
            raise HTTPException(status_code=500, detail=f"Missing frontend file: {html_path}")
        return html_path.read_text(encoding="utf-8")

    @app.get("/api/state")
    def api_state() -> JSONResponse:
        with session.lock:
            return JSONResponse(session.state())

    @app.get("/api/frame")
    def api_frame(fmt: str = "png") -> Response:
        fmt_norm = "png" if str(fmt).lower() not in {"jpg", "jpeg"} else "jpg"
        with session.lock:
            img = session.render_frame(image_format=fmt_norm)
        media_type = "image/png" if fmt_norm == "png" else "image/jpeg"
        return Response(content=img, media_type=media_type)

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

