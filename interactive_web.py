from __future__ import annotations

import argparse
import json
import logging
import threading
import hashlib
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
logger = logging.getLogger("uvicorn.error")


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
    p.add_argument(
        "--restore-flags",
        action="store_true",
        help="Restore flagged status from autosave files (disabled by default).",
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


class BrushIn(BaseModel):
    x: float
    y: float
    radius: int = 12
    erase: bool = False


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
    visited: bool = False
    flagged: bool = False


class AnnotatorSession:
    def __init__(
        self,
        images: list[Path],
        class_list: list[str],
        out_dir: Path,
        source_path: str = "",
        backend: Literal["sam", "mobile_sam"] = "sam",
        model_id: str | None = None,
        restore_flags: bool = False,
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
        self.sam_mask: np.ndarray | None = None
        self.current_mask: np.ndarray | None = None
        self.last_latency_ms = 0.0
        self.last_score = 0.0
        self.base_bgr: np.ndarray | None = None
        self.polygon_epsilon_ratio = 0.005
        self.restore_flags = bool(restore_flags)

        self.states: dict[str, ImageSessionState] = {
            str(p): ImageSessionState(instances=[], is_dirty=False) for p in images
        }
        self._preload_flags_from_autosave()
        if self.images:
            self._load_image(0)

    @staticmethod
    def _label_color_bgr(label_name: str) -> tuple[int, int, int]:
        # Stable per-class color: same class name -> same color across frames/sessions.
        h = sum(label_name.encode("utf-8")) % 360
        hsv = np.uint8([[[h / 2, 180, 255]]])  # OpenCV hue range: 0..179
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

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

    def _image_key(self, image: Path) -> str:
        resolved = str(image.resolve())
        digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:12]
        safe_stem = image.stem.replace(" ", "_")
        return f"{safe_stem}_{digest}"

    def _autosave_json_candidates(self, image: Path) -> list[Path]:
        # Prefer collision-safe filename; keep legacy stem-based filename for compatibility.
        return [
            self.autosave_dir / f"{self._image_key(image)}_autosave.json",
            self.autosave_dir / f"{image.stem}_autosave.json",
        ]

    def _autosave_mask_path(self, image: Path, inst_idx: int, label_name: str) -> Path:
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label_name)
        return self.autosave_dir / f"{self._image_key(image)}_inst_{inst_idx}_{safe_label}.png"

    @staticmethod
    def _parse_flagged_value(flagged_raw: object) -> bool:
        if isinstance(flagged_raw, str):
            return flagged_raw.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(flagged_raw)

    @staticmethod
    def _payload_image_matches(payload: dict, image: Path) -> bool:
        raw = payload.get("image")
        if not raw:
            return True
        try:
            payload_path = Path(str(raw)).resolve()
            image_path = image.resolve()
            return str(payload_path) == str(image_path)
        except Exception:
            return False

    def _preload_flags_from_autosave(self) -> None:
        if not self.images:
            return
        for img in self.images:
            payload = None
            for autosave_json in self._autosave_json_candidates(img):
                if not autosave_json.exists():
                    continue
                try:
                    payload = json.loads(autosave_json.read_text(encoding="utf-8"))
                    if not self._payload_image_matches(payload, img):
                        payload = None
                        continue
                    break
                except Exception:
                    continue
            if payload is None:
                continue
            if self.restore_flags:
                st = self.states[str(img)]
                st.flagged = self._parse_flagged_value(payload.get("flagged", False))

    def _restore_autosave_for_current_image(self) -> None:
        if not self.has_images:
            return
        st = self._image_state()
        if st.instances:
            return

        autosave_json = None
        for candidate in self._autosave_json_candidates(self.current_image):
            if candidate.exists():
                autosave_json = candidate
                break
        if autosave_json is None:
            return

        try:
            payload = json.loads(autosave_json.read_text(encoding="utf-8"))
        except Exception:
            return
        if not self._payload_image_matches(payload, self.current_image):
            return

        items = payload.get("instances", [])
        if not isinstance(items, list):
            return
        if self.restore_flags:
            st.flagged = self._parse_flagged_value(payload.get("flagged", False))

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
        self._image_state().visited = True
        cache = self.prefetch.pop_ready(self.current_image)
        if cache is not None:
            self.service.load_cache(cache)
        else:
            self.service.set_image(self.current_image)
        self.base_bgr = cv2.cvtColor(self.service.image_rgb, cv2.COLOR_RGB2BGR)
        self.points.clear()
        self.point_labels.clear()
        self.current_box = None
        self.sam_mask = None
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
        self._preload_flags_from_autosave()
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
        autosave_json = self._autosave_json_candidates(self.current_image)[0]
        if not st.is_dirty:
            return
        if not st.instances and not st.flagged:
            for p in self._autosave_json_candidates(self.current_image):
                self.autosave_manager.submit_delete(p)
            st.is_dirty = False
            return

        payload = {
            "image": str(self.current_image),
            "updated_unix": int(time()),
            "class_list": self.class_list,
            "flagged": bool(st.flagged),
            "instances": [],
        }
        for i, (label_name, m, score) in enumerate(self._instances()):
            mask_path = self._autosave_mask_path(self.current_image, i, label_name)
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
            self.sam_mask = None
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
        mask_u8 = (pred.mask > 0).astype(np.uint8)
        self.sam_mask = mask_u8
        self.current_mask = mask_u8.copy()
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
        self.sam_mask = None
        self.current_mask = None
        self.last_score = 0.0
        self.last_latency_ms = 0.0
        self._write_autosave_if_dirty()
        return True

    def brush(self, x: float, y: float, radius: int, erase: bool) -> bool:
        if not self.has_images:
            return False
        if self.current_mask is None:
            return False
        h, w = self.service.image_rgb.shape[:2]
        cx = int(max(0, min(w - 1, round(float(x)))))
        cy = int(max(0, min(h - 1, round(float(y)))))
        rr = int(max(1, min(128, int(radius))))
        target = self.current_mask.astype(np.uint8).copy()
        value = 0 if erase else 1
        cv2.circle(target, (cx, cy), rr, color=value, thickness=-1, lineType=cv2.LINE_AA)
        self.current_mask = target
        self._image_state().is_dirty = True
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

    def toggle_flag_current(self) -> bool:
        if not self.has_images:
            return False
        st = self._image_state()
        st.flagged = not st.flagged
        st.is_dirty = True
        self._write_autosave_if_dirty()
        return st.flagged

    def progress(self) -> dict:
        total = len(self.images)
        if total == 0:
            return {
                "total_images": 0,
                "visited_count": 0,
                "labeled_count": 0,
                "flagged_count": 0,
                "total_instances": 0,
                "visit_rate": 0.0,
                "current_index": 0,
                "flagged_items": [],
            }
        states = list(self.states.values())
        visited_count = sum(1 for s in states if s.visited)
        labeled_count = sum(1 for s in states if len(s.instances) > 0)
        flagged_count = sum(1 for s in states if s.flagged)
        total_instances = sum(len(s.instances) for s in states)
        flagged_items: list[dict[str, int | str | bool]] = []
        for i, img in enumerate(self.images):
            st = self.states[str(img)]
            if st.flagged:
                flagged_items.append(
                    {
                        "index": i + 1,
                        "name": img.name,
                        "visited": bool(st.visited),
                        "labeled": len(st.instances) > 0,
                    }
                )
        return {
            "total_images": int(total),
            "visited_count": int(visited_count),
            "labeled_count": int(labeled_count),
            "flagged_count": int(flagged_count),
            "total_instances": int(total_instances),
            "visit_rate": round(float(visited_count) / float(total), 4),
            "current_index": int(self.current_idx + 1) if self.has_images else 0,
            "flagged_items": flagged_items,
        }

    def overview(self) -> dict:
        items: list[dict[str, int | str | bool]] = []
        for i, img in enumerate(self.images):
            st = self.states[str(img)]
            items.append(
                {
                    "index": i + 1,
                    "name": img.name,
                    "visited": bool(st.visited),
                    "labeled": len(st.instances) > 0,
                    "flagged": bool(st.flagged),
                    "instances": int(len(st.instances)),
                    "is_current": bool((i + 1) == (self.current_idx + 1)),
                }
            )
        return {"items": items, "total_images": int(len(self.images))}

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
                for p in self._autosave_json_candidates(self.current_image):
                    self.autosave_manager.submit_delete(p)
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
        elif action == "toggle_flag":
            self.toggle_flag_current()
        elif action == "revert_mask":
            if self.sam_mask is not None:
                self.current_mask = self.sam_mask.copy()
                self._image_state().is_dirty = True

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
        for label_name, m, _ in self._instances():
            color = np.zeros_like(view)
            cb, cg, cr = self._label_color_bgr(label_name)
            color[:, :, 0] = cb
            color[:, :, 1] = cg
            color[:, :, 2] = cr
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
                "flagged": False,
                "has_mask": False,
            }
        h, w = self.service.image_rgb.shape[:2]
        pf = self.prefetch.status()
        inst_detail = [
            {
                "index": i,
                "label": label_name,
                "score": round(float(score), 4),
                "color_bgr": list(self._label_color_bgr(label_name)),
            }
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
            "autosave": self._autosave_json_candidates(self.current_image)[0].name,
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
            "flagged": bool(self._image_state().flagged),
            "has_mask": self.current_mask is not None,
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

    @app.get("/api/progress")
    def api_progress() -> JSONResponse:
        with session.lock:
            return JSONResponse(session.progress())

    @app.get("/api/overview")
    def api_overview() -> JSONResponse:
        with session.lock:
            return JSONResponse(session.overview())

    @app.get("/api/thumb")
    def api_thumb(index: int, size: int = 240) -> Response:
        with session.lock:
            if index < 1 or index > len(session.images):
                raise HTTPException(status_code=400, detail="Invalid index")
            img_path = session.images[index - 1]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=404, detail="Image not found")
        h, w = img.shape[:2]
        max_side = max(1, int(size))
        scale = min(1.0, float(max_side) / float(max(h, w)))
        tw = max(1, int(round(w * scale)))
        th = max(1, int(round(h * scale)))
        if (tw, th) != (w, h):
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise HTTPException(status_code=500, detail="Thumbnail encode failed")
        return Response(content=enc.tobytes(), media_type="image/jpeg")

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

    @app.post("/api/brush")
    def api_brush(data: BrushIn) -> JSONResponse:
        with session.lock:
            session.brush(float(data.x), float(data.y), int(data.radius), bool(data.erase))
            out = session.state()
        return JSONResponse(out)

    @app.post("/api/action")
    def api_action(data: ActionIn) -> JSONResponse:
        with session.lock:
            try:
                before_idx = session.current_idx + 1 if session.has_images else 0
                before_flagged = bool(session._image_state().flagged) if session.has_images else False
                session.do_action(data.action, data.class_idx, data.epsilon, data.index)
                after_idx = session.current_idx + 1 if session.has_images else 0
                after_flagged = bool(session._image_state().flagged) if session.has_images else False
                logger.info(
                    "api_action action=%s before=(idx:%s flagged:%s) after=(idx:%s flagged:%s)",
                    data.action,
                    before_idx,
                    before_flagged,
                    after_idx,
                    after_flagged,
                )
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
        restore_flags=args.restore_flags,
    )
    app = build_app(session)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

