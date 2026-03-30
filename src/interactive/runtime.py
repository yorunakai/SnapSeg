from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import sleep
import json

import cv2
import numpy as np
import torch

from .exporter import AnnotationExporter, MaskAnnotation
from .sam_service import SamEmbeddingCacheService, SamImageCache


def gpu_free_gb(device: str) -> float:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return 999.0
    try:
        dev_idx = 0
        if ":" in device:
            dev_idx = int(device.split(":")[1])
        free_bytes, _ = torch.cuda.mem_get_info(dev_idx)
        return float(free_bytes) / float(1024 ** 3)
    except Exception:
        return 0.0


class PrefetchQueue:
    def __init__(self, device: str, min_free_gb: float = 2.0) -> None:
        self.device = device
        self.min_free_gb = float(min_free_gb)
        self.max_retries = 3
        self._service = SamEmbeddingCacheService(device=device)
        self._lock = threading.Lock()
        self._requested: deque[Path] = deque()
        self._ready: dict[str, SamImageCache] = {}
        self._retry_count: dict[str, int] = {}
        self._last_free_gb: float = 999.0
        self._paused_low_vram: bool = False
        self._stop = False
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop = True

    def request(self, image_path: Path) -> None:
        with self._lock:
            key = str(image_path)
            if key in self._ready:
                return
            if any(str(p) == key for p in self._requested):
                return
            self._requested.append(image_path)

    def clear_pending(self) -> None:
        with self._lock:
            self._requested.clear()
            self._retry_count.clear()

    def pop_ready(self, image_path: Path) -> SamImageCache | None:
        with self._lock:
            return self._ready.pop(str(image_path), None)

    def status(self) -> dict[str, float | bool]:
        with self._lock:
            return {
                "free_gb": float(self._last_free_gb),
                "paused_low_vram": bool(self._paused_low_vram),
            }

    def _loop(self) -> None:
        while not self._stop:
            with self._lock:
                target = self._requested.popleft() if self._requested else None
            if target is None:
                sleep(0.05)
                continue

            free_gb = gpu_free_gb(self.device)
            with self._lock:
                self._last_free_gb = free_gb
                self._paused_low_vram = free_gb < self.min_free_gb
            if free_gb < self.min_free_gb:
                with self._lock:
                    self._requested.insert(0, target)
                sleep(0.1)
                continue

            try:
                self._service.set_image(target)
                cache = self._service.snapshot_cache(to_cpu=True)
                with self._lock:
                    key = str(target)
                    self._ready[key] = cache
                    self._retry_count.pop(key, None)
            except Exception:
                with self._lock:
                    key = str(target)
                    retries = int(self._retry_count.get(key, 0)) + 1
                    if retries <= self.max_retries:
                        self._retry_count[key] = retries
                        self._requested.append(target)
                    else:
                        self._retry_count.pop(key, None)
                sleep(0.2)


@dataclass
class SaveTask:
    image_path: Path
    image_out: Path
    annotations: list[MaskAnnotation]
    polygon_epsilon_ratio: float


class AsyncSaveManager:
    def __init__(self) -> None:
        self._q: queue.Queue[SaveTask] = queue.Queue()
        self._stop = False
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def pending(self) -> int:
        return self._q.qsize()

    def stop(self) -> None:
        self._stop = True

    def submit(self, task: SaveTask) -> None:
        self._q.put(task)

    def _loop(self) -> None:
        while not self._stop:
            try:
                task = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                task.image_out.mkdir(parents=True, exist_ok=True)
                for i, ann in enumerate(task.annotations):
                    mask_u8 = (ann.mask.astype(np.uint8) * 255)
                    cv2.imwrite(str(task.image_out / f"{task.image_path.stem}_mask_{i}_{ann.category_name}.png"), mask_u8)

                exp = AnnotationExporter(polygon_epsilon_ratio=task.polygon_epsilon_ratio)
                exp.export_coco(task.annotations, task.image_out / "annotations_coco.json")
                exp.export_yolo_seg(task.annotations, task.image_out / "labels_yolo_seg", task.image_out / "classes_yolo_seg.txt")
                exp.export_yolo_bbox(task.annotations, task.image_out / "labels_yolo_bbox", task.image_out / "classes_yolo_bbox.txt")
            finally:
                self._q.task_done()


@dataclass
class AutosaveTask:
    json_path: Path
    payload: dict | None
    masks: list[tuple[Path, np.ndarray]] | None = None
    delete_only: bool = False


class AsyncAutosaveManager:
    def __init__(self) -> None:
        self._q: queue.Queue[AutosaveTask] = queue.Queue()
        self._stop = False
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def pending(self) -> int:
        return self._q.qsize()

    def stop(self) -> None:
        self._stop = True

    def submit_write(self, json_path: Path, payload: dict, masks: list[tuple[Path, np.ndarray]] | None = None) -> None:
        self._q.put(AutosaveTask(json_path=json_path, payload=payload, masks=masks, delete_only=False))

    def submit_delete(self, json_path: Path) -> None:
        self._q.put(AutosaveTask(json_path=json_path, payload=None, delete_only=True))

    def _loop(self) -> None:
        while not self._stop:
            try:
                task = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if task.delete_only:
                    if task.json_path.exists():
                        task.json_path.unlink(missing_ok=True)
                    continue
                for mask_path, mask_u8 in (task.masks or []):
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(mask_path), mask_u8)
                task.json_path.parent.mkdir(parents=True, exist_ok=True)
                task.json_path.write_text(json.dumps(task.payload, ensure_ascii=False, indent=2), encoding="utf-8")
            finally:
                self._q.task_done()
