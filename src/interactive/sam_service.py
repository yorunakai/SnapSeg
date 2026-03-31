from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep, time
from typing import Any, Literal, Protocol

import numpy as np
import torch
from PIL import Image


@dataclass
class SamPrediction:
    mask: np.ndarray
    score: float
    latency_ms: float


@dataclass
class TransformersEmbeddingCache:
    image_rgb: np.ndarray
    image_embeddings: torch.Tensor
    orig_h: int
    orig_w: int
    reshape_h: int
    reshape_w: int


@dataclass
class NativeEmbeddingCache:
    image_rgb: np.ndarray
    features: torch.Tensor
    original_size: tuple[int, int]
    input_size: tuple[int, int]


@dataclass
class SamImageCache:
    image_path: Path
    backend_kind: str
    payload: Any


class SamBackend(Protocol):
    def set_image(self, image_rgb: np.ndarray) -> None:
        ...

    def compute_embedding(self, image_rgb: np.ndarray) -> Any:
        ...

    def load_embedding(self, cache: Any) -> None:
        ...

    def export_embedding(self, to_cpu: bool = True) -> Any:
        ...

    def predict(
        self,
        point_coords: list[list[float]] | None,
        point_labels: list[int] | None,
        box_xyxy: list[float] | None,
        mask_input: np.ndarray | None,
        multimask_output: bool,
    ) -> SamPrediction:
        ...

    @property
    def image_rgb(self) -> np.ndarray:
        ...


CHECKPOINT_SEARCH_ORDER = [
    "sam_vit_b_01ec64.pth",
    "sam_vit_b.pth",
    "sam_vit_h_4b8939.pth",
    "sam_vit_l_0b3195.pth",
]

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "Model Checkpoints"


class TransformersSamBackend:
    def __init__(self, model_id: str, device: str, local_files_only: bool) -> None:
        from transformers import SamModel, SamProcessor

        self.model_id = model_id
        self.device = device
        self._infer_lock = threading.Lock()
        self._processor = SamProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        self._model = SamModel.from_pretrained(model_id, local_files_only=local_files_only).to(device)
        self._model.eval()

        self._image_rgb: np.ndarray | None = None
        self._image_embeddings: torch.Tensor | None = None
        self._orig_h: int = 0
        self._orig_w: int = 0
        self._reshape_h: int = 0
        self._reshape_w: int = 0

    def set_image(self, image_rgb: np.ndarray) -> None:
        cache = self.compute_embedding(image_rgb)
        self.load_embedding(cache)

    def compute_embedding(self, image_rgb: np.ndarray) -> TransformersEmbeddingCache:
        with self._infer_lock:
            pil_image = Image.fromarray(image_rgb)
            inputs = self._processor(images=pil_image, return_tensors="pt")
            orig = inputs["original_sizes"][0].tolist()
            reshaped = inputs["reshaped_input_sizes"][0].tolist()
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.no_grad():
                embeddings = self._model.get_image_embeddings(pixel_values)
            return TransformersEmbeddingCache(
                image_rgb=image_rgb.copy(),
                image_embeddings=embeddings.detach().cpu(),
                orig_h=int(orig[0]),
                orig_w=int(orig[1]),
                reshape_h=int(reshaped[0]),
                reshape_w=int(reshaped[1]),
            )

    def load_embedding(self, cache: TransformersEmbeddingCache) -> None:
        with self._infer_lock:
            self._image_rgb = cache.image_rgb.copy()
            self._orig_h = int(cache.orig_h)
            self._orig_w = int(cache.orig_w)
            self._reshape_h = int(cache.reshape_h)
            self._reshape_w = int(cache.reshape_w)
            self._image_embeddings = cache.image_embeddings.to(self.device)

    def export_embedding(self, to_cpu: bool = True) -> TransformersEmbeddingCache:
        if self._image_rgb is None or self._image_embeddings is None:
            raise RuntimeError("No image cache available. Call set_image first.")
        emb = self._image_embeddings.detach()
        if to_cpu:
            emb = emb.cpu()
        return TransformersEmbeddingCache(
            image_rgb=self._image_rgb.copy(),
            image_embeddings=emb,
            orig_h=self._orig_h,
            orig_w=self._orig_w,
            reshape_h=self._reshape_h,
            reshape_w=self._reshape_w,
        )

    @property
    def image_rgb(self) -> np.ndarray:
        if self._image_rgb is None:
            raise RuntimeError("Image is not set. Call set_image first.")
        return self._image_rgb

    def predict(
        self,
        point_coords: list[list[float]] | None,
        point_labels: list[int] | None,
        box_xyxy: list[float] | None,
        mask_input: np.ndarray | None,
        multimask_output: bool,
    ) -> SamPrediction:
        del mask_input
        if self._image_embeddings is None:
            raise RuntimeError("Image embeddings are not cached. Call set_image first.")
        has_points = bool(point_coords)
        has_box = bool(box_xyxy)
        if not has_points and not has_box:
            raise ValueError("At least one prompt is required (point or box).")
        if has_points:
            if point_labels is None:
                raise ValueError("point_labels is required when point_coords is provided.")
            if len(point_coords or []) != len(point_labels):
                raise ValueError("point_coords and point_labels must have same length.")

        input_points: torch.Tensor | None = None
        input_labels: torch.Tensor | None = None
        if has_points:
            coords_np = np.asarray(point_coords, dtype=np.float32)
            coords_np[:, 0] = (coords_np[:, 0] + 0.5) * (float(self._reshape_w) / max(1.0, float(self._orig_w)))
            coords_np[:, 1] = (coords_np[:, 1] + 0.5) * (float(self._reshape_h) / max(1.0, float(self._orig_h)))
            input_points = torch.from_numpy(coords_np).to(self.device).unsqueeze(0).unsqueeze(0)
            input_labels = torch.tensor(point_labels, dtype=torch.int64, device=self.device).unsqueeze(0).unsqueeze(0)

        input_boxes: torch.Tensor | None = None
        if has_box:
            if box_xyxy is None or len(box_xyxy) != 4:
                raise ValueError("box_xyxy must be [x1, y1, x2, y2].")
            bx = np.asarray(box_xyxy, dtype=np.float32).copy()
            bx[0] = (bx[0] + 0.5) * (float(self._reshape_w) / max(1.0, float(self._orig_w)))
            bx[1] = (bx[1] + 0.5) * (float(self._reshape_h) / max(1.0, float(self._orig_h)))
            bx[2] = (bx[2] + 0.5) * (float(self._reshape_w) / max(1.0, float(self._orig_w)))
            bx[3] = (bx[3] + 0.5) * (float(self._reshape_h) / max(1.0, float(self._orig_h)))
            input_boxes = torch.from_numpy(bx).to(self.device).unsqueeze(0).unsqueeze(0)

        t0 = perf_counter()
        with self._infer_lock:
            with torch.no_grad():
                model_kwargs: dict[str, Any] = {
                    "image_embeddings": self._image_embeddings,
                    "multimask_output": multimask_output,
                }
                if input_points is not None and input_labels is not None:
                    model_kwargs["input_points"] = input_points
                    model_kwargs["input_labels"] = input_labels
                if input_boxes is not None:
                    model_kwargs["input_boxes"] = input_boxes
                outputs = self._model(**model_kwargs)

                iou_scores = outputs.iou_scores[0, 0]
                best_idx = int(torch.argmax(iou_scores).item())
                original_sizes = torch.tensor(
                    [[self._orig_h, self._orig_w]],
                    dtype=torch.int64,
                    device=outputs.pred_masks.device,
                )
                reshaped_input_sizes = torch.tensor(
                    [[self._reshape_h, self._reshape_w]],
                    dtype=torch.int64,
                    device=outputs.pred_masks.device,
                )
                post_masks = self._processor.image_processor.post_process_masks(
                    outputs.pred_masks,
                    original_sizes=original_sizes,
                    reshaped_input_sizes=reshaped_input_sizes,
                )
                best_post = post_masks[0][0, best_idx]
                binary = (best_post > 0.0).to(torch.uint8)
                score = float(iou_scores[best_idx].item())
        latency_ms = (perf_counter() - t0) * 1000.0
        mask_np = binary.detach().cpu().numpy().astype(np.uint8)
        return SamPrediction(mask=mask_np, score=score, latency_ms=latency_ms)


class NativeSamBackend:
    def __init__(self, checkpoint_path: Path, device: str) -> None:
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except Exception:
            raise RuntimeError(
                "Local .pth checkpoint found but 'segment_anything' is not installed. "
                "Install it with: pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from None

        self.checkpoint_path = checkpoint_path
        self.device = device
        self._infer_lock = threading.Lock()
        model_type = self._infer_model_type(checkpoint_path)
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        sam.to(device)
        sam.eval()
        self._predictor = SamPredictor(sam)
        self._image_rgb: np.ndarray | None = None

    @staticmethod
    def _infer_model_type(checkpoint_path: Path) -> str:
        name = checkpoint_path.stem.lower()
        if "vit_h" in name or "_h_" in name:
            return "vit_h"
        if "vit_l" in name or "_l_" in name:
            return "vit_l"
        return "vit_b"

    def set_image(self, image_rgb: np.ndarray) -> None:
        cache = self.compute_embedding(image_rgb)
        self.load_embedding(cache)

    def compute_embedding(self, image_rgb: np.ndarray) -> NativeEmbeddingCache:
        with self._infer_lock:
            self._predictor.set_image(image_rgb)
            feats = self._predictor.features
            if feats is None:
                raise RuntimeError("SAM native backend failed to produce image features.")
            return NativeEmbeddingCache(
                image_rgb=image_rgb.copy(),
                features=feats.detach().cpu().clone(),
                original_size=tuple(self._predictor.original_size),
                input_size=tuple(self._predictor.input_size),
            )

    def load_embedding(self, cache: NativeEmbeddingCache) -> None:
        with self._infer_lock:
            self._predictor.features = cache.features.to(self.device)
            self._predictor.original_size = tuple(cache.original_size)
            self._predictor.input_size = tuple(cache.input_size)
            self._predictor.is_image_set = True
            self._image_rgb = cache.image_rgb.copy()

    def export_embedding(self, to_cpu: bool = True) -> NativeEmbeddingCache:
        with self._infer_lock:
            feats = self._predictor.features
            if self._image_rgb is None or feats is None:
                raise RuntimeError("No image cache available. Call set_image first.")
            out_feats = feats.detach()
            if to_cpu:
                out_feats = out_feats.cpu()
            return NativeEmbeddingCache(
                image_rgb=self._image_rgb.copy(),
                features=out_feats.clone(),
                original_size=tuple(self._predictor.original_size),
                input_size=tuple(self._predictor.input_size),
            )

    @property
    def image_rgb(self) -> np.ndarray:
        if self._image_rgb is None:
            raise RuntimeError("Image is not set. Call set_image first.")
        return self._image_rgb

    def predict(
        self,
        point_coords: list[list[float]] | None,
        point_labels: list[int] | None,
        box_xyxy: list[float] | None,
        mask_input: np.ndarray | None,
        multimask_output: bool,
    ) -> SamPrediction:
        if point_coords and point_labels is None:
            raise ValueError("point_labels is required when point_coords is provided.")
        t0 = perf_counter()
        coords_np = np.asarray(point_coords, dtype=np.float32) if point_coords else None
        labels_np = np.asarray(point_labels, dtype=np.int32) if point_labels else None
        box_np = np.asarray(box_xyxy, dtype=np.float32) if box_xyxy is not None else None
        mask_np = np.asarray(mask_input, dtype=np.float32) if mask_input is not None else None

        with self._infer_lock:
            masks, scores, _ = self._predictor.predict(
                point_coords=coords_np,
                point_labels=labels_np,
                box=box_np,
                mask_input=mask_np,
                multimask_output=multimask_output,
            )
        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM native backend returned no masks.")
        idx = int(np.argmax(scores)) if scores is not None and len(scores) > 0 else 0
        best_mask = (masks[idx] > 0).astype(np.uint8)
        score = float(scores[idx]) if scores is not None and len(scores) > 0 else 0.0
        latency_ms = (perf_counter() - t0) * 1000.0
        return SamPrediction(mask=best_mask, score=score, latency_ms=latency_ms)


_service_registry: dict[tuple[str, str, str, str], "SamEmbeddingCacheService"] = {}
_registry_lock = threading.Lock()


def get_global_service(
    backend: Literal["sam", "mobile_sam"] = "sam",
    model_id: str | None = None,
    device: str | None = None,
    checkpoint_dir: Path | None = None,
) -> "SamEmbeddingCacheService":
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_model_id = model_id or SamEmbeddingCacheService._default_model_id(backend)
    resolved_ckpt = str((checkpoint_dir or DEFAULT_CHECKPOINT_DIR).resolve())
    key = (backend, resolved_model_id, resolved_device, resolved_ckpt)
    svc = _service_registry.get(key)
    if svc is not None:
        return svc
    with _registry_lock:
        svc = _service_registry.get(key)
        if svc is None:
            svc = SamEmbeddingCacheService(
                backend=backend,
                model_id=resolved_model_id,
                device=resolved_device,
                checkpoint_dir=Path(resolved_ckpt),
            )
            _service_registry[key] = svc
        return svc


class SamEmbeddingCacheService:
    def __init__(
        self,
        backend: Literal["sam", "mobile_sam"] = "sam",
        model_id: str | None = None,
        device: str | None = None,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.requested_backend = backend
        self.backend = backend
        self.model_id = model_id or self._default_model_id(backend)
        self.last_load_warning: str = ""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = (checkpoint_dir or DEFAULT_CHECKPOINT_DIR).resolve()

        if self.device == "cpu":
            max_threads = max(1, (os.cpu_count() or 2) // 2)
            try:
                torch.set_num_threads(max_threads)
            except Exception:
                pass
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass

        self._backend: SamBackend | None = None
        self._backend_kind: str = ""
        self._model_load_lock = threading.Lock()

        self._image_path: Path | None = None
        self.model_status: str = "idle"
        self.model_error: str = ""
        self.model_source: str = "unknown"
        self.model_checkpoint_name: str = ""
        self.model_loading_started_at: float | None = None
        self.last_model_load_ms: float | None = None

    @staticmethod
    def _default_model_id(backend: Literal["sam", "mobile_sam"]) -> str:
        if backend == "mobile_sam":
            return "nielsr/slimsam-50-uniform"
        return "facebook/sam-vit-base"

    @staticmethod
    def _is_cache_complete(model_id: str) -> bool:
        try:
            from huggingface_hub import try_to_load_from_cache
        except Exception:
            return False
        required_cfg = ("config.json", "preprocessor_config.json")
        for fname in required_cfg:
            if try_to_load_from_cache(model_id, fname) is None:
                return False
        weight_candidates = ("pytorch_model.bin", "model.safetensors", "model.safetensors.index.json")
        return any(try_to_load_from_cache(model_id, fname) is not None for fname in weight_candidates)

    @staticmethod
    def find_local_checkpoint(checkpoint_dir: Path) -> tuple[Path | None, bool]:
        if not checkpoint_dir.exists():
            return None, False
        for name in CHECKPOINT_SEARCH_ORDER:
            p = checkpoint_dir / name
            if p.is_file():
                return p, True
        candidates = sorted(checkpoint_dir.glob("*.pth"))
        if candidates:
            return candidates[0], False
        return None, False

    @staticmethod
    def _read_image_rgb(image_path: Path) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        return np.array(image, dtype=np.uint8)

    def _build_transformers_backend(self, model_id: str) -> tuple[TransformersSamBackend, str]:
        use_local = self._is_cache_complete(model_id)
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                return TransformersSamBackend(model_id=model_id, device=self.device, local_files_only=use_local), (
                    "hf_cache" if use_local else "hf_download"
                )
            except Exception as exc:
                last_exc = exc
                if use_local:
                    use_local = False
                    continue
                if attempt < 2:
                    sleep(2**attempt)
        if last_exc is None:
            raise RuntimeError("Unknown model load error")
        raise RuntimeError(
            f"Failed to load SAM model '{model_id}'. Use --model-id to specify a valid checkpoint."
        ) from last_exc

    def _ensure_model_unlocked(self) -> None:
        local_ckpt, is_standard = self.find_local_checkpoint(self.checkpoint_dir)
        local_load_error: str = ""
        if local_ckpt is not None:
            try:
                self._backend = NativeSamBackend(local_ckpt, device=self.device)
                self._backend_kind = "native"
                self.backend = "sam"
                self.model_source = "local_pth" if is_standard else "local_pth_fallback"
                self.model_checkpoint_name = local_ckpt.name
                self.last_load_warning = "" if is_standard else "Using fallback .pth checkpoint from Model Checkpoints."
                return
            except Exception as exc:
                # Local checkpoint is optional; if it cannot be initialized, fallback to HF path.
                local_load_error = str(exc)

        try:
            tf_backend, source = self._build_transformers_backend(self.model_id)
            self._backend = tf_backend
            self._backend_kind = "transformers"
            self.model_source = source
            self.model_checkpoint_name = self.model_id
            if local_load_error:
                self.last_load_warning = (
                    f"Local checkpoint fallback failed ({local_ckpt.name}): {local_load_error}. "
                    "Using HuggingFace model instead."
                )
            else:
                self.last_load_warning = ""
            return
        except Exception as exc:
            if self.backend == "mobile_sam":
                fallback_id = "facebook/sam-vit-base"
                tf_backend, source = self._build_transformers_backend(fallback_id)
                self._backend = tf_backend
                self._backend_kind = "transformers"
                self.backend = "sam"
                self.model_id = fallback_id
                self.model_source = source
                self.model_checkpoint_name = fallback_id
                if local_load_error:
                    self.last_load_warning = (
                        f"Local checkpoint fallback failed ({local_ckpt.name}): {local_load_error}. "
                        "Requested mobile_sam could not be loaded; fell back to sam/facebook-sam-vit-base."
                    )
                else:
                    self.last_load_warning = (
                        "Requested mobile_sam backend could not be loaded. Fell back to sam/facebook-sam-vit-base."
                    )
                return
            raise exc

    def ensure_model(self) -> None:
        if self._backend is not None:
            if self.model_status != "ready":
                self.model_status = "ready"
            return
        with self._model_load_lock:
            if self._backend is not None:
                if self.model_status != "ready":
                    self.model_status = "ready"
                return
            self.model_status = "loading"
            self.model_error = ""
            self.model_loading_started_at = time()
            t0 = perf_counter()
            try:
                self._ensure_model_unlocked()
                self.model_status = "ready"
                self.last_model_load_ms = round((perf_counter() - t0) * 1000.0, 2)
            except Exception as exc:
                self.model_status = "error"
                self.model_error = str(exc)
                self.last_model_load_ms = None
                raise

    def set_image(self, image_path: Path) -> None:
        self.ensure_model()
        if self._backend is None:
            raise RuntimeError("Model backend is unavailable.")
        image_rgb = self._read_image_rgb(image_path)
        self._backend.set_image(image_rgb)
        self._image_path = image_path

    def compute_embedding_for_prefetch(self, image_path: Path) -> SamImageCache | None:
        self.ensure_model()
        if self._backend is None:
            return None
        try:
            image_rgb = self._read_image_rgb(image_path)
            payload = self._backend.compute_embedding(image_rgb)
            return SamImageCache(image_path=image_path, backend_kind=self._backend_kind, payload=payload)
        except Exception:
            return None

    def snapshot_cache(self, to_cpu: bool = True) -> SamImageCache:
        if self._backend is None or self._image_path is None:
            raise RuntimeError("No image cache available. Call set_image first.")
        payload = self._backend.export_embedding(to_cpu=to_cpu)
        return SamImageCache(image_path=self._image_path, backend_kind=self._backend_kind, payload=payload)

    def load_cache(self, cache: SamImageCache) -> None:
        self.ensure_model()
        if self._backend is None:
            raise RuntimeError("Model backend is unavailable.")
        if cache.backend_kind != self._backend_kind:
            raise RuntimeError(
                f"Cache backend mismatch: cache={cache.backend_kind}, active={self._backend_kind}."
            )
        self._backend.load_embedding(cache.payload)
        self._image_path = cache.image_path

    @property
    def image_rgb(self) -> np.ndarray:
        if self._backend is None:
            raise RuntimeError("Image is not set. Call set_image first.")
        return self._backend.image_rgb

    def predict(
        self,
        point_coords: list[list[float]] | None = None,
        point_labels: list[int] | None = None,
        box_xyxy: list[float] | None = None,
        multimask_output: bool = False,
    ) -> SamPrediction:
        if self._backend is None:
            raise RuntimeError("Image embeddings are not cached. Call set_image first.")
        return self._backend.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box_xyxy=box_xyxy,
            mask_input=None,
            multimask_output=multimask_output,
        )
