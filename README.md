# SnapSeg

[繁體中文 README](README.zh-TW.md)

**Annotate faster. Export instantly.**  
Interactive image segmentation powered by SAM - click to segment, export to COCO or YOLO in seconds.

## Demo

![SnapSeg Demo](docs/demo_v2.gif)

### Overview

![SnapSeg Overview](docs/overview.gif)

## Why SnapSeg

- **Point-and-click segmentation** - positive/negative prompts with real-time mask preview
- **Two annotation paths** - use bbox prompt for fast object capture, or brush-only editing for tiny defects
- **Built for speed** - embedding cache + next-image prefetch keeps interaction responsive
- **Multi-instance, multi-class** - annotate complex scenes in one web session
- **Ready to train** - export directly to COCO JSON or YOLO segmentation format

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Open `http://127.0.0.1:7861`, pick a folder, set classes, and start annotating.

Optional run modes:

```bash
python main.py --backend mobile_sam
python main.py --restore-flags
python main.py --backend mobile_sam --model-id <huggingface_model_id>
```

Optional custom checkpoint directory:

```bash
python main.py --checkpoint-dir "C:\path\to\Model Checkpoints"
```

## Workflow

1. Pick source folder or image
2. Set class names (comma-separated)
3. Click **Load Source**
4. Left-click include, right-click exclude, `Enter` confirm, `S` save

### Brush Edit Mode

- Press `E` to toggle brush edit mode for the current mask
- Drag **left mouse** to add mask area, drag **right mouse** to erase
- Use `[` / `]` to decrease/increase brush radius
- Press `T` to revert edited mask back to the SAM prediction

### BBox Or Brush-Only

- Press `B` and drag a box to segment by bbox prompt
- For tiny defects, you can skip bbox/points and use brush mode directly to paint the target area

More details:

- [Full keyboard shortcuts and controls](docs/controls.md)
- [Project layout](docs/project-layout.md)

## Output

```text
outputs/<run>/<image_stem>/
  annotations_coco.json
  labels_yolo_seg/*.txt
  labels_yolo_bbox/*.txt
  classes_yolo_bbox.txt
  *_mask_*.png

outputs/<run>/autosave/
  <image_stem>_<image_path_hash>_autosave.json

outputs/<run>/dataset/
  train/images/*
  train/labels/*
  val/images/        (created)
  val/labels/        (created)
  test/images/       (created)
  test/labels/       (created)
  classes.txt
  dataset.yaml
```

`dataset.yaml` is auto-generated in this format:

```yaml
train: <abs_path>/train/images
val:   <abs_path>/val/images
test:  <abs_path>/test/images

nc: <num_classes>
names: [class1, class2, ...]
```

## Requirements

- Python 3.10+
- CUDA GPU recommended (CPU works, slower)
- `torch`, `torchvision`, `transformers`, `fastapi`, `uvicorn`, `opencv-python`, `numpy`, `pillow`
- `segment-anything` (for local `.pth` checkpoint mode)
- Installing dependencies does **not** include pretrained SAM checkpoint files.

## Model Checkpoints

SnapSeg supports two model startup paths:

1. **Local `.pth` first**: if a checkpoint exists in `Model Checkpoints/`, SnapSeg will try that first.
2. **HF fallback**: if local checkpoint is missing or fails to initialize, SnapSeg falls back to `facebook/sam-vit-base` from Hugging Face (cache first, then download if needed).

Place local checkpoints here:

```text
Snapseg/Model Checkpoints/
  sam_vit_b_01ec64.pth
```

Checkpoint download (official SAM):

- `vit_b`: [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

Users are responsible for downloading and using checkpoints in compliance with the original model licenses and terms.

## Third-Party Models & Dependencies

SnapSeg provides the annotation workflow and web UI layer.  
SAM implementations, model checkpoints, and external model hosting (for example, Hugging Face models such as `facebook/sam-vit-base`) are third-party dependencies/assets and are provided under their respective licenses and terms.

## License

MIT applies to the SnapSeg source code in this repository.  
Third-party dependencies and model assets are subject to their own licenses and terms. See [LICENSE](LICENSE).
