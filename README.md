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

## Workflow

1. Pick source folder or image
2. Set class names (comma-separated)
3. Click **Load Source**
4. Left-click include, right-click exclude, `Enter` confirm, `S` save

More details:

- [Full keyboard shortcuts and controls](docs/controls.md)
- [Project layout](docs/project-layout.md)

## Output

```text
outputs/<run>/<image_stem>/
  annotations_coco.json
  labels_yolo_seg/*.txt
  *_mask_*.png

outputs/<run>/autosave/
  <image_stem>_<image_path_hash>_autosave.json
```

## Requirements

- Python 3.10+
- CUDA GPU recommended (CPU works, slower)
- Model weights auto-downloaded from Hugging Face on first run

## License

MIT. See [LICENSE](LICENSE).
