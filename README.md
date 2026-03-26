# SnapSeg

[繁體中文說明](README.zh-TW.md)

Web-based interactive segmentation annotation tool built around SAM.

## Demo

![SnapSeg Demo](docs/demo.gif)

## Features

- Positive/negative point prompts
- Box prompt (drag to segment large objects)
- Mixed prompts (box + points)
- Backend switch: `sam` / `mobile_sam`
- Zoom/pan for tiny targets
- Manual mask editing (brush add/erase + revert to SAM)
- Multi-instance annotation per image
- Undo last confirmed instance (`Backspace`)
- Instance list with per-instance delete
- Manual navigation (`Prev` / `Next` / `Goto`)
- In-UI source picker (folder or image)
- In-UI class configuration
- SAM embedding cache (`set_image` once per image)
- Next-image prefetch queue
- VRAM guard (pause prefetch if free VRAM < 2GB)
- Async save + dirty-state autosave
- Autosave restore on image reload
- Lossless PNG preview frame rendering in Web UI (sharper mask edges)
- Overview Lite page (thumbnail wall + status filters + quick jump)
- Export: COCO + YOLO Segmentation
- Polygon simplification control (`epsilon`)

## Install

Requirements:

- Python 3.10+
- CUDA GPU recommended (CPU supported, slower)

```bash
python -m pip install -r requirements.txt
```

## Model

- Backend runtime: `transformers` SAM stack
- Default `sam` model: `facebook/sam-vit-base`
- Default `mobile_sam` model: `nielsr/slimsam-50-uniform` (Transformers-compatible lightweight SAM)
- Weights download automatically on first run (Hugging Face cache)

## Run

```bash
python main.py
```

Open: `http://127.0.0.1:7861`

Run with MobileSAM backend:

```bash
python main.py --backend mobile_sam
```

Restore flagged status from autosave on startup (optional):

```bash
python main.py --restore-flags
```

Override model id:

```bash
python main.py --backend mobile_sam --model-id <huggingface_model_id>
```

If the selected `mobile_sam` checkpoint is not Transformers-SAM compatible, SnapSeg auto-falls back to `sam` (`facebook/sam-vit-base`) and shows a warning in UI status.

## Basic Workflow

1. Pick source folder/image
2. Set classes (comma-separated)
3. Click `Load Source`
4. Annotate and save

## Controls

- Left click: positive point
- Right click: negative point
- Mouse wheel: zoom
- `Shift + Left drag`: pan
- `B`: toggle box mode
- Box mode + left drag: create box prompt
- `E`: toggle edit mask mode
- `Flag` button: toggle flag for current image
- `T`: revert current mask to SAM prediction
- `[` / `]`: brush radius - / +
- `Enter`: confirm current instance
- `S`: save all confirmed instances for current image
- `Backspace`: undo last confirmed instance
- `U`: undo last point
- `R`: reset current points/mask
- `Space` / `Right`: next image
- `Left`: previous image
- `N` / `P` / `1~9`: switch class

UI tools:

- `Overview` button: open thumbnail wall (`All / Flagged / Labeled / Unseen`) and jump to image.

Note:

- `Enter` only confirms in memory
- `S` writes to disk
- Switching back to an image restores confirmed instances from autosave if present
- Flagged status is not restored by default; use `--restore-flags` to enable it

## Output

Per image:

- `outputs/<run>/<image_stem>/annotations_coco.json`
- `outputs/<run>/<image_stem>/labels_yolo_seg/*.txt`
- `outputs/<run>/<image_stem>/*_mask_*.png`

Autosave:

- `outputs/<run>/autosave/<image_stem>_<image_path_hash>_autosave.json`


## Project Layout

- `main.py` - entry point
- `interactive_web.py` - Web UI + API
- `web/index.html` - frontend page template (HTML/CSS/JS)
- `src/interactive/sam_service.py` - SAM service and embedding cache
- `src/interactive/runtime.py` - prefetch + async save/autosave
- `src/interactive/exporter.py` - COCO/YOLO export

## License

MIT. See [LICENSE](LICENSE).
