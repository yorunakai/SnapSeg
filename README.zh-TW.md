# SnapSeg

[English README](README.md)

**標得更快，匯出更快。**  
以 SAM 驅動的互動式影像分割標註工具，點擊即可分割，幾秒內匯出 COCO / YOLO。

## Demo

![SnapSeg Demo](docs/demo_v2.gif)

### Overview

![SnapSeg Overview](docs/overview.gif)

## 為什麼選 SnapSeg

- **點擊式分割**：正負點提示 + 即時遮罩預覽
- **為速度而生**：embedding 快取 + 下一張預載，減少等待
- **多類別、多實例**：同一頁面完成複雜場景標註
- **訓練即用**：直接匯出 COCO JSON 或 YOLO Segmentation

## 快速開始

```bash
pip install -r requirements.txt
python main.py
```

打開 `http://127.0.0.1:7861`，選資料來源、設定類別後即可開始標註。

可選啟動參數：

```bash
python main.py --backend mobile_sam
python main.py --restore-flags
python main.py --backend mobile_sam --model-id <huggingface_model_id>
```

## 工作流程

1. 選擇資料夾或單張圖片
2. 輸入類別名稱（逗號分隔）
3. 點擊 **Load Source**
4. 左鍵納入、右鍵排除，`Enter` 確認、`S` 儲存

更多說明：

- [完整快捷鍵與操作](docs/controls.md)
- [專案結構](docs/project-layout.md)

## 輸出

```text
outputs/<run>/<image_stem>/
  annotations_coco.json
  labels_yolo_seg/*.txt
  *_mask_*.png

outputs/<run>/autosave/
  <image_stem>_<image_path_hash>_autosave.json
```

## 需求

- Python 3.10+
- 建議使用 CUDA GPU（CPU 可用但較慢）
- 首次執行會自動從 Hugging Face 下載模型權重

## 授權

MIT，詳見 [LICENSE](LICENSE)。
