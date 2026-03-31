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
- **雙標註路徑**：可用 bbox 框選快速抓目標，也可純畫筆處理細小瑕疵
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

可選：自訂本地權重目錄

```bash
python main.py --checkpoint-dir "C:\path\to\Model Checkpoints"
```

## 工作流程

1. 選擇資料夾或單張圖片
2. 輸入類別名稱（逗號分隔）
3. 點擊 **Load Source**
4. 左鍵納入、右鍵排除，`Enter` 確認、`S` 儲存

### 畫筆修邊模式

- 按 `E` 切換到畫筆修邊模式（針對目前遮罩）
- **左鍵拖曳**新增遮罩，**右鍵拖曳**擦除遮罩
- 用 `[` / `]` 縮小/放大畫筆半徑
- 按 `T` 可還原為 SAM 原始預測遮罩

### BBox 或純畫筆

- 按 `B` 後拖曳方框，可用 bbox prompt 快速分割
- 對於細小瑕疵，可不使用 bbox/點提示，直接用畫筆模式塗出目標區域

更多說明：

- [完整快捷鍵與操作](docs/controls.md)
- [專案結構](docs/project-layout.md)

## 輸出

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
  val/images/        (自動建立)
  val/labels/        (自動建立)
  test/images/       (自動建立)
  test/labels/       (自動建立)
  classes.txt
  dataset.yaml
```

`dataset.yaml` 會自動生成為：

```yaml
train: <abs_path>/train/images
val:   <abs_path>/val/images
test:  <abs_path>/test/images

nc: <num_classes>
names: [class1, class2, ...]
```

## 需求

- Python 3.10+
- 建議使用 CUDA GPU（CPU 可用但較慢）
- `torch`、`torchvision`、`transformers`、`fastapi`、`uvicorn`、`opencv-python`、`numpy`、`pillow`
- `segment-anything`（本地 `.pth` 權重模式需要）
- 安裝上述依賴**不包含**預訓練 SAM 權重檔案。

## Model Checkpoints（本地權重）

SnapSeg 支援兩種模型啟動方式：

1. **優先本地 `.pth`**：若 `Model Checkpoints/` 有權重，會先嘗試本地載入。
2. **HF 後備**：若本地權重不存在或初始化失敗，會改用 Hugging Face 的 `facebook/sam-vit-base`（先用快取，沒有才下載）。

請把本地權重放在：

```text
Snapseg/Model Checkpoints/
  sam_vit_b_01ec64.pth
```

官方權重下載連結：

- `vit_b`：[ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

使用者需自行確認並遵守原始模型/權重的授權與使用條款後再下載與使用。

## 第三方模型與依賴聲明

SnapSeg 提供的是標註流程與 Web UI 工具層。  
SAM 實作、模型權重與外部模型託管來源（例如 Hugging Face 的 `facebook/sam-vit-base`）皆屬第三方依賴/資產，並受其各自授權與使用條款約束。

## 授權

MIT 僅適用於本儲存庫中的 SnapSeg 原始碼。  
第三方依賴與模型資產適用其各自授權與條款，詳見 [LICENSE](LICENSE)。
