# SnapSeg

[English README](README.md)

基於 SAM 的 Web 互動式分割標註工具。

## Demo

![SnapSeg Demo](docs/demo.gif)

## 功能

- 正負點提示分割
- 框選提示（拖曳方框分割大型目標）
- 混合提示（box + points）
- 後端可切換：`sam` / `mobile_sam`
- 縮放/平移，適合小目標
- 單張圖多 instance 標註
- 支援反悔（`Backspace` 撤回最後一筆已確認 instance）
- 已確認 instance 清單，可逐筆刪除
- 手動翻頁（`Prev` / `Next` / `Goto`）
- 介面內選擇資料來源（資料夾或單張圖片）
- 介面內設定標註類別
- SAM embedding 快取（每張圖只做一次 `set_image`）
- 下一張預載（prefetch queue）
- 顯存保護（可用 VRAM < 2GB 時暫停預載）
- 非同步存檔 + dirty-state autosave
- 重新載入圖片時自動還原 autosave 標註
- 匯出 COCO + YOLO Segmentation
- `epsilon` 輪廓簡化控制

## 安裝

需求：

- Python 3.10+
- 建議使用 CUDA GPU（CPU 可跑但較慢）

```bash
python -m pip install -r requirements.txt
```

## 模型

- 後端執行：`transformers` SAM 堆疊
- `sam` 預設模型：`facebook/sam-vit-base`
- `mobile_sam` 預設模型：`nielsr/slimsam-50-uniform`（Transformers 相容的輕量 SAM）
- 首次執行自動下載權重（Hugging Face cache）

## 啟動

```bash
python main.py
```

開啟：`http://127.0.0.1:7861`

使用 MobileSAM 後端：

```bash
python main.py --backend mobile_sam
```

覆蓋模型 ID：

```bash
python main.py --backend mobile_sam --model-id <huggingface_model_id>
```

若選到的 `mobile_sam` 權重不相容於 Transformers SAM，SnapSeg 會自動降級到 `sam`（`facebook/sam-vit-base`），並在 UI 狀態欄顯示警告。

## 基本流程

1. 選資料夾或圖片
2. 輸入類別（逗號分隔）
3. 點 `Load Source`
4. 開始標註並存檔

## 操作鍵

- 左鍵：正點
- 右鍵：負點
- 滾輪：縮放
- `Shift + 左鍵拖曳`：平移
- `B`：切換 Box Mode
- Box Mode + 左鍵拖曳：建立框選提示
- `Enter`：確認目前 instance
- `S`：儲存目前圖片所有已確認 instance
- `Backspace`：撤回最後一筆已確認 instance
- `U`：復原上一個點
- `R`：重置目前點與暫時遮罩
- `Space` / `Right`：下一張
- `Left`：上一張
- `N` / `P` / `1~9`：切換類別

說明：

- `Enter` 只做記憶體確認
- `S` 才會寫入磁碟
- 重新切回同一張圖片時，若有 autosave 會自動還原已確認 instance

## 輸出

每張圖：

- `outputs/<run>/<image_stem>/annotations_coco.json`
- `outputs/<run>/<image_stem>/labels_yolo_seg/*.txt`
- `outputs/<run>/<image_stem>/*_mask_*.png`

暫存：

- `outputs/<run>/autosave/<image_stem>_autosave.json`

## 專案結構

- `main.py` - 啟動入口
- `interactive_web.py` - Web UI + API
- `src/interactive/sam_service.py` - SAM 服務與 embedding 快取
- `src/interactive/runtime.py` - 預載 + 非同步存檔/暫存
- `src/interactive/exporter.py` - COCO/YOLO 匯出

## 授權

MIT，詳見 [LICENSE](LICENSE)。
