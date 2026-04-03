import shutil
import tempfile
from pathlib import Path

import numpy as np

from src.interactive.exporter import AnnotationExporter, MaskAnnotation


def test_bbox_and_sanitize_behaviors() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:8, 2:6] = 1

    bbox = AnnotationExporter._bbox_xywh(mask)
    assert bbox == [2.0, 3.0, 4.0, 5.0]

    assert AnnotationExporter._sanitize_bbox_xywh([1, 2, 3, 4]) == [1.0, 2.0, 3.0, 4.0]
    assert AnnotationExporter._sanitize_bbox_xywh([1, 2, 0, 4]) is None
    assert AnnotationExporter._sanitize_bbox_xywh(None) is None


def test_export_coco_and_yolo_outputs() -> None:
    base_tmp = Path.cwd() / ".tmp_tests"
    base_tmp.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="exporter_", dir=str(base_tmp)))
    try:
        image_path = workdir / "sample.jpg"
        image_path.write_bytes(b"dummy")

        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:12, 5:11] = 1

        ann = MaskAnnotation(image_path=image_path, category_name="scratch", mask=mask, score=0.9)
        exp = AnnotationExporter(polygon_epsilon_ratio=0.01)

        coco_path = workdir / "out" / "annotations_coco.json"
        exp.export_coco([ann], coco_path)

        coco_text = coco_path.read_text(encoding="utf-8")
        assert '"file_name": "sample.jpg"' in coco_text
        assert '"name": "scratch"' in coco_text
        assert '"segmentation":' in coco_text

        labels_dir = workdir / "out" / "labels_yolo_seg"
        classes_path = workdir / "out" / "classes_yolo_seg.txt"
        exp.export_yolo_seg([ann], labels_dir, classes_path)

        label_file = labels_dir / "sample.txt"
        assert label_file.exists()
        first_line = label_file.read_text(encoding="utf-8").strip().splitlines()[0]
        assert first_line.startswith("0 ")
        assert len(first_line.split()) >= 7

        assert classes_path.read_text(encoding="utf-8").strip() == "scratch"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
