import shutil
import tempfile
from pathlib import Path

from src.interactive.dataset_packager import DatasetPackager


def test_update_class_metadata_normalizes_and_writes() -> None:
    base_tmp = Path.cwd() / ".tmp_tests"
    base_tmp.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="packager_meta_", dir=str(base_tmp)))
    try:
        packager = DatasetPackager(workdir / "dataset")
        classes = packager.update_class_metadata([" scratch ", "particle", "scratch", " "])

        assert classes == ["scratch", "particle"]
        assert packager.classes_path.read_text(encoding="utf-8").splitlines() == ["scratch", "particle"]

        yaml_text = packager.yaml_path.read_text(encoding="utf-8")
        assert "nc: 2" in yaml_text
        assert "names: [scratch, particle]" in yaml_text
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_package_yolo_seg_copies_and_remaps_labels() -> None:
    base_tmp = Path.cwd() / ".tmp_tests"
    base_tmp.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="packager_labels_", dir=str(base_tmp)))
    try:
        image_path = workdir / "img1.jpg"
        image_path.write_bytes(b"dummy-image")

        image_out = workdir / "outputs" / "img1"
        labels_dir = image_out / "labels_yolo_seg"
        labels_dir.mkdir(parents=True, exist_ok=True)

        (labels_dir / "img1.txt").write_text(
            "0 0.1 0.2 0.3 0.4 0.5 0.6\n1 0.2 0.3 0.4 0.5 0.6 0.7\n",
            encoding="utf-8",
        )
        (image_out / "classes_yolo_seg.txt").write_text("scratch\nparticle\n", encoding="utf-8")

        packager = DatasetPackager(workdir / "dataset")
        packager.package_yolo_seg(image_path=image_path, image_out=image_out, class_list=["scratch", "particle"])

        copied_images = list(packager.train_images_dir.glob("img1_*.jpg"))
        copied_labels = list(packager.train_labels_dir.glob("img1_*.txt"))

        assert len(copied_images) == 1
        assert len(copied_labels) == 1

        label_lines = copied_labels[0].read_text(encoding="utf-8").splitlines()
        assert label_lines[0].startswith("0 ")
        assert label_lines[1].startswith("1 ")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
