from pathlib import Path
import shutil

from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"

TASKS = [
    {
        "task": "detect",
        "weight": "yolo26n.pt",
        "imgsz": 640,
    },
    {
        "task": "classify",
        "weight": "yolo26n-cls.pt",
        "imgsz": 224,
    },
    {
        "task": "pose",
        "weight": "yolo26n-pose.pt",
        "imgsz": 640,
    },
    {
        "task": "segment",
        "weight": "yolo26n-seg.pt",
        "imgsz": 640,
    },
]


def export_task(config: dict[str, object]) -> None:
    task = str(config["task"])
    weight_name = str(config["weight"])
    imgsz = int(config["imgsz"])

    weight_path = MODELS_DIR / weight_name
    if not weight_path.exists():
        print(f"[skip] {task}: missing {weight_path}")
        return

    output_dir = MODELS_DIR / task
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_export_dir = weight_path.with_name(f"{weight_path.stem}_web_model")
    final_dir = output_dir / expected_export_dir.name

    print()
    print(f"[export] task: {task}")
    print(f"  source: {weight_path}")
    print(f"  imgsz: {imgsz}")

    if expected_export_dir.exists():
        shutil.rmtree(expected_export_dir)

    model = YOLO(weight_path)
    exported = Path(model.export(format="tfjs", imgsz=imgsz)).resolve()

    if final_dir.exists():
        shutil.rmtree(final_dir)

    shutil.move(str(exported), str(final_dir))

    print(f"  exported: {exported}")
    print(f"  final: {final_dir}")


def main() -> None:
    print(f"models dir: {MODELS_DIR}")
    for config in TASKS:
        export_task(config)


if __name__ == "__main__":
    main()
