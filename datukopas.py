import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Helicopter", "Airplane"],   # UAVs aren't in Open Images — source separately
)

dataset.export(
    export_dir="dataset/combined",
    dataset_type=fo.types.YOLOv5Dataset,
    classes=["Helicopter", "Airplane"],   # order here determines class ID assignment
)