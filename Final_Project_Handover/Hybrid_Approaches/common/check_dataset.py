import sys
from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(COMMON_DIR))

from config import DATASET_70, DATASET_80


def count_files(folder, extensions):
    folder = Path(folder)

    if not folder.exists():
        return 0

    return sum(
        1
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )


def get_dataset_root(yaml_path):
    """
    Read the 'path:' entry from the YOLO dataset YAML.

    Supports both:
    - absolute paths
    - paths relative to the YAML file
    """

    yaml_path = Path(yaml_path)
    yaml_text = yaml_path.read_text()

    for line in yaml_text.splitlines():

        line = line.strip()

        if line.startswith("path:"):
            path_value = line.split(":", 1)[1].strip()

            # Remove quotes if present
            path_value = path_value.strip("\"'")

            dataset_root = Path(path_value)

            # If path is relative, make it relative to YAML location
            if not dataset_root.is_absolute():
                dataset_root = yaml_path.parent / dataset_root

            return dataset_root.resolve()

    # Fallback if no path: entry exists
    return yaml_path.parent.resolve()


def check_split(dataset_name, yaml_path):

    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name}")
    print("=" * 70)

    yaml_path = Path(yaml_path)

    print("YAML:", yaml_path)

    if not yaml_path.exists():
        print("ERROR: YAML file not found.")
        return

    print("YAML exists: YES")

    # Read YAML manually to avoid changing project dependencies
    yaml_text = yaml_path.read_text()

    print("\nYAML contents:")
    print("-" * 50)
    print(yaml_text)
    print("-" * 50)

    # Correctly read dataset root from YAML
    dataset_root = get_dataset_root(yaml_path)

    print("\nDataset root:", dataset_root)

    if dataset_root.exists():
        print("Dataset root exists: YES")
    else:
        print("Dataset root exists: NO")

    image_ext = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    label_ext = {".txt"}

    for split in ["train", "val", "test"]:

        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split

        image_count = count_files(image_dir, image_ext)
        label_count = count_files(label_dir, label_ext)

        print(f"\n{split.upper()}")
        print("Images:", image_count)
        print("Labels:", label_count)

        if image_count == label_count:
            print("Status: OK")
        else:
            print("Status: WARNING - image/label count mismatch")


def main():

    print("=" * 70)
    print("SUGARBEET HYBRID APPROACH - DATASET CHECK")
    print("=" * 70)

    check_split("70:20:10", DATASET_70)
    check_split("80:10:10", DATASET_80)

    print("\n" + "=" * 70)
    print("DATASET CHECK COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()