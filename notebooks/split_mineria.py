"""Script to move mineria images and labels into data/Imagenes/train and data/Etiquetas/train.

The notebook exploracion_datos.ipynb handles unification and strategic splitting,
so this script only needs to place the files into data/ for the pipeline to pick them up.
"""

import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

SOURCE_IMAGES = BASE_DIR / "mineria" / "frames"
SOURCE_LABELS = BASE_DIR / "mineria" / "labels"

DEST_IMAGES = BASE_DIR / "data" / "Imagenes" / "train"
DEST_LABELS = BASE_DIR / "data" / "Etiquetas" / "train"


def main() -> None:
    """Move all mineria images and labels into data/Imagenes/train and data/Etiquetas/train."""
    DEST_IMAGES.mkdir(parents=True, exist_ok=True)
    DEST_LABELS.mkdir(parents=True, exist_ok=True)

    images = list(SOURCE_IMAGES.glob("*.jpg"))
    if not images:
        print("No images found in mineria/frames/")
        return

    moved = 0
    for img_path in sorted(images):
        stem = img_path.stem
        src_lbl = SOURCE_LABELS / f"{stem}.txt"

        shutil.move(str(img_path), str(DEST_IMAGES / img_path.name))
        if src_lbl.exists():
            shutil.move(str(src_lbl), str(DEST_LABELS / src_lbl.name))
        moved += 1

    print(f"Moved {moved} images and labels to data/Imagenes/train and data/Etiquetas/train.")
    print("Run exploracion_datos.ipynb to unify and split the dataset.")


if __name__ == "__main__":
    main()
