"""Create a tiny synthetic dataset (40 items) to check the pipeline runs end to end.

The texts and images are random. Any accuracy on this data is meaningless.
Output: demo_data/demo.csv and demo_data/images/*.png
"""
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path("demo_data")
REAL = ["Council confirms new bus timetable from Monday", "University publishes annual research report",
        "Met Office issues routine weather update", "Hospital opens extended clinic hours"]
FAKE = ["Scientists confirm the moon is made of cheese", "Miracle pill cures every disease overnight",
        "Celebrity secretly replaced by robot, insiders say", "Drinking seawater boosts IQ by 50 points"]


def main():
    rng = np.random.default_rng(42)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rows = ["text,image_path,label"]
    for i in range(40):
        label = i % 2
        text = (FAKE if label else REAL)[i % 4]
        img = (rng.random((64, 64, 3)) * 255).astype("uint8")
        name = f"images/item_{i:02d}.png"
        Image.fromarray(img).save(OUT / name)
        rows.append(f'"{text}",{name},{label}')
    (OUT / "demo.csv").write_text("\n".join(rows) + "\n")
    print(f"Wrote {len(rows) - 1} demo items to {OUT}/")


if __name__ == "__main__":
    main()
