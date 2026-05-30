import os
os.environ.pop("MPLBACKEND", None)
import matplotlib
matplotlib.use("Agg")

import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scienceplots

plt.style.use(["science", "no-latex"])

root     = Path(__file__).parent.parent / "assets"
gray_dir = root / "gray"
hist_dir = root / "hist"

pattern = re.compile(r"doom_(\w+)_(\d+)_gs\.jpg")

# collect entries: {filter_type: {size: (gray_path, hist_path)}}
groups = defaultdict(dict)
for f in sorted(gray_dir.iterdir()):
    m = pattern.match(f.name)
    if not m:
        continue
    ft, sz = m.group(1), int(m.group(2))
    hist_path = hist_dir / f.name.replace("_gs.jpg", "_hist.csv")
    if hist_path.exists():
        groups[ft][sz] = (f, hist_path)

filter_types = sorted(groups.keys())
sizes        = sorted({s for g in groups.values() for s in g})

# 2 sub-rows per image (image + histogram), one column per kernel size, one section per filter type
fig, axes = plt.subplots(
    len(filter_types) * 2, len(sizes),
    figsize=(len(sizes) * 3, len(filter_types) * 4.5),
)
fig.suptitle("Grayscale images and histograms", fontsize=13, y=1.01)

for fi, ft in enumerate(filter_types):
    for ci, sz in enumerate(sizes):
        img_ax  = axes[fi * 2][ci]
        hist_ax = axes[fi * 2 + 1][ci]

        entry = groups[ft].get(sz)

        # top: grayscale image
        if entry:
            img_ax.imshow(mpimg.imread(entry[0]), cmap="gray")
        if ci == 0:
            img_ax.set_ylabel(ft, fontsize=8)
        if fi == 0:
            img_ax.set_title(f"k={sz}", fontsize=8)
        img_ax.axis("off")

        # bottom: histogram
        if entry:
            data = np.loadtxt(entry[1], delimiter=",", skiprows=1)
            hist_ax.bar(data[:, 0], data[:, 1], width=1, color="green", linewidth=0)
            hist_ax.set_xlim(0, 255)
            hist_ax.tick_params(labelsize=5)
        else:
            hist_ax.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.99])
out = root / "hist_grid.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
