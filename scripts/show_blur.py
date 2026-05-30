import os
os.environ.pop("MPLBACKEND", None)
import matplotlib
matplotlib.use("Agg")

import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scienceplots

plt.style.use(["science", "no-latex"])

blur_dir = Path(__file__).parent.parent / "assets" / "blur"
pattern = re.compile(r"doom_(\w+)_(\d+)\.jpg")

groups = defaultdict(dict)
for f in sorted(blur_dir.iterdir()):
    m = pattern.match(f.name)
    if m:
        groups[m.group(1)][int(m.group(2))] = f

filter_types = sorted(groups.keys())
sizes = sorted({s for g in groups.values() for s in g})

rows, cols = len(sizes), len(filter_types)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.4))
fig.suptitle("Blur filtering results", fontsize=13, y=1.00)

for c, ft in enumerate(filter_types):
    for r, sz in enumerate(sizes):
        ax = axes[r][c]
        path = groups[ft].get(sz)
        if path:
            ax.imshow(mpimg.imread(path))
        if r == 0:
            ax.set_title(ft, fontsize=9)
        if c == 0:
            ax.text(-0.05, 0.5, f"k={sz}", transform=ax.transAxes,
                    fontsize=8, va="center", ha="right")
        ax.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.99])
out = Path(__file__).parent.parent / "assets" / "blur_grid.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
