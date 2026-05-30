import os
os.environ.pop("MPLBACKEND", None)  # unset Jupyter backend before matplotlib loads
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path

plt.style.use(["science", "no-latex", "grid"])

assets = Path(__file__).parent.parent / "assets"
data = np.loadtxt(assets / "laplace_result.csv", delimiter=",")

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(
    data,
    origin="lower",       # iy=0 at bottom (math convention)
    cmap="hot",
    interpolation="nearest",
    extent=[0, 1, 0, 1],  # physical domain [0,1]x[0,1]
)
fig.colorbar(im, ax=ax, label="u(x, y)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Laplace equation — Jacobi solution")

out = assets / "laplace_heatmap.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
plt.show()
