"""Build the 1280x640 cover image (GitHub social preview, LinkedIn media) from real results.

Run after `aerorul uncertainty`:

    python scripts/make_cover.py

Reads results/intervals.parquet and results/interval_scores.csv, writes docs/assets/cover.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path("results")
OUT = Path("docs/assets/cover.png")
METHOD, ALPHA, ENGINE = "cv", 0.1, 15

INK, MUTED, ACCENT, BG = "#1B1F24", "#5B6470", "#2F6DB5", "#FFFFFF"

intervals = pd.read_parquet(RESULTS / "intervals.parquet")
scores = pd.read_csv(RESULTS / "interval_scores.csv")

sel = intervals[(intervals["method"] == METHOD) & (intervals["alpha"].round(3) == ALPHA)]
sel = sel[(sel["seed"] == sel["seed"].min()) & (sel["unit"] == ENGINE)].sort_values("cycle")
coverage = scores[(scores["method"] == METHOD) & (scores["target"].round(3) == 1 - ALPHA)][
    "coverage"
].mean()

fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=BG)

# left: text
fig.text(0.05, 0.80, "aero-rul", fontsize=46, fontweight="bold", color=INK)
fig.text(0.05, 0.70, "How many flights does this engine have left,", fontsize=17, color=MUTED)
fig.text(0.05, 0.645, "and how far can that answer be trusted?", fontsize=17, color=MUTED)
fig.text(0.05, 0.43, f"{coverage:.1%}", fontsize=58, fontweight="bold", color=ACCENT)
fig.text(0.05, 0.36, "coverage of a 90 % prediction interval", fontsize=16, color=INK)
fig.text(0.05, 0.315, "on turbofan engines the model never saw", fontsize=16, color=INK)
fig.text(0.05, 0.10, "NASA N-CMAPSS  ·  conformal prediction  ·  pip install aerorul",
         fontsize=13, color=MUTED)  # fmt: skip

# right: one engine's prediction band
ax = fig.add_axes([0.56, 0.17, 0.40, 0.68])
ax.fill_between(sel["cycle"], sel["lo"], sel["hi"], color=ACCENT, alpha=0.18, lw=0,
                label="90 % interval")  # fmt: skip
ax.plot(sel["cycle"], sel["pred"], color=ACCENT, lw=2, label="predicted")
ax.plot(sel["cycle"], sel["RUL"], color=INK, lw=2, ls="--", label="true")
ax.set_title(f"Test engine {ENGINE}", fontsize=14, color=INK, loc="left")
ax.set_xlabel("Flight", fontsize=12, color=MUTED)
ax.set_ylabel("Remaining flights", fontsize=12, color=MUTED)
ax.tick_params(colors=MUTED, labelsize=11)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#C9CED6")
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=11, loc="upper right")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=100, facecolor=BG)
print(f"wrote {OUT}")
