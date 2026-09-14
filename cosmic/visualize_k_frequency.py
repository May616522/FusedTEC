"""Plot the frequency distribution of k1/k values in a COSMIC daily CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from scipy.stats import skew


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = (
    PROJECT_ROOT
    / "Results"
    / "Cosmic2021_K_IRI2020"
    / "COSMIC2_K_IRI2020_2021_056.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize COSMIC k1 frequency")
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.csv.with_name(f"{args.csv.stem}_k_frequency.png")

    frame = pd.read_csv(args.csv, encoding="utf-8-sig")
    column = "k" 
    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy()
    if values.size == 0:
        raise SystemExit(f"Column {column!r} contains no valid numeric values")

    # Fixed 0.1 bin width, with limits derived from the actual k1 range.
    bin_start = np.floor(values.min() * 10.0) / 10.0
    bin_end = np.ceil(values.max() * 10.0) / 10.0
    bins = np.arange(bin_start, bin_end + 0.05, 0.1)
    counts, edges = np.histogram(values, bins=bins)
    percentages = counts / values.size * 100.0

    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    chinese_font = FontProperties(fname=font_path) if font_path.exists() else None
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(14, 7), dpi=160)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bars = axis.bar(
        centers,
        counts,
        width=0.088,
        color="#2878B5",
        edgecolor="white",
        linewidth=1.0,
    )
    axis.set_xlim(bin_start, bin_end)
    axis.set_xticks(np.arange(bin_start, bin_end + 0.05, 0.1))
    axis.tick_params(axis="x", labelrotation=45)
    axis.set_xlabel("k1", fontproperties=chinese_font, fontsize=12)
    axis.set_ylabel("频次", fontproperties=chinese_font, fontsize=12)
    axis.set_title(
        "COSMIC-2 2021 年积日 056：k1 频次分布（组距 0.1）",
        fontproperties=chinese_font,
        fontsize=15,
        pad=14,
    )
    axis.bar_label(
        bars,
        labels=[str(value) if value else "" for value in counts],
        padding=3,
        fontsize=8,
    )
    axis.axvline(
        values.mean(), color="#C43C39", linestyle="--", linewidth=1.6,
        label=f"Mean = {values.mean():.3f}",
    )
    axis.axvline(
        np.median(values), color="#E69F00", linestyle=":", linewidth=1.8,
        label=f"Median = {np.median(values):.3f}",
    )
    axis.legend(frameon=False)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.99,
        0.01,
        f"n={values.size}；均值={values.mean():.3f}；中位数={np.median(values):.3f}",
        ha="right",
        fontproperties=chinese_font,
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    quartiles = np.quantile(values, [0.25, 0.5, 0.75])
    mode_index = int(np.argmax(counts))
    print(f"column={column}")
    print(f"n={values.size}")
    print(f"counts={counts.tolist()}")
    print(f"percentages={percentages.round(2).tolist()}")
    print(f"mean={values.mean():.10f}")
    print(f"median={np.median(values):.10f}")
    print(f"std={values.std(ddof=1):.10f}")
    print(f"quartiles={quartiles.tolist()}")
    print(f"min={values.min():.10f}")
    print(f"max={values.max():.10f}")
    print(f"skewness={skew(values, bias=False):.10f}")
    print(f"mode_bin={edges[mode_index]:.1f}-{edges[mode_index + 1]:.1f}")
    print(f"mode_count={counts[mode_index]}")
    print(f"ge_0.6={int((values >= 0.6).sum())},{(values >= 0.6).mean():.10f}")
    print(f"ge_0.7={int((values >= 0.7).sum())},{(values >= 0.7).mean():.10f}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
