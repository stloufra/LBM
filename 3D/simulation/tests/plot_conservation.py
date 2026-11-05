#!/usr/bin/env python3
"""Plot conservation test histories written by `conservation --dump DIR`."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASE_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def read_history(path):
    meta, rows = {}, []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                key, value = line[1:].split(":", 1)
                meta[key.strip()] = value.strip()
            else:
                rows.append(line)
    table = list(csv.DictReader(rows))
    columns = {key: [float(r[key]) for r in table] for key in table[0]}
    return meta["name"], meta["open"] == "1", columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--baseline", type=Path, help="dump dir of a reference build, drawn dashed")
    parser.add_argument("--labels", nargs=2, default=["fixed", "baseline (main)"], help="solid and dashed line labels")
    parser.add_argument("-o", "--output", type=Path, default=Path("conservation.pdf"))
    args = parser.parse_args()

    paths = sorted(args.dump_dir.glob("*.csv"))
    if not paths:
        raise SystemExit(f"No CSV histories in {args.dump_dir}")

    plt.rcParams.update({"font.family": "serif", "savefig.transparent": True})
    fig, axes = plt.subplots(3, 2, sharex="col", figsize=(10, 8), layout="constrained")
    axes[0, 0].set_title("closed domains")
    axes[0, 1].set_title("open channel")

    for colour, path in zip(CASE_COLOURS, paths):
        runs = [(path, "-")]
        if args.baseline and (args.baseline / path.name).exists():
            runs.append((args.baseline / path.name, "--"))
        for run_path, style in runs:
            name, is_open, h = read_history(run_path)
            column = axes[:, 1 if is_open else 0]
            label = name if style == "-" else None
            # Closed: mass must be conserved, show magnitude. Open: inflow/outflow imbalance, show sign.
            drift = h["mass_drift"] if is_open else [abs(d) for d in h["mass_drift"]]
            column[0].plot(h["step"][1:], drift[1:], style, color=colour, lw=1.5, label=label)
            column[1].plot(h["step"][1:], h["max_density_deviation"][1:], style, color=colour, lw=1.5)
            column[2].plot(h["step"], h["periodic_mismatch"], style, color=colour, lw=1.5)

    axes[0, 0].set_yscale("log")
    axes[1, 0].set_yscale("log")
    # Symlog keeps exact zeros (fixed periodic mismatch) visible.
    for ax in axes[2]:
        ax.set_yscale("symlog", linthresh=1e-9)
        ax.set_ylim(bottom=-3e-10)
    axes[0, 0].set_ylabel("|relative mass drift|")
    axes[0, 1].set_ylabel("relative mass change")
    axes[1, 0].set_ylabel(r"max $|\rho - 1|$")
    axes[1, 1].set_ylabel(r"max $|\rho - 1|$")
    axes[2, 0].set_ylabel("periodic plane mismatch")
    axes[2, 1].set_ylabel("periodic plane mismatch")
    for ax in axes.flat:
        ax.grid(True, which="major", color="0.85", lw=0.5)
        ax.set_facecolor("none")
    for ax in axes[-1]:
        ax.set_xlabel("time step")

    handles, labels = [], []
    for ax in axes[0]:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    if args.baseline:
        handles += [plt.Line2D([], [], color="0.3", ls="-"), plt.Line2D([], [], color="0.3", ls="--")]
        labels += args.labels
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False, fontsize="small")
    fig.savefig(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
