import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def plot_fitness(logs, outdir: Path):
    plt.figure(figsize=(8, 4))
    for name, rows in logs.items():
        xs = [r["gen"] for r in rows]
        ys = [r.get("best_fitness") for r in rows]
        plt.plot(xs, ys, marker="o", linewidth=2, label=name)
        if xs and ys:
            plt.text(xs[-1], ys[-1], f" {name}", fontsize=9, va="center")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (lower is better)")
    plt.title("Fitness vs Generation")
    plt.grid(alpha=0.3)
    plt.legend()
    out = outdir / "fitness_vs_gen.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("wrote", out)


def _abbr(label: str) -> str:
    mapping = {
        "IMPROVING": "I",
        "STAGNATING": "S",
        "TOO_MANY_INVALIDS": "T",
        "DEFAULT": "D",
    }
    return mapping.get(label or "DEFAULT", "D")


def plot_operator(logs, outdir: Path):
    all_ops = sorted({r.get("chosen_operator", "NA") for rows in logs.values() for r in rows})
    op_to_y = {op: i for i, op in enumerate(all_ops)}

    plt.figure(figsize=(10, 4.8))
    color_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for idx, (name, rows) in enumerate(logs.items()):
        xs = [r["gen"] for r in rows]
        ys = [op_to_y.get(r.get("chosen_operator", "NA"), -1) for r in rows]
        color = color_cycle[idx % len(color_cycle)]
        plt.plot(xs, ys, marker="o", linestyle="-", color=color, label=name)
        for x, y, r in zip(xs, ys, rows):
            lbl = _abbr(r.get("diagnosis_label", "DEFAULT"))
            plt.text(x, y + 0.08, lbl, fontsize=8, ha="center", color=color)

    plt.yticks(list(op_to_y.values()), list(op_to_y.keys()))
    plt.xlabel("Generation")
    plt.ylabel("Chosen Operator")
    plt.title("Operator Over Time (Point Labels: Diagnosis I/S/T/D)")
    plt.grid(alpha=0.3)
    plt.legend()
    out = outdir / "operator_over_time.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("wrote", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs",
        nargs="+",
        default=[
            "./compare_runs/baseline/results/run_log.jsonl",
            "./compare_runs/routed/results/run_log.jsonl",
        ],
        help="One or more run_log.jsonl files",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels aligned to --logs",
    )
    parser.add_argument("--outdir", default="./compare_runs/plots", help="Output directory for figures")
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    logs = {}
    for i, log_path in enumerate(args.logs):
        p = Path(log_path).resolve()
        if args.labels and i < len(args.labels):
            label = args.labels[i]
        else:
            # /.../<mode>/results/run_log.jsonl -> mode name
            label = p.parents[1].name if len(p.parents) >= 2 else p.stem
        logs[label] = read_jsonl(p)
        print("loaded", p, "rows:", len(logs[label]))

    plot_fitness(logs, outdir)
    plot_operator(logs, outdir)


if __name__ == "__main__":
    main()
