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
        plt.plot(xs, ys, marker="o", label=name)
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


def plot_operator(logs, outdir: Path):
    all_ops = sorted({r.get("chosen_operator", "NA") for rows in logs.values() for r in rows})
    op_to_y = {op: i for i, op in enumerate(all_ops)}

    plt.figure(figsize=(9, 4))
    for name, rows in logs.items():
        xs = [r["gen"] for r in rows]
        ys = [op_to_y.get(r.get("chosen_operator", "NA"), -1) for r in rows]
        plt.plot(xs, ys, marker="o", linestyle="-", label=name)
    plt.yticks(list(op_to_y.values()), list(op_to_y.keys()))
    plt.xlabel("Generation")
    plt.ylabel("Chosen Operator")
    plt.title("Operator Over Time")
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
        label = args.labels[i] if args.labels and i < len(args.labels) else p.parent.parent.parent.name
        logs[label] = read_jsonl(p)
        print("loaded", p, "rows:", len(logs[label]))

    plot_fitness(logs, outdir)
    plot_operator(logs, outdir)


if __name__ == "__main__":
    main()
