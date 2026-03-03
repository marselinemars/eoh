import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _mode_label_from_log_path(path: Path) -> str:
    # /.../<mode>/results/run_log.jsonl -> mode
    return path.parents[1].name if len(path.parents) >= 2 else path.stem


def _resolve_operator_log(run_log_path: Path) -> Path:
    return run_log_path.parent / "operator_events.jsonl"


def _abbr(label: str) -> str:
    mapping = {
        "IMPROVING": "I",
        "STAGNATING": "S",
        "TOO_MANY_INVALIDS": "T",
        "DEFAULT": "D",
    }
    return mapping.get(label or "DEFAULT", "D")


def _safe_mean(values):
    if not values:
        return None
    return float(np.mean(values))


def plot_fitness(logs_by_mode, outdir: Path):
    plt.figure(figsize=(8, 4))
    for name, rows in logs_by_mode.items():
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


def plot_operator(logs_by_mode, outdir: Path):
    all_ops = sorted({r.get("chosen_operator", "NA") for rows in logs_by_mode.values() for r in rows})
    op_to_y = {op: i for i, op in enumerate(all_ops)}

    plt.figure(figsize=(10, 4.8))
    color_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for idx, (name, rows) in enumerate(logs_by_mode.items()):
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


def plot_invalid_diversity(logs_by_mode, outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for name, rows in logs_by_mode.items():
        xs = [r["gen"] for r in rows]
        invalid_rate = [r.get("invalid_rate") for r in rows]
        diversity = [r.get("diversity") for r in rows]
        axes[0].plot(xs, invalid_rate, marker="o", linewidth=2, label=name)
        axes[1].plot(xs, diversity, marker="o", linewidth=2, label=name)

    axes[0].set_title("Generation Invalid Rate")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Invalid Offspring Rate")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Population Diversity")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Unique Code Hashes")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    out = outdir / "invalid_and_diversity.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("wrote", out)


def _collect_operator_metrics(operator_rows):
    metrics = {}
    for r in operator_rows:
        if not r.get("executed", True):
            continue
        op = r.get("operator", "NA")
        metrics.setdefault(op, {"delta_best": [], "invalid_rate": [], "n": 0})
        metrics[op]["n"] += 1
        delta = r.get("delta_best")
        inv = r.get("invalid_rate")
        if delta is not None:
            metrics[op]["delta_best"].append(float(delta))
        if inv is not None:
            metrics[op]["invalid_rate"].append(float(inv))
    return metrics


def plot_operator_effect(operator_logs_by_mode, outdir: Path):
    all_ops = sorted({r.get("operator", "NA") for rows in operator_logs_by_mode.values() for r in rows if r.get("executed", True)})
    modes = list(operator_logs_by_mode.keys())
    if not all_ops or not modes:
        print("skipping operator effect plot: no operator events")
        return

    x = np.arange(len(all_ops))
    width = 0.8 / max(len(modes), 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, mode in enumerate(modes):
        m = _collect_operator_metrics(operator_logs_by_mode[mode])
        mean_delta = [_safe_mean(m.get(op, {}).get("delta_best", [])) for op in all_ops]
        mean_invalid = [_safe_mean(m.get(op, {}).get("invalid_rate", [])) for op in all_ops]
        offsets = x - 0.4 + (i + 0.5) * width
        axes[0].bar(offsets, [v if v is not None else 0.0 for v in mean_delta], width=width, label=mode)
        axes[1].bar(offsets, [v if v is not None else 0.0 for v in mean_invalid], width=width, label=mode)

    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(all_ops, rotation=20)
    axes[0].set_title("Mean Operator Delta Best (positive is better)")
    axes[0].set_ylabel("mean(best_before - best_after)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(all_ops, rotation=20)
    axes[1].set_title("Mean Operator Invalid Rate")
    axes[1].set_ylabel("invalid_rate")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    out = outdir / "operator_effect_by_mode.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("wrote", out)


def plot_diagnosis_effect(operator_logs_by_mode, outdir: Path):
    routed = operator_logs_by_mode.get("routed", [])
    if not routed:
        print("skipping diagnosis effect plot: routed operator events missing")
        return

    label_order = ["IMPROVING", "STAGNATING", "TOO_MANY_INVALIDS", "DEFAULT"]
    buckets = {k: {"delta_best": [], "invalid_rate": [], "n": 0} for k in label_order}
    for r in routed:
        label = r.get("diagnosis_label", "DEFAULT")
        if label not in buckets:
            buckets[label] = {"delta_best": [], "invalid_rate": [], "n": 0}
        buckets[label]["n"] += 1
        if r.get("delta_best") is not None:
            buckets[label]["delta_best"].append(float(r["delta_best"]))
        if r.get("invalid_rate") is not None:
            buckets[label]["invalid_rate"].append(float(r["invalid_rate"]))

    labels = list(buckets.keys())
    counts = [buckets[l]["n"] for l in labels]
    mean_delta = [_safe_mean(buckets[l]["delta_best"]) or 0.0 for l in labels]
    mean_invalid = [_safe_mean(buckets[l]["invalid_rate"]) or 0.0 for l in labels]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].bar(x, counts, color="tab:blue")
    axes[0].set_title("Diagnosis Counts (Routed)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([_abbr(l) for l in labels])
    axes[0].set_ylabel("count")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, mean_delta, color="tab:green")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("Diagnosis Mean Delta Best")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([_abbr(l) for l in labels])
    axes[1].set_ylabel("mean delta_best")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, mean_invalid, color="tab:red")
    axes[2].set_title("Diagnosis Mean Invalid Rate")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([_abbr(l) for l in labels])
    axes[2].set_ylabel("mean invalid_rate")
    axes[2].grid(axis="y", alpha=0.3)

    out = outdir / "diagnosis_effect_routed.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("wrote", out)


def print_summary(logs_by_mode, operator_logs_by_mode):
    print("\n=== Summary ===")
    for mode, rows in logs_by_mode.items():
        best = rows[-1].get("best_fitness") if rows else None
        inv = _safe_mean([r.get("invalid_rate") for r in rows if r.get("invalid_rate") is not None])
        div = _safe_mean([r.get("diversity") for r in rows if r.get("diversity") is not None])
        print(f"{mode}: gens={len(rows)} final_best={best} mean_invalid={inv} mean_diversity={div}")

        op_metrics = _collect_operator_metrics(operator_logs_by_mode.get(mode, []))
        if not op_metrics:
            print("  operator events: none")
            continue
        for op in sorted(op_metrics.keys()):
            delta = _safe_mean(op_metrics[op]["delta_best"])
            inval = _safe_mean(op_metrics[op]["invalid_rate"])
            n = op_metrics[op]["n"]
            print(f"  op={op}: n={n} mean_delta={delta} mean_invalid={inval}")


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

    logs_by_mode = {}
    operator_logs_by_mode = {}
    for i, log_path in enumerate(args.logs):
        p = Path(log_path).resolve()
        if args.labels and i < len(args.labels):
            label = args.labels[i]
        else:
            label = _mode_label_from_log_path(p)
        logs_by_mode[label] = read_jsonl(p)
        op_path = _resolve_operator_log(p)
        operator_logs_by_mode[label] = read_jsonl(op_path)
        print("loaded", p, "rows:", len(logs_by_mode[label]))
        print("loaded", op_path, "rows:", len(operator_logs_by_mode[label]))

    plot_fitness(logs_by_mode, outdir)
    plot_operator(logs_by_mode, outdir)
    plot_invalid_diversity(logs_by_mode, outdir)
    plot_operator_effect(operator_logs_by_mode, outdir)
    plot_diagnosis_effect(operator_logs_by_mode, outdir)
    print_summary(logs_by_mode, operator_logs_by_mode)


if __name__ == "__main__":
    main()
