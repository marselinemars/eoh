import argparse
import json
from collections import Counter
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


def _safe_std(values):
    if not values:
        return None
    return float(np.std(values))


def _mode_label_from_log_path(path: Path) -> str:
    # /.../<mode>/results/run_log.jsonl -> mode
    return path.parents[1].name if len(path.parents) >= 2 else path.stem


def _resolve_operator_log(run_log_path: Path) -> Path:
    return run_log_path.parent / "operator_events.jsonl"


def _parse_seed_from_path(path: Path):
    for part in path.parts:
        if part.startswith("seed_"):
            try:
                return int(part.split("seed_", 1)[1])
            except Exception:
                return None
    return None


def discover_runs(root: Path):
    runs = []
    seed_dirs = sorted([p for p in root.glob("seed_*") if p.is_dir()])

    if seed_dirs:
        for seed_dir in seed_dirs:
            seed = _parse_seed_from_path(seed_dir)
            for mode in ["baseline", "routed"]:
                run_log = seed_dir / mode / "results" / "run_log.jsonl"
                if not run_log.exists():
                    continue
                operator_log = _resolve_operator_log(run_log)
                runs.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "label": f"{mode}-seed{seed}",
                        "run_log_path": run_log,
                        "operator_log_path": operator_log,
                        "rows": read_jsonl(run_log),
                        "operator_rows": read_jsonl(operator_log),
                    }
                )
    else:
        for mode in ["baseline", "routed"]:
            run_log = root / mode / "results" / "run_log.jsonl"
            if not run_log.exists():
                continue
            operator_log = _resolve_operator_log(run_log)
            runs.append(
                {
                    "mode": mode,
                    "seed": None,
                    "label": mode,
                    "run_log_path": run_log,
                    "operator_log_path": operator_log,
                    "rows": read_jsonl(run_log),
                    "operator_rows": read_jsonl(operator_log),
                }
            )
    return runs


def load_runs_from_logs(log_paths, labels=None):
    runs = []
    for i, log_path in enumerate(log_paths):
        run_log = Path(log_path).resolve()
        mode = _mode_label_from_log_path(run_log)
        label = labels[i] if labels and i < len(labels) else mode
        operator_log = _resolve_operator_log(run_log)
        runs.append(
            {
                "mode": mode,
                "seed": _parse_seed_from_path(run_log),
                "label": label,
                "run_log_path": run_log,
                "operator_log_path": operator_log,
                "rows": read_jsonl(run_log),
                "operator_rows": read_jsonl(operator_log),
            }
        )
    return runs


def group_runs_by_mode(runs):
    grouped = {}
    for run in runs:
        grouped.setdefault(run["mode"], []).append(run)
    return grouped


def _values_by_gen(runs, key):
    data = {}
    for run in runs:
        for r in run["rows"]:
            gen = r.get("gen")
            val = r.get(key)
            if gen is None or val is None:
                continue
            try:
                data.setdefault(int(gen), []).append(float(val))
            except (TypeError, ValueError):
                continue
    return data


def _mean_std_by_gen(values_by_gen):
    if not values_by_gen:
        return [], [], []
    xs = sorted(values_by_gen.keys())
    means = [float(np.mean(values_by_gen[x])) for x in xs]
    stds = [float(np.std(values_by_gen[x])) for x in xs]
    return xs, means, stds


def plot_fitness(runs_by_mode, outdir: Path):
    plt.figure(figsize=(9, 4.5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for idx, (mode, runs) in enumerate(sorted(runs_by_mode.items())):
        color = colors[idx % len(colors)]
        for run in runs:
            xs = [r["gen"] for r in run["rows"]]
            ys = [r.get("best_fitness") for r in run["rows"]]
            if xs and ys:
                plt.plot(xs, ys, color=color, alpha=0.22, linewidth=1.2)

        x, mean, std = _mean_std_by_gen(_values_by_gen(runs, "best_fitness"))
        if not x:
            continue
        plt.plot(x, mean, color=color, marker="o", linewidth=2.4, label=f"{mode} mean")
        low = [m - s for m, s in zip(mean, std)]
        high = [m + s for m, s in zip(mean, std)]
        plt.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
        plt.text(x[-1], mean[-1], f" {mode}", fontsize=9, va="center", color=color)

    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (lower is better)")
    plt.title("Fitness vs Generation (mean +/- std)")
    plt.grid(alpha=0.3)
    plt.legend()
    out = outdir / "fitness_vs_gen.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("wrote", out)


def _majority_by_gen(runs, key):
    out = {}
    for run in runs:
        for r in run["rows"]:
            gen = r.get("gen")
            value = r.get(key)
            if gen is None or value is None:
                continue
            out.setdefault(int(gen), []).append(value)
    result = {}
    for gen, values in out.items():
        c = Counter(values)
        best, best_count = c.most_common(1)[0]
        share = best_count / len(values)
        result[gen] = (best, share)
    return result


def plot_operator(runs_by_mode, outdir: Path):
    all_ops = sorted(
        {
            r.get("chosen_operator", "NA")
            for runs in runs_by_mode.values()
            for run in runs
            for r in run["rows"]
        }
    )
    if not all_ops:
        print("skipping operator plot: no run logs")
        return
    op_to_y = {op: i for i, op in enumerate(all_ops)}

    plt.figure(figsize=(10.2, 5.0))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for idx, (mode, runs) in enumerate(sorted(runs_by_mode.items())):
        color = colors[idx % len(colors)]
        op_major = _majority_by_gen(runs, "chosen_operator")
        dg_major = _majority_by_gen(runs, "diagnosis_label")
        gens = sorted(op_major.keys())
        if not gens:
            continue
        ys = [op_to_y.get(op_major[g][0], -1) for g in gens]
        plt.plot(gens, ys, marker="o", linewidth=2.2, color=color, label=mode)
        for g, y in zip(gens, ys):
            diag = dg_major.get(g, ("DEFAULT", 0.0))[0]
            share = op_major[g][1]
            plt.text(g, y + 0.08, f"{_abbr(diag)} {share:.0%}", fontsize=8, ha="center", color=color)

    plt.yticks(list(op_to_y.values()), list(op_to_y.keys()))
    plt.xlabel("Generation")
    plt.ylabel("Chosen Operator (majority across seeds)")
    plt.title("Operator Over Time (Point Labels: Diagnosis + Majority Share)")
    plt.grid(alpha=0.3)
    plt.legend()
    out = outdir / "operator_over_time.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("wrote", out)


def plot_invalid_diversity(runs_by_mode, outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for idx, (mode, runs) in enumerate(sorted(runs_by_mode.items())):
        color = colors[idx % len(colors)]

        x_inv, mean_inv, std_inv = _mean_std_by_gen(_values_by_gen(runs, "invalid_rate"))
        x_div, mean_div, std_div = _mean_std_by_gen(_values_by_gen(runs, "diversity"))

        if x_inv:
            axes[0].plot(x_inv, mean_inv, marker="o", linewidth=2.2, color=color, label=mode)
            axes[0].fill_between(
                x_inv,
                [m - s for m, s in zip(mean_inv, std_inv)],
                [m + s for m, s in zip(mean_inv, std_inv)],
                color=color,
                alpha=0.15,
                linewidth=0,
            )
        if x_div:
            axes[1].plot(x_div, mean_div, marker="o", linewidth=2.2, color=color, label=mode)
            axes[1].fill_between(
                x_div,
                [m - s for m, s in zip(mean_div, std_div)],
                [m + s for m, s in zip(mean_div, std_div)],
                color=color,
                alpha=0.15,
                linewidth=0,
            )

    axes[0].set_title("Generation Invalid Rate (mean +/- std)")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Invalid Offspring Rate")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Population Diversity (mean +/- std)")
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
    all_ops = sorted(
        {
            r.get("operator", "NA")
            for rows in operator_logs_by_mode.values()
            for r in rows
            if r.get("executed", True)
        }
    )
    modes = sorted(list(operator_logs_by_mode.keys()))
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

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))
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


def print_summary(runs_by_mode, operator_logs_by_mode):
    print("\n=== Summary ===")
    for mode in sorted(runs_by_mode.keys()):
        runs = runs_by_mode[mode]
        final_bests = []
        for run in runs:
            if run["rows"]:
                final_bests.append(run["rows"][-1].get("best_fitness"))
        final_bests = [float(v) for v in final_bests if v is not None]
        mean_best = _safe_mean(final_bests)
        std_best = _safe_std(final_bests)
        print(
            f"{mode}: n_runs={len(runs)} final_best_mean={mean_best} final_best_std={std_best} "
            f"runs_with_best={len(final_bests)}"
        )

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
        "--root",
        default="./compare_runs",
        help="Root directory for auto-discovery (supports single-seed and seed_*/ layout).",
    )
    parser.add_argument(
        "--logs",
        nargs="*",
        default=None,
        help="Optional explicit run_log.jsonl paths. If set, auto-discovery is skipped.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels aligned to --logs",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for figures. Default: <root>/plots",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else (root / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.logs:
        runs = load_runs_from_logs(args.logs, args.labels)
        print("source: explicit --logs")
    else:
        runs = discover_runs(root)
        print("source: auto-discovery from", root)

    if not runs:
        print("No runs found. Nothing to plot.")
        return

    for run in runs:
        print(
            "loaded",
            run["run_log_path"],
            "rows:",
            len(run["rows"]),
            "| operator rows:",
            len(run["operator_rows"]),
        )

    runs_by_mode = group_runs_by_mode(runs)
    operator_logs_by_mode = {
        mode: [row for run in mode_runs for row in run["operator_rows"]]
        for mode, mode_runs in runs_by_mode.items()
    }

    plot_fitness(runs_by_mode, outdir)
    plot_operator(runs_by_mode, outdir)
    plot_invalid_diversity(runs_by_mode, outdir)
    plot_operator_effect(operator_logs_by_mode, outdir)
    plot_diagnosis_effect(operator_logs_by_mode, outdir)
    print_summary(runs_by_mode, operator_logs_by_mode)


if __name__ == "__main__":
    main()
