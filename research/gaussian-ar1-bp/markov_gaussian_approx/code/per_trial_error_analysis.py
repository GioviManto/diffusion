"""
Per-trial error analysis for Gaussian-projected BP on the Laplace chain.

The aggregate experiment (bp_general_markov_diffusion_experiments.py) reports
mean and standard deviation of the Gaussian-message score error over trials.
Those statistics are dominated by rare catastrophic trials in which the exact
posterior belief becomes strongly non-Gaussian (e.g. bimodal) and the Gaussian
projection commits to a wrong compromise. This script looks at the full
per-trial distribution instead:

1. For every diffusion time t, run many independent trials and record the
   relative score error of Gaussian projected BP against reference grid BP.
2. Report median and quantile bands, which are robust to outliers, plus the
   failure rate P(relative error > 0.1).
3. Save the single worst trial: the belief at the most-wrong site, showing
   how the Gaussian projection fails when the true belief is multimodal.

Outputs (written to --output-dir):
    laplace_per_trial_errors.csv
    laplace_score_error_quantiles.png
    laplace_score_error_histogram.png
    worst_case_belief.png
    per_trial_summary.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bp_general_markov_diffusion_experiments import (
    ExperimentConfig,
    build_transition_matrix,
    cosine_similarity,
    gaussian_projected_bp,
    grid_bp,
    likelihood_matrix,
    make_grid,
    sample_clean_chain,
    sample_noisy_observation,
    score_from_mean,
)

FAILURE_THRESHOLD = 0.1


def run_per_trial(config: ExperimentConfig, n_trials: int, seed: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    grid, weights = make_grid(config, config.ref_grid_size)
    K = build_transition_matrix(grid, config.rho, prior="laplace")

    rows: list[dict[str, float]] = []
    worst: dict[str, object] | None = None

    for t in config.t_values:
        for trial in range(n_trials):
            a = sample_clean_chain(rng, config.n, config.rho, prior="laplace")
            x, alpha, delta = sample_noisy_observation(rng, a, t)
            ell = likelihood_matrix(grid, x, alpha, delta)

            mean_ref, beliefs_ref, _ = grid_bp(grid, weights, K, ell)
            mean_g, beliefs_g, _, _ = gaussian_projected_bp(grid, weights, K, ell)

            score_ref = score_from_mean(x, mean_ref, alpha, delta)
            score_g = score_from_mean(x, mean_g, alpha, delta)

            rel = float(np.linalg.norm(score_g - score_ref) / np.linalg.norm(score_ref))
            cos = cosine_similarity(score_g, score_ref)
            mean_mse = float(np.mean((mean_g - mean_ref) ** 2))

            rows.append({
                "t": float(t),
                "trial": float(trial),
                "relative_score_error": rel,
                "score_cosine": float(cos),
                "posterior_mean_mse": mean_mse,
            })

            if worst is None or rel > float(worst["rel"]):
                site = int(np.argmax(np.abs(mean_g - mean_ref)))
                worst = {
                    "rel": rel,
                    "t": float(t),
                    "trial": trial,
                    "site": site,
                    "grid": grid.copy(),
                    "belief_ref": beliefs_ref[site].copy(),
                    "belief_g": beliefs_g[site].copy(),
                    "mean_ref": float(mean_ref[site]),
                    "mean_g": float(mean_g[site]),
                }

    write_rows(output_dir / "laplace_per_trial_errors.csv", rows)
    plot_quantiles(config, rows, output_dir)
    plot_histogram(config, rows, output_dir)
    plot_worst_case(worst, output_dir)
    write_summary(config, rows, worst, n_trials, seed, output_dir)


def write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def errors_by_t(config: ExperimentConfig, rows: list[dict[str, float]]) -> dict[float, np.ndarray]:
    return {
        t: np.array([r["relative_score_error"] for r in rows if r["t"] == t])
        for t in config.t_values
    }


def plot_quantiles(config: ExperimentConfig, rows: list[dict[str, float]], output_dir: Path) -> None:
    by_t = errors_by_t(config, rows)
    t_arr = np.array(list(by_t.keys()))
    q10 = np.array([np.quantile(v, 0.10) for v in by_t.values()])
    q50 = np.array([np.quantile(v, 0.50) for v in by_t.values()])
    q90 = np.array([np.quantile(v, 0.90) for v in by_t.values()])
    vmax = np.array([np.max(v) for v in by_t.values()])

    plt.figure(figsize=(7.2, 4.6))
    plt.fill_between(t_arr, q10, q90, alpha=0.25, label="10%-90% quantile band")
    plt.plot(t_arr, q50, marker="o", label="median")
    plt.plot(t_arr, vmax, marker="^", linestyle="--", alpha=0.7, label="worst trial")
    plt.yscale("log")
    plt.xlabel("diffusion time t")
    plt.ylabel("relative score error, Gaussian projected vs grid BP")
    plt.title("Laplace chain: per-trial distribution of Gaussian-message score error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "laplace_score_error_quantiles.png", dpi=220)
    plt.close()


def plot_histogram(config: ExperimentConfig, rows: list[dict[str, float]], output_dir: Path) -> None:
    by_t = errors_by_t(config, rows)
    # Histogram at the diffusion time with the heaviest error tail.
    t_worst = max(by_t, key=lambda t: float(np.quantile(by_t[t], 0.90)))
    values = by_t[t_worst]

    plt.figure(figsize=(7.2, 4.6))
    bins = np.logspace(np.log10(max(values.min(), 1e-6)), np.log10(values.max() * 1.2), 40)
    plt.hist(values, bins=bins, edgecolor="black", alpha=0.8)
    plt.axvline(FAILURE_THRESHOLD, color="red", linestyle="--", label=f"failure threshold {FAILURE_THRESHOLD}")
    plt.xscale("log")
    plt.xlabel("relative score error")
    plt.ylabel("number of trials")
    plt.title(f"Laplace chain, t={t_worst}: per-trial score errors ({len(values)} trials)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "laplace_score_error_histogram.png", dpi=220)
    plt.close()


def plot_worst_case(worst: dict[str, object] | None, output_dir: Path) -> None:
    if worst is None:
        return
    grid = worst["grid"]
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(grid, worst["belief_ref"], label="reference grid BP belief")
    plt.plot(grid, worst["belief_g"], linestyle="--", label="Gaussian projected belief")
    plt.axvline(float(worst["mean_ref"]), linestyle=":", color="C0", label="reference mean")
    plt.axvline(float(worst["mean_g"]), linestyle=":", color="C1", label="Gaussian-message mean")
    plt.xlabel(f"a_{worst['site']}")
    plt.ylabel("posterior belief density")
    plt.title(
        f"Worst trial: t={worst['t']}, relative score error={float(worst['rel']):.2f}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "worst_case_belief.png", dpi=220)
    plt.close()


def write_summary(
    config: ExperimentConfig,
    rows: list[dict[str, float]],
    worst: dict[str, object] | None,
    n_trials: int,
    seed: int,
    output_dir: Path,
) -> None:
    by_t = errors_by_t(config, rows)
    lines = [
        "Per-trial Gaussian-message error analysis (Laplace chain)",
        "=========================================================",
        f"n = {config.n}, rho = {config.rho}, ref grid = {config.ref_grid_size} on "
        f"[{config.grid_min}, {config.grid_max}], trials per t = {n_trials}, seed = {seed}",
        "",
        f"{'t':>6}  {'median':>10}  {'q10':>10}  {'q90':>10}  {'max':>10}  {'P(err>' + str(FAILURE_THRESHOLD) + ')':>12}",
    ]
    for t, v in by_t.items():
        lines.append(
            f"{t:>6.2f}  {np.quantile(v, 0.5):>10.4f}  {np.quantile(v, 0.1):>10.4f}  "
            f"{np.quantile(v, 0.9):>10.4f}  {np.max(v):>10.4f}  {float(np.mean(v > FAILURE_THRESHOLD)):>12.3f}"
        )
    if worst is not None:
        lines += [
            "",
            f"Worst trial: t={worst['t']}, trial={worst['trial']}, site={worst['site']}, "
            f"relative score error={float(worst['rel']):.4f}",
            f"reference mean={float(worst['mean_ref']):.4f} vs Gaussian-message mean={float(worst['mean_g']):.4f}",
        ]
    text = "\n".join(lines) + "\n"
    (output_dir / "per_trial_summary.txt").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-trial Gaussian-message error analysis.")
    parser.add_argument("--n-trials", type=int, default=100, help="trials per diffusion time")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="per_trial_outputs")
    args = parser.parse_args()

    config = ExperimentConfig()
    run_per_trial(config, n_trials=args.n_trials, seed=args.seed, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
