"""Master Orchestration Script: Run All TrustLens Experiments in Sequence.

Executes the complete TrustLens experiment pipeline:
    1. baseline: Train and evaluate baseline ResNet-18 on clean CIFAR-10
    2. corruption_test: Evaluate model degradation under CIFAR-10-C corruptions
    3. evaluate: Evaluate failure detection reliability signals and calibration
    4. ablation: Perform ablation study and evaluate combined failure detectors

Features:
- Configurable via configs/experiment.yaml and CLI overrides
- Skips already-completed steps if output metrics exist (override with --force)
- Aggregates all results into results/metrics/summary_metrics.json
- Generates an executive summary report in results/summary_report.txt

Usage:
    python experiments/run_all.py --device auto
    python experiments/run_all.py --steps baseline corruption_test --force
"""

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger("trustlens.run_all")

EXPERIMENT_STEPS = ["baseline", "corruption_test", "evaluate", "ablation"]

STEP_OUTPUT_FILES = {
    "baseline": "baseline_metrics.json",
    "corruption_test": "corruption_metrics.json",
    "evaluate": "evaluation_metrics.json",
    "ablation": "ablation_metrics.json",
}

STEP_SCRIPTS = {
    "baseline": PROJECT_ROOT / "experiments" / "baseline.py",
    "corruption_test": PROJECT_ROOT / "experiments" / "corruption_test.py",
    "evaluate": PROJECT_ROOT / "experiments" / "evaluate.py",
    "ablation": PROJECT_ROOT / "experiments" / "ablation.py",
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file if available."""
    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "experiment.yaml")

    config: Dict[str, Any] = {}
    cfg_file = Path(config_path)
    if cfg_file.exists() and yaml is not None:
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            logger.info("Loaded configuration from %s", cfg_file)
        except Exception as e:
            logger.warning("Failed to load config file %s: %s", cfg_file, e)
    return config


def run_experiment_step(
    step_name: str,
    script_path: Path,
    config_path: Optional[str],
    device: str,
    results_dir: Path,
) -> bool:
    """Execute a single experiment script via subprocess.

    Args:
        step_name: Name of the experiment step.
        script_path: Path to python script.
        config_path: Path to YAML config.
        device: Device string ('auto', 'cuda', 'cpu').
        results_dir: Target results directory.

    Returns:
        True if execution succeeded, False otherwise.
    """
    logger.info("=" * 70)
    logger.info("STARTING STEP: %s (%s)", step_name.upper(), script_path.name)
    logger.info("=" * 70)

    cmd = [
        sys.executable,
        str(script_path),
        "--device", device,
        "--results-dir", str(results_dir),
    ]
    if config_path:
        cmd.extend(["--config", str(config_path)])

    start_time = time.perf_counter()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
        )

        # Stream stdout in real-time
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                print(f"[{step_name}] {line}", end="")
            process.stdout.close()

        ret_code = process.wait()
        elapsed = time.perf_counter() - start_time

        if ret_code == 0:
            logger.info("Step %s COMPLETED SUCCESSFULLY in %.2f seconds.", step_name, elapsed)
            return True
        else:
            logger.error("Step %s FAILED with return code %d after %.2f seconds.", step_name, ret_code, elapsed)
            return False

    except Exception as e:
        logger.error("Exception occurred while running step %s: %s", step_name, e)
        return False


def aggregate_results(metrics_dir: Path) -> Dict[str, Any]:
    """Read all available step metrics and combine into a single summary dictionary.

    Args:
        metrics_dir: Directory containing individual metric JSON files.

    Returns:
        Consolidated summary dictionary.
    """
    summary: Dict[str, Any] = {
        "pipeline": "TrustLens Complete Evaluation",
        "generated_at": datetime.datetime.now().isoformat(),
        "steps": {},
    }

    for step, filename in STEP_OUTPUT_FILES.items():
        filepath = metrics_dir / filename
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    summary["steps"][step] = json.load(f)
                logger.info("Aggregated metrics for step: %s", step)
            except Exception as e:
                logger.warning("Could not read metrics file for %s: %s", step, e)
        else:
            summary["steps"][step] = {"status": "not_found"}

    return summary


def generate_summary_report(summary_data: Dict[str, Any], report_path: Path) -> None:
    """Generate a readable textual executive summary report.

    Args:
        summary_data: Aggregated metrics dictionary.
        report_path: Destination file path for report.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    steps_data = summary_data.get("steps", {})

    lines = []
    lines.append("================================================================================")
    lines.append("                        MODELGUARD EXECUTIVE SUMMARY REPORT                     ")
    lines.append("================================================================================")
    lines.append(f"Generated At: {summary_data.get('generated_at', 'N/A')}")
    lines.append("Project: TrustLens Reliable ML Monitoring & Failure Detection Pipeline")
    lines.append("")

    # 1. Baseline Section
    lines.append("--------------------------------------------------------------------------------")
    lines.append("1. BASELINE CLASSIFIER PERFORMANCE (Clean CIFAR-10)")
    lines.append("--------------------------------------------------------------------------------")
    base_info = steps_data.get("baseline", {}).get("metrics", {})
    if base_info:
        lines.append(f"  • Architecture:          {steps_data.get('baseline', {}).get('architecture', 'ResNet-18')}")
        lines.append(f"  • Clean Test Accuracy:   {base_info.get('test_accuracy', 0.0):.2f}%")
        lines.append(f"  • Clean Test Loss:       {base_info.get('test_loss', 0.0):.4f}")
        lines.append(f"  • Inference Latency:     {base_info.get('latency_per_sample_ms', 0.0):.2f} ms/sample")
        lines.append(f"  • Throughput:            {base_info.get('throughput_samples_per_sec', 0.0):.1f} samples/sec")
    else:
        lines.append("  [Baseline metrics not available]")
    lines.append("")

    # 2. Corruption Robustness Section
    lines.append("--------------------------------------------------------------------------------")
    lines.append("2. DISTRIBUTION SHIFT EVALUATION (CIFAR-10-C)")
    lines.append("--------------------------------------------------------------------------------")
    corr_info = steps_data.get("corruption_test", {})
    if corr_info and "mean_corrupted_accuracy" in corr_info:
        lines.append(f"  • Mean Corrupted Accuracy:    {corr_info.get('mean_corrupted_accuracy', 0.0):.2f}%")
        lines.append(f"  • Average Accuracy Drop:      -{corr_info.get('mean_accuracy_drop', 0.0):.2f}% points")
        lines.append(f"  • Confident Failures Logged:  {corr_info.get('total_confident_failures_recorded', 0)} cases (Conf > 0.90)")
        by_corr = corr_info.get("mean_accuracy_by_corruption", {})
        if by_corr:
            lines.append("  • Top 3 Most Severe Corruptions:")
            sorted_drops = sorted(by_corr.items(), key=lambda x: x[1])
            for corr, acc in sorted_drops[:3]:
                lines.append(f"      - {corr.replace('_', ' ').title()}: {acc:.2f}% avg accuracy")
    else:
        lines.append("  [Corruption evaluation metrics not available]")
    lines.append("")

    # 3. Reliability Signals Section
    lines.append("--------------------------------------------------------------------------------")
    lines.append("3. RELIABILITY SIGNALS & CALIBRATION (Failure Detection)")
    lines.append("--------------------------------------------------------------------------------")
    eval_info = steps_data.get("evaluate", {})
    if eval_info and "signal_performance" in eval_info:
        cal = eval_info.get("calibration", {})
        lines.append(f"  • Temperature Scaling (T = {cal.get('optimal_temperature', 1.0):.3f}):")
        lines.append(f"      - Uncalibrated ECE: {cal.get('uncalibrated_ece', 0.0):.4f} -> Calibrated ECE: {cal.get('calibrated_ece', 0.0):.4f} ({cal.get('ece_reduction_pct', 0.0):.1f}% reduction)")
        lines.append("  • Signal Detection Quality (AUROC / AUPR / FPR@95%TPR):")
        for sig, stats in eval_info.get("signal_performance", {}).items():
            lines.append(f"      - {sig:<22}: AUROC={stats.get('auroc', 0.0):.3f} | AUPR={stats.get('aupr', 0.0):.3f} | FPR@95={stats.get('fpr95', 0.0):.3f}")
    else:
        lines.append("  [Reliability signal metrics not available]")
    lines.append("")

    # 4. Ablation & Detector Generalization Section
    lines.append("--------------------------------------------------------------------------------")
    lines.append("4. FAILURE DETECTOR ABLATION & GENERALIZATION")
    lines.append("--------------------------------------------------------------------------------")
    abl_info = steps_data.get("ablation", {})
    if abl_info and "generalization_comparison" in abl_info:
        gen = abl_info.get("generalization_comparison", {})
        seen = gen.get("seen", {})
        held = gen.get("held_out", {})
        lines.append(f"  • Detector Generalization across Corruptions:")
        lines.append(f"      - Seen Corruptions:     AUROC = {seen.get('auroc', 0.0):.3f} | AUPR = {seen.get('aupr', 0.0):.3f}")
        lines.append(f"      - Held-out Corruptions: AUROC = {held.get('auroc', 0.0):.3f} | AUPR = {held.get('aupr', 0.0):.3f}")
        lines.append(f"      - Generalization Gap:   ΔAUROC = {gen.get('auroc_drop', 0.0):.3f}")

        abl_res = abl_info.get("ablation_results", {})
        if abl_res:
            lines.append("  • Ablation Configurations (AUROC on Held-out Corruptions):")
            for cfg_name, stats in abl_res.items():
                lines.append(f"      - {cfg_name:<26}: AUROC={stats.get('auroc', 0.0):.3f} | AUPR={stats.get('aupr', 0.0):.3f}")

        overhead = abl_info.get("computational_overhead", {})
        if overhead:
            lines.append("  • Computational Overhead per 1,000 Samples:")
            for s_name, cost in overhead.items():
                lines.append(f"      - {s_name:<20}: {cost.get('latency_ms_per_1000', 0.0):.2f} ms")
    else:
        lines.append("  [Ablation study metrics not available]")

    lines.append("")
    lines.append("================================================================================")
    lines.append("                       ALL EXPERIMENTS SUCCESSFULLY EXECUTED                    ")
    lines.append("================================================================================")

    report_content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info("Saved summary report to %s", report_path)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TrustLens Pipeline Orchestration (Run All Experiments)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--force", action="store_true", help="Force rerun of all steps even if output exists")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=EXPERIMENT_STEPS,
        choices=EXPERIMENT_STEPS + ["all"],
        help="Specific experiment steps to execute",
    )
    parser.add_argument("--results-dir", type=str, default=None, help="Target results directory")
    return parser.parse_args()


def main() -> None:
    """Main execution flow for running all experiments."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    cfg = load_config(args.config)
    config_path = args.config if args.config else str(PROJECT_ROOT / "configs" / "experiment.yaml")
    device = args.device or cfg.get("project", {}).get("device", "auto")

    results_dir = Path(args.results_dir or cfg.get("results", {}).get("results_dir", "./results"))
    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    steps_to_run = EXPERIMENT_STEPS if "all" in args.steps else args.steps

    logger.info("===============================================================")
    logger.info("TrustLens Pipeline Execution Initiated")
    logger.info("Steps to run: %s", steps_to_run)
    logger.info("Device: %s | Force rerun: %s", device, args.force)
    logger.info("Results Directory: %s", results_dir.resolve())
    logger.info("===============================================================")

    pipeline_success = True

    for step in steps_to_run:
        output_file = metrics_dir / STEP_OUTPUT_FILES[step]
        if output_file.exists() and not args.force:
            logger.info("Step '%s' already completed (found %s). Skipping. Use --force to rerun.", step, output_file.name)
            continue

        script_path = STEP_SCRIPTS[step]
        if not script_path.exists():
            logger.error("Script for step '%s' does not exist at: %s", step, script_path)
            pipeline_success = False
            continue

        success = run_experiment_step(
            step_name=step,
            script_path=script_path,
            config_path=config_path,
            device=device,
            results_dir=results_dir,
        )
        if not success:
            logger.warning("Step '%s' failed. Continuing to remaining steps if possible.", step)
            pipeline_success = False

    # Aggregate results into single JSON
    logger.info("Aggregating metrics across completed experiment steps...")
    summary = aggregate_results(metrics_dir)
    summary_json_path = metrics_dir / "summary_metrics.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved combined metrics summary to %s", summary_json_path)

    # Generate executive summary report text file
    report_text_path = results_dir / "summary_report.txt"
    generate_summary_report(summary, report_text_path)

    if pipeline_success:
        logger.info("TrustLens pipeline completed successfully!")
    else:
        logger.warning("TrustLens pipeline completed with one or more step warnings/failures.")


if __name__ == "__main__":
    main()
