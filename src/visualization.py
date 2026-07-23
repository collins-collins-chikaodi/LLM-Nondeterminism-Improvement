
"""
visualization.py
================

Project:
    Deterministic LLM Inference Evaluation Framework

Author:
    Collins Collins

Description
-----------
This module generates visualisations for deterministic LLM inference
experiments.

The plots summarise:

    • Inference Time
    • GPU Memory Usage

The generated figures are automatically saved into the project's
images directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import (
    GPU_MEMORY_FIGURE,
    INFERENCE_TIME_FIGURE,
)


def plot_inference_time(results: pd.DataFrame) -> Path:
    """
    Plot inference time for each experiment.

    Parameters
    ----------
    results : pandas.DataFrame
        Evaluation results.

    Returns
    -------
    pathlib.Path
        Location of the saved figure.
    """

    plt.figure(figsize=(8, 5))

    plt.bar(
        range(1, len(results) + 1),
        results["Inference_Time_sec"],
    )

    plt.title("Inference Time per Prompt")

    plt.xlabel("Prompt Number")

    plt.ylabel("Inference Time (seconds)")

    plt.xticks(range(1, len(results) + 1))

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        INFERENCE_TIME_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return INFERENCE_TIME_FIGURE


def plot_gpu_memory(results: pd.DataFrame) -> Path:
    """
    Plot GPU memory usage.

    Parameters
    ----------
    results : pandas.DataFrame
        Evaluation results.

    Returns
    -------
    pathlib.Path
        Location of the saved figure.
    """

    plt.figure(figsize=(8, 5))

    plt.bar(
        range(1, len(results) + 1),
        results["GPU_Memory_MB"],
    )

    plt.title("GPU Memory Usage per Prompt")

    plt.xlabel("Prompt Number")

    plt.ylabel("GPU Memory (MB)")

    plt.xticks(range(1, len(results) + 1))

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        GPU_MEMORY_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return GPU_MEMORY_FIGURE


def generate_all_figures(results: pd.DataFrame) -> None:
    """
    Generate every visualisation.

    Parameters
    ----------
    results : pandas.DataFrame
        Precision evaluation results.
    """

    print("=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    inference_path = plot_inference_time(results)

    print(f"✓ Saved: {inference_path}")

    memory_path = plot_gpu_memory(results)

    print(f"✓ Saved: {memory_path}")

    print()

    print("Visualisation completed.")

    print("=" * 70)


if __name__ == "__main__":

    print(
        "Visualization module for deterministic LLM evaluation."
    )
