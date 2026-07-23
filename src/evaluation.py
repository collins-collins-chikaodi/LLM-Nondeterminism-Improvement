
"""
evaluation.py
=============

Project:
    Deterministic LLM Inference Evaluation Framework

Author:
    Collins Collins

Description
-----------
This module performs the baseline and precision evaluation experiments.

It coordinates deterministic inference across prompts and converts the
results into pandas DataFrames for downstream analysis.

The module does not calculate statistics or export files.
Those responsibilities are delegated to the metrics and export modules.
"""

from typing import List

import pandas as pd

from config import (
    BASELINE_RUNS,
    PRECISION_RUNS,
)

from inference import (
    generate_response,
    repeat_inference,
)

from prompts import (
    get_baseline_prompt,
    get_precision_prompts,
)


def run_baseline_evaluation(
    model,
    tokenizer,
) -> pd.DataFrame:
    """
    Execute the baseline deterministic inference experiment.

    Parameters
    ----------
    model
        Loaded TinyLlama model.

    tokenizer
        Hugging Face tokenizer.

    Returns
    -------
    pandas.DataFrame
        Baseline evaluation results.
    """

    print("=" * 70)
    print("RUNNING BASELINE EVALUATION")
    print("=" * 70)

    prompt = get_baseline_prompt()

    results = repeat_inference(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        runs=BASELINE_RUNS,
    )

    results_df = pd.DataFrame(results)

    print("Baseline evaluation completed.")
    print()

    return results_df


def run_precision_evaluation(
    model,
    tokenizer,
) -> pd.DataFrame:
    """
    Execute the precision evaluation experiment.

    Parameters
    ----------
    model
        Loaded TinyLlama model.

    tokenizer
        Hugging Face tokenizer.

    Returns
    -------
    pandas.DataFrame
        Precision evaluation results.
    """

    print("=" * 70)
    print("RUNNING PRECISION EVALUATION")
    print("=" * 70)

    prompts = get_precision_prompts()

    precision_results = []

    for index, prompt in enumerate(prompts, start=1):

        print(f"Running Prompt {index}/{len(prompts)}")

        result = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
        )

        precision_results.append(result)

    precision_df = pd.DataFrame(precision_results)

    print()
    print("Precision evaluation completed.")
    print()

    return precision_df


def preview_results(
    results: pd.DataFrame,
    rows: int = 5,
) -> None:
    """
    Display the first few rows of an evaluation DataFrame.

    Parameters
    ----------
    results : pandas.DataFrame

    rows : int
        Number of rows to display.
    """

    print("=" * 70)
    print("RESULT PREVIEW")
    print("=" * 70)

    print(results.head(rows))

    print()


def experiment_information() -> None:
    """
    Display information about the available experiments.
    """

    print("=" * 70)
    print("AVAILABLE EXPERIMENTS")
    print("=" * 70)

    print(f"Baseline Runs  : {BASELINE_RUNS}")
    print(f"Precision Runs : {PRECISION_RUNS}")

    print()


def run_all_evaluations(
    model,
    tokenizer,
) -> List[pd.DataFrame]:
    """
    Execute both evaluation phases.

    Parameters
    ----------
    model
        Loaded TinyLlama model.

    tokenizer
        Hugging Face tokenizer.

    Returns
    -------
    list
        List containing:

        - Baseline DataFrame
        - Precision DataFrame
    """

    baseline_results = run_baseline_evaluation(
        model=model,
        tokenizer=tokenizer,
    )

    precision_results = run_precision_evaluation(
        model=model,
        tokenizer=tokenizer,
    )

    return [
        baseline_results,
        precision_results,
    ]


if __name__ == "__main__":

    print(
        "This module performs the baseline and precision "
        "evaluation experiments."
    )
