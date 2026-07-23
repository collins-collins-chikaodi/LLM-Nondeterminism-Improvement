
"""
metrics.py
==========

Project:
    Deterministic LLM Inference Evaluation Framework

Author:
    Collins Collins

Description
-----------
This module provides utility functions for computing evaluation metrics
from deterministic LLM inference experiments.

The functions are responsible for calculating descriptive statistics,
checking response determinism and generating summary tables suitable for
export to CSV and Excel.
"""

from typing import Dict

import pandas as pd


def calculate_average_inference_time(results: pd.DataFrame) -> float:
    """
    Calculate the average inference time.

    Parameters
    ----------
    results : pd.DataFrame
        Evaluation results.

    Returns
    -------
    float
        Average inference time.
    """

    return round(results["Inference_Time_sec"].mean(), 3)


def calculate_minimum_inference_time(results: pd.DataFrame) -> float:
    """
    Calculate the minimum inference time.
    """

    return round(results["Inference_Time_sec"].min(), 3)


def calculate_maximum_inference_time(results: pd.DataFrame) -> float:
    """
    Calculate the maximum inference time.
    """

    return round(results["Inference_Time_sec"].max(), 3)


def calculate_average_gpu_memory(results: pd.DataFrame) -> float:
    """
    Calculate average GPU memory usage.
    """

    return round(results["GPU_Memory_MB"].mean(), 2)


def calculate_minimum_gpu_memory(results: pd.DataFrame) -> float:
    """
    Calculate minimum GPU memory usage.
    """

    return round(results["GPU_Memory_MB"].min(), 2)


def calculate_maximum_gpu_memory(results: pd.DataFrame) -> float:
    """
    Calculate maximum GPU memory usage.
    """

    return round(results["GPU_Memory_MB"].max(), 2)


def calculate_average_response_length(results: pd.DataFrame) -> float:
    """
    Calculate average response length.

    Parameters
    ----------
    results : pd.DataFrame

    Returns
    -------
    float
        Average response length.
    """

    return round(results["Response_Length"].mean(), 0)


def check_determinism(results: pd.DataFrame) -> bool:
    """
    Check whether every generated response is identical.

    Parameters
    ----------
    results : pd.DataFrame

    Returns
    -------
    bool
        True if all responses are identical.
    """

    return results["Response"].nunique() == 1


def unique_responses(results: pd.DataFrame) -> int:
    """
    Count the number of unique generated responses.

    Parameters
    ----------
    results : pd.DataFrame

    Returns
    -------
    int
        Number of unique responses.
    """

    return results["Response"].nunique()


def generate_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary DataFrame containing all evaluation metrics.

    Parameters
    ----------
    results : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Summary statistics.
    """

    summary = pd.DataFrame({

        "Metric": [

            "Average Inference Time (sec)",

            "Minimum Inference Time (sec)",

            "Maximum Inference Time (sec)",

            "Average GPU Memory (MB)",

            "Minimum GPU Memory (MB)",

            "Maximum GPU Memory (MB)",

            "Average Response Length",

            "Unique Responses",

            "Deterministic Output"

        ],

        "Value": [

            calculate_average_inference_time(results),

            calculate_minimum_inference_time(results),

            calculate_maximum_inference_time(results),

            calculate_average_gpu_memory(results),

            calculate_minimum_gpu_memory(results),

            calculate_maximum_gpu_memory(results),

            calculate_average_response_length(results),

            unique_responses(results),

            "Yes" if check_determinism(results) else "No"

        ]

    })

    return summary


def print_summary(results: pd.DataFrame) -> None:
    """
    Print evaluation summary.

    Parameters
    ----------
    results : pd.DataFrame
    """

    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Average Inference Time : {calculate_average_inference_time(results)} sec")

    print(f"Minimum Inference Time : {calculate_minimum_inference_time(results)} sec")

    print(f"Maximum Inference Time : {calculate_maximum_inference_time(results)} sec")

    print(f"Average GPU Memory     : {calculate_average_gpu_memory(results)} MB")

    print(f"Minimum GPU Memory     : {calculate_minimum_gpu_memory(results)} MB")

    print(f"Maximum GPU Memory     : {calculate_maximum_gpu_memory(results)} MB")

    print(f"Average Response Length: {calculate_average_response_length(results)}")

    print(f"Unique Responses       : {unique_responses(results)}")

    print(
        f"Deterministic Output   : "
        f"{'Yes' if check_determinism(results) else 'No'}"
    )

    print("=" * 70)


if __name__ == "__main__":

    print(
        "metrics.py contains utility functions for evaluation metrics."
    )
