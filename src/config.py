"""
config.py
=========

Project:
    Deterministic LLM Inference Evaluation Framework

Author:
    Collins Collins

Description
-----------
This module centralises all configurable parameters used throughout the
project, including model settings, text generation parameters,
experiment configuration and output file locations.

Keeping configuration values in a single location improves code
maintainability, readability and reproducibility.
"""

from pathlib import Path

# =============================================================================
# Project Directories
# =============================================================================

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory for exported CSV and Excel files
RESULTS_DIR = PROJECT_ROOT / "results"

# Directory for generated figures
IMAGES_DIR = PROJECT_ROOT / "images"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Model Configuration
# =============================================================================

MODEL_NAME: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# =============================================================================
# Text Generation Configuration
# =============================================================================

MAX_NEW_TOKENS: int = 120
MIN_NEW_TOKENS: int = 40

DO_SAMPLE: bool = False

TEMPERATURE = None
TOP_P = None
TOP_K = None

REPETITION_PENALTY: float = 1.0

# =============================================================================
# Experiment Configuration
# =============================================================================

BASELINE_RUNS: int = 5
PRECISION_RUNS: int = 5

# =============================================================================
# Output Files
# =============================================================================

BASELINE_RESULTS_CSV = RESULTS_DIR / "baseline_results.csv"
BASELINE_RESULTS_XLSX = RESULTS_DIR / "Baseline_Results.xlsx"

BASELINE_EXPERIMENT_CSV = RESULTS_DIR / "baseline_experiment.csv"

PRECISION_RESULTS_CSV = RESULTS_DIR / "precision_evaluation.csv"
PRECISION_RESULTS_XLSX = RESULTS_DIR / "precision_evaluation.xlsx"

PRECISION_SUMMARY_CSV = RESULTS_DIR / "precision_summary.csv"
PRECISION_SUMMARY_XLSX = RESULTS_DIR / "precision_summary.xlsx"

# =============================================================================
# Figure Files
# =============================================================================

INFERENCE_TIME_FIGURE = IMAGES_DIR / "inference_time.png"
GPU_MEMORY_FIGURE = IMAGES_DIR / "gpu_memory.png"

# =============================================================================
# Random Seed
# =============================================================================

RANDOM_SEED: int = 42
