
"""
config.py

Project configuration file for the MSc Deep Learning and Generative AI project.

Title:
Deterministic LLM Inference Evaluation Framework

Author:
Collins Collins

This module stores all configurable parameters used throughout the project.
"""

# ==========================================================
# Model Configuration
# ==========================================================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ==========================================================
# Text Generation Parameters
# ==========================================================

MAX_NEW_TOKENS = 120
MIN_NEW_TOKENS = 40

DO_SAMPLE = False
TEMPERATURE = None
TOP_P = None
TOP_K = None
REPETITION_PENALTY = 1.0

# ==========================================================
# Experiment Configuration
# ==========================================================

BASELINE_RUNS = 5
PRECISION_RUNS = 5

# ==========================================================
# Output Directories
# ==========================================================

RESULTS_DIR = "results"
IMAGES_DIR = "images"

# ==========================================================
# Output Files
# ==========================================================

BASELINE_RESULTS_CSV = "baseline_results.csv"
BASELINE_EXPERIMENT_CSV = "baseline_experiment.csv"

PRECISION_RESULTS_CSV = "precision_evaluation.csv"
PRECISION_RESULTS_XLSX = "precision_evaluation.xlsx"

PRECISION_SUMMARY_CSV = "precision_summary.csv"
PRECISION_SUMMARY_XLSX = "precision_summary.xlsx"

BASELINE_RESULTS_XLSX = "Baseline_Results.xlsx"

INFERENCE_TIME_FIGURE = "inference_time.png"
GPU_MEMORY_FIGURE = "gpu_memory.png"

# ==========================================================
# Random Seed (Reserved for Future Work)
# ==========================================================

RANDOM_SEED = 42
