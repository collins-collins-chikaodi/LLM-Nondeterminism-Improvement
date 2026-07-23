
"""
prompts.py
==========

Project:
    Deterministic LLM Inference Evaluation Framework

Author:
    Collins Collins

Description
-----------
This module contains all prompts used throughout the evaluation
experiments. Storing prompts in a dedicated module improves code
organisation, readability and maintainability.

The prompts are intentionally diverse to evaluate deterministic
behaviour across different artificial intelligence concepts.
"""

from typing import List

# =============================================================================
# Baseline Evaluation Prompt
# =============================================================================

BASELINE_PROMPT: str = (
    "Explain numerical determinism in deep learning in simple terms."
)

# =============================================================================
# Precision Evaluation Prompts
# =============================================================================

PRECISION_PROMPTS: List[str] = [

    "Explain Artificial Intelligence in simple terms.",

    "What is Machine Learning?",

    "Explain Deep Learning in simple terms.",

    "What is Numerical Determinism in deep learning?",

    "Explain Reinforcement Learning with a simple example."

]

# =============================================================================
# Helper Functions
# =============================================================================

def get_baseline_prompt() -> str:
    """
    Return the baseline evaluation prompt.

    Returns
    -------
    str
        Prompt used during the baseline experiment.
    """

    return BASELINE_PROMPT


def get_precision_prompts() -> List[str]:
    """
    Return all precision evaluation prompts.

    Returns
    -------
    List[str]
        Collection of prompts used for deterministic
        precision evaluation.
    """

    return PRECISION_PROMPTS.copy()


def total_prompts() -> int:
    """
    Return the total number of precision prompts.

    Returns
    -------
    int
        Number of prompts.
    """

    return len(PRECISION_PROMPTS)


if __name__ == "__main__":

    print("=" * 70)
    print("AVAILABLE PROMPTS")
    print("=" * 70)

    print()
    print("Baseline Prompt")
    print("----------------")
    print(BASELINE_PROMPT)

    print()

    print("Precision Evaluation Prompts")
    print("----------------------------")

    for index, prompt in enumerate(PRECISION_PROMPTS, start=1):
        print(f"{index}. {prompt}")

    print()
    print(f"Total Prompts: {total_prompts()}")
