"""
model_loader.py
===============

Project:
    Deterministic LLM Inference Evaluation Framework

Author:
    Collins Collins

Description
-----------
This module is responsible for loading the TinyLlama language model and
its associated tokenizer using the Hugging Face Transformers library.

The model is configured for inference only and is automatically placed
on the most appropriate available device.
"""

from typing import Tuple

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_NAME


def print_system_information() -> None:
    """
    Display information about the current execution environment.

    This includes the installed PyTorch version, Transformers version,
    CUDA availability and GPU information (if available).
    """

    print("=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)

    print(f"PyTorch Version      : {torch.__version__}")
    print(f"Transformers Version : {transformers.__version__}")
    print(f"CUDA Available       : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU Device           : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU Device           : CPU")

    print("=" * 70)
    print()


def load_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load the TinyLlama tokenizer and language model.

    Returns
    -------
    tuple
        A tuple containing:

        - tokenizer : AutoTokenizer
        - model : AutoModelForCausalLM

    Raises
    ------
    RuntimeError
        If the tokenizer or model cannot be loaded.
    """

    print("=" * 70)
    print("LOADING LANGUAGE MODEL")
    print("=" * 70)

    try:

        print("Loading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        print("Tokenizer loaded successfully.")

        print()

        print("Loading model...")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Ensure the model runs in inference mode
        model.eval()

        print("Model loaded successfully.")
        print("=" * 70)
        print()

        return tokenizer, model

    except Exception as error:

        raise RuntimeError(
            f"Failed to load TinyLlama model.\nReason: {error}"
        ) from error


def get_device(model: AutoModelForCausalLM) -> str:
    """
    Return the device currently used by the language model.

    Parameters
    ----------
    model : AutoModelForCausalLM
        Loaded TinyLlama model.

    Returns
    -------
    str
        Device name (e.g., 'cuda:0' or 'cpu').
    """

    return str(model.device)


def model_summary(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM
) -> None:
    """
    Display a concise summary of the loaded model.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        Loaded tokenizer.

    model : AutoModelForCausalLM
        Loaded language model.
    """

    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    print(f"Model Name           : {MODEL_NAME}")
    print(f"Vocabulary Size      : {tokenizer.vocab_size:,}")
    print(f"Model Device         : {get_device(model)}")

    print("=" * 70)
    print()


if __name__ == "__main__":

    print_system_information()

    tokenizer, model = load_model()

    model_summary(
        tokenizer=tokenizer,
        model=model
    )
