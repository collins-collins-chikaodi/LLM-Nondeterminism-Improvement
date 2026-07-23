
"""
model_loader.py

Loads the tokenizer and TinyLlama model used throughout the project.

Author:
Collins Collins

Project:
Deterministic LLM Inference Evaluation Framework
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MODEL_NAME


def load_model():
    """
    Load the tokenizer and TinyLlama language model.

    Returns
    -------
    tokenizer : AutoTokenizer
        Hugging Face tokenizer.

    model : AutoModelForCausalLM
        Pre-trained TinyLlama model ready for inference.
    """

    print("=" * 60)
    print("Loading TinyLlama Model...")
    print("=" * 60)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

    print("Model loaded successfully.")
    print("=" * 60)

    return tokenizer, model


def print_system_information():
    """
    Display PyTorch, Transformers and GPU information.
    """

    import transformers

    print("=" * 60)
    print("System Information")
    print("=" * 60)

    print(f"PyTorch Version      : {torch.__version__}")
    print(f"Transformers Version : {transformers.__version__}")
    print(f"CUDA Available       : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU Device           : {torch.cuda.get_device_name(0)}")

    print("=" * 60)


if __name__ == "__main__":
    print_system_information()
    tokenizer, model = load_model()
