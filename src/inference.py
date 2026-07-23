"""
inference.py
============

Project:
    Deterministic LLM Inference Evaluation Framework

Author:
    Collins Collins

Description
-----------
This module performs deterministic inference using the TinyLlama
language model. It handles prompt tokenisation, response generation,
performance measurement and repeated inference experiments.
"""

from typing import Dict, List

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    DO_SAMPLE,
    MAX_NEW_TOKENS,
    MIN_NEW_TOKENS,
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)


def prepare_input(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
):
    """
    Convert a text prompt into model-ready tensors.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        Hugging Face tokenizer.

    model : AutoModelForCausalLM
        Loaded TinyLlama model.

    prompt : str
        User prompt.

    Returns
    -------
    BatchEncoding
        Tokenised prompt moved to the appropriate device.
    """

    return tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    ).to(model.device)


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
) -> Dict:
    """
    Perform deterministic inference for a single prompt.

    Parameters
    ----------
    model : AutoModelForCausalLM
        Loaded TinyLlama model.

    tokenizer : AutoTokenizer
        Hugging Face tokenizer.

    prompt : str
        Input prompt.

    Returns
    -------
    dict
        Dictionary containing inference statistics and
        generated response.
    """

    inputs = prepare_input(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MIN_NEW_TOKENS,

            do_sample=DO_SAMPLE,

            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,

            repetition_penalty=REPETITION_PENALTY,

            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    end_time = time.perf_counter()

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    inference_time = round(
        end_time - start_time,
        3,
    )

    if torch.cuda.is_available():

        gpu_memory = round(
            torch.cuda.max_memory_allocated()
            / (1024 ** 2),
            2,
        )

    else:

        gpu_memory = 0.0

    return {

        "Prompt": prompt,

        "Inference_Time_sec": inference_time,

        "GPU_Memory_MB": gpu_memory,

        "Response_Length": len(response),

        "Response": response,

    }


def repeat_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    runs: int,
) -> List[Dict]:
    """
    Execute deterministic inference multiple times.

    Parameters
    ----------
    model : AutoModelForCausalLM
        Loaded language model.

    tokenizer : AutoTokenizer
        Hugging Face tokenizer.

    prompt : str
        Input prompt.

    runs : int
        Number of repeated executions.

    Returns
    -------
    list
        List of dictionaries containing inference results.
    """

    results = []

    for run in range(1, runs + 1):

        result = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
        )

        result["Run"] = run

        results.append(result)

    return results


def display_result(result: Dict) -> None:
    """
    Print a formatted inference result.

    Parameters
    ----------
    result : dict
        Dictionary returned by generate_response().
    """

    print("=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(result["Prompt"])

    print()

    print("=" * 80)
    print("GENERATED RESPONSE")
    print("=" * 80)
    print(result["Response"])

    print()

    print(f"Inference Time : {result['Inference_Time_sec']} sec")
    print(f"GPU Memory     : {result['GPU_Memory_MB']} MB")
    print(f"Response Length: {result['Response_Length']} characters")

    print("=" * 80)
    print()


def check_determinism(results: List[Dict]) -> bool:
    """
    Check whether repeated inference produced identical responses.

    Parameters
    ----------
    results : list
        Output from repeat_inference().

    Returns
    -------
    bool
        True if every generated response is identical.
    """

    responses = [
        result["Response"]
        for result in results
    ]

    return len(set(responses)) == 1
