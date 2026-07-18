# Improving Numerical Determinism in LLM Inference using TinyLlama

## Overview

This project investigates numerical determinism during Large Language Model (LLM) inference using the TinyLlama-1.1B-Chat model. The study evaluates whether deterministic decoding produces consistent outputs while monitoring inference time, GPU memory utilisation, and response characteristics.

The project was developed as part of the MSc Deep Learning module.

---

## Objectives

The objectives of this project are to:

- Investigate numerical determinism during LLM inference.
- Evaluate the consistency of generated responses.
- Measure inference time across multiple executions.
- Analyse GPU memory utilisation during inference.
- Compare model behaviour across different prompts.

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- TinyLlama-1.1B-Chat
- Pandas
- NumPy
- Matplotlib
- Kaggle Notebook

---

## Methodology

The implementation consists of four major stages:

1. Environment setup and model loading.
2. Baseline inference experiment.
3. Determinism evaluation using repeated executions.
4. Precision evaluation using multiple prompts.
5. Statistical analysis and visualisation.
6. Export of experimental results.

---

## Experimental Results

The experiments evaluate:

- Deterministic text generation
- Inference time
- GPU memory usage
- Response length
- Statistical summary of experimental results

Generated outputs are exported in both CSV and Excel formats for further analysis.

---

## Repository Structure

```
Improving-Numerical-Determinism-in-LLM-Inference
│
├── README.md
├── notebook.ipynb
├── requirements.txt
├── results/
├── images/
└── report/
```

---

## How to Run

1. Clone the repository.
2. Install the required dependencies.

```
pip install -r requirements.txt
```

3. Open the notebook.

4. Run all cells sequentially.

---

## Results

The notebook generates:

- Baseline inference results
- Determinism evaluation
- Precision evaluation
- Statistical summaries
- Performance visualisations

---



Collins Collins

MSc Deep Learning
