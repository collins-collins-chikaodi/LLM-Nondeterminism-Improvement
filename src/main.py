"""
main.py
=======

Project:
    Deterministic LLM Inference Evaluation Framework

Author:
    Collins Collins

Description
-----------
Entry point for the Deterministic LLM Inference Evaluation Framework.

Workflow
--------
1. Display system information
2. Load TinyLlama model
3. Execute baseline evaluation
4. Execute precision evaluation
5. Compute summary statistics
6. Export CSV and Excel files
7. Generate visualisations
"""

import logging
import sys

from evaluation import (
    run_baseline_evaluation,
    run_precision_evaluation,
)

from export_results import (
    export_all_results,
)

from metrics import (
    generate_summary,
)

from model_loader import (
    load_model,
    model_summary,
    print_system_information,
)

from visualization import (
    generate_all_figures,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


logger = logging.getLogger(__name__)


def main() -> None:
    """
    Execute the complete deterministic inference workflow.
    """

    try:

        logger.info("=" * 70)
        logger.info("Deterministic LLM Inference Evaluation Framework")
        logger.info("=" * 70)

        # ---------------------------------------------------------
        # System Information
        # ---------------------------------------------------------

        print_system_information()

        # ---------------------------------------------------------
        # Load Model
        # ---------------------------------------------------------

        tokenizer, model = load_model()

        model_summary(
            tokenizer=tokenizer,
            model=model,
        )

        # ---------------------------------------------------------
        # Baseline Evaluation
        # ---------------------------------------------------------

        logger.info("Running baseline evaluation...")

        baseline_experiment = run_baseline_evaluation(
            model=model,
            tokenizer=tokenizer,
        )

        # Baseline single-result file
        baseline_results = baseline_experiment.iloc[[0]].copy()

        # ---------------------------------------------------------
        # Precision Evaluation
        # ---------------------------------------------------------

        logger.info("Running precision evaluation...")

        precision_results = run_precision_evaluation(
            model=model,
            tokenizer=tokenizer,
        )

        # ---------------------------------------------------------
        # Summary Statistics
        # ---------------------------------------------------------

        logger.info("Generating evaluation summary...")

        summary = generate_summary(
            precision_results,
        )

        # ---------------------------------------------------------
        # Export Results
        # ---------------------------------------------------------

        logger.info("Exporting project outputs...")

        export_all_results(
            baseline_results=baseline_results,
            baseline_experiment=baseline_experiment,
            precision_results=precision_results,
            summary=summary,
        )

        # ---------------------------------------------------------
        # Generate Figures
        # ---------------------------------------------------------

        logger.info("Generating figures...")

        generate_all_figures(
            precision_results,
        )

        logger.info("=" * 70)
        logger.info("Project completed successfully.")
        logger.info("=" * 70)

    except Exception as error:

        logger.exception(
            "Project execution failed."
        )

        sys.exit(
            f"\nApplication terminated.\nReason: {error}"
        )


if __name__ == "__main__":

    main()
