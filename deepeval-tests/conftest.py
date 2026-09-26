"""
Shared fixtures and configuration for deepeval LLM evaluation tests.
Configured to use Ollama Cloud API as the evaluation LLM judge.
"""

import os
import pytest
from deepeval.metrics import GEval
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCaseParams

# ---------------------------------------------------------------------------
# Ollama Evaluation Model Configuration
# ---------------------------------------------------------------------------

OLLAMA_API_KEY = os.getenv(
    "OLLAMA_API_KEY", "9631a3972df34aba9bb27acc7f709b24.s4SOSjZLrR3PhuRaXerzvjgA"
)
_raw_base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
OLLAMA_BASE_URL = (
    _raw_base_url if _raw_base_url.endswith("/v1") or _raw_base_url.endswith("/v1/")
    else f"{_raw_base_url.rstrip('/')}/v1"
)
OLLAMA_EVAL_MODEL = os.getenv("OLLAMA_EVAL_MODEL", "glm-5.3-flash")

eval_model = OpenAIModel(
    model=OLLAMA_EVAL_MODEL,
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
)


# ---------------------------------------------------------------------------
# Reusable GEval metric factories
# ---------------------------------------------------------------------------

def json_schema_metric(schema_description: str):
    """Creates a GEval metric that checks JSON schema compliance."""
    return GEval(
        name="JSON Schema Compliance",
        criteria=(
            "Evaluate whether the actual output is valid JSON that conforms to "
            "the required schema. Only check structure, key names, and data "
            "types — do NOT penalize for specific values. "
            + schema_description
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=eval_model,
    )


def output_correctness_metric():
    """Creates a GEval metric that checks factual/logical correctness."""
    return GEval(
        name="Output Correctness",
        criteria=(
            "Determine whether the actual output is logically correct and "
            "reasonable given the input text. The analysis should make sense "
            "for the provided input."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=eval_model,
    )


def answer_relevancy_metric():
    """Creates a GEval metric that checks whether the output is topically
    relevant to the input.  Unlike AnswerRelevancyMetric (which assumes a
    Q&A format), this works for classification and analysis endpoints where
    the output is structured metadata about the input text."""
    return GEval(
        name="Answer Relevancy",
        criteria=(
            "Evaluate whether the actual output is topically relevant to the "
            "input text. The labels, categories, or analysis in the output "
            "should directly relate to the subject matter of the input. "
            "Structured metadata (labels, categories, confidence scores) that "
            "accurately describes the input text should be considered relevant."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=eval_model,
    )
