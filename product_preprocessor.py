import os
from typing import Optional

from agents.preprocessor import Preprocessor


def _reasoning_effort_for_model(model_name: str) -> Optional[str]:
    return "low" if "gpt-oss" in (model_name or "") else None


def preprocess_description(description: str) -> str:
    model_name = os.getenv("PRICER_PREPROCESSOR_MODEL")
    if not model_name:
        raise ValueError("Missing PRICER_PREPROCESSOR_MODEL in environment.")
    if not description or not description.strip():
        raise ValueError("description must be a non-empty string.")

    preprocessor = Preprocessor(
        model_name=model_name,
        reasoning_effort=_reasoning_effort_for_model(model_name),
    )
    return preprocessor.preprocess(description.strip())

