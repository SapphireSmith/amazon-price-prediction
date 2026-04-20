import os
from typing import Any

import requests
from prompt_builder import build_prompt


def _get_url() -> str:
    # Default is your already-deployed Phase 1 endpoint; override via env.
    return os.getenv(
        "MODAL_PRICE_API_URL",
        "https://sapphiresmith--amazon-price-prediction-price-api.modal.run",
    )


def predict_price(structured_description: str) -> float:
    if not structured_description or not structured_description.strip():
        raise ValueError("structured_description must be a non-empty string.")

    prompt = build_prompt(structured_description.strip())
    # print(f"\n prompt={prompt}\n")
    url = _get_url()
    payload: dict[str, Any] = {"description": prompt}

    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("ok", False):
        raise ValueError(f"Modal predictor error: {data}")

    if "predicted_price" not in data:
        raise ValueError(f"Modal predictor missing predicted_price: {data}")

    return float(data["predicted_price"])
