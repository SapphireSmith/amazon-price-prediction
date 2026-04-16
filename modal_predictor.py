import os
from typing import Any, Optional

import requests


def _get_url() -> str:
    # Default is your already-deployed Phase 1 endpoint; override via env.
    return os.getenv(
        "MODAL_PRICE_API_URL",
        "https://sapphiresmith--amazon-price-prediction-price-api.modal.run",
    )


def predict_price(description: str) -> float:
    if not description or not description.strip():
        raise ValueError("description must be a non-empty string.")

    url = _get_url()
    payload: dict[str, Any] = {"description": description.strip()}

    # Modal cold starts can take a while; give a bit more time for Phase 1 models.
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("ok", False):
        raise ValueError(f"Modal predictor error: {data}")

    if "predicted_price" not in data:
        raise ValueError(f"Modal predictor missing predicted_price: {data}")

    return float(data["predicted_price"])

