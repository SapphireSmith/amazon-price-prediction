import os
import re
from typing import Any

import modal


APP_NAME = os.getenv("MODAL_APP_NAME", "amazon-price-prediction")
GPU_TYPE = os.getenv("MODAL_GPU", "T4")
HF_SECRET_NAME = os.getenv("MODAL_HF_SECRET_NAME", "huggingface-secret")

BASE_MODEL = os.getenv("HF_BASE_MODEL", "meta-llama/Llama-3.2-3B")
ADAPTER_MODEL = os.getenv("HF_ADAPTER_MODEL", "ed-donner/price-2025-11-28_18.47.07")
ADAPTER_REVISION = os.getenv(
    "HF_ADAPTER_REVISION", "b19c8bfea3b6ff62237fbb0a8da9779fc12cefbd"
)

QUESTION = "What does this cost to the nearest dollar?"
PREFIX = "Price is $"

app = modal.App(APP_NAME)
image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers",
    "bitsandbytes",
    "accelerate",
    "peft",
    "fastapi[standard]",
)
secrets = [modal.Secret.from_name(HF_SECRET_NAME)]


def _extract_price(text: str) -> float:
    normalized = text.replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", normalized)
    if not match:
        raise ValueError(f"Could not parse numeric price from model output: {text!r}")
    return float(match.group())


@app.function(image=image, secrets=secrets, gpu=GPU_TYPE, timeout=1800)
def predict_price(description: str) -> dict[str, Any]:
    if not description or not description.strip():
        raise ValueError("Description must be a non-empty string.")

    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        set_seed,
    )

    if not hasattr(predict_price, "_model_ready"):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto",
        )
        fine_tuned_model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_MODEL,
            revision=ADAPTER_REVISION,
        )

        predict_price._tokenizer = tokenizer
        predict_price._model = fine_tuned_model
        predict_price._model_ready = True
        cold_start = True
    else:
        cold_start = False

    prompt = f"{QUESTION}\n\n{description.strip()}\n\n{PREFIX}"
    tokenizer = predict_price._tokenizer
    model = predict_price._model

    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=5)

    decoded = tokenizer.decode(outputs[0])
    parsed_segment = decoded.split(PREFIX)[-1]
    price = _extract_price(parsed_segment)

    return {
        "ok": True,
        "predicted_price": price,
        "currency": "USD",
        "cold_start": cold_start,
    }


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def price_api(payload: dict[str, Any]) -> dict[str, Any]:
    description = str(payload.get("description", "")).strip()
    if not description:
        return {"ok": False, "error": "Missing 'description' in request body."}

    try:
        return predict_price.remote(description)
    except Exception as exc:  # pragma: no cover - remote path
        return {"ok": False, "error": str(exc)}


@app.local_entrypoint()
def main(description: str = "Apple iPhone 13, 128GB, unlocked, used condition.") -> None:
    result = predict_price.remote(description)
    print(result)
