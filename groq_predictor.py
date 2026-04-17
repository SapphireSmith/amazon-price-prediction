import os
import re

from litellm import completion

from prompt_builder import build_prompt


def _extract_price(text: str) -> float:
    """
    Extract numeric price from text that is expected to include `Price is $<number>`.
    """
    normalized = (text or "").replace(",", "")

    # Prefer the exact contract we asked for.
    # Example: `Price is $180` or `Price is $ 180`
    m = re.search(r"Price is\s*\$\s*([-+]?\d+)", normalized)
    if m:
        return float(m.group(1))

    # Fallback: extract the first number we see.
    match = re.search(r"[-+]?\d*\.\d+|\d+", normalized)
    if not match:
        raise ValueError(f"Could not parse numeric price from response: {text!r}")
    return float(match.group())


def predict_price(structured_description: str) -> float:
    model_name = os.getenv("PRICER_PREPROCESSOR_MODEL")
    if not model_name:
        raise ValueError("Missing PRICER_PREPROCESSOR_MODEL in environment.")

    if not structured_description or not structured_description.strip():
        raise ValueError("structured_description must be a non-empty string.")

    prompt = build_prompt(structured_description.strip())

    # 3) Ask Groq for a strictly formatted answer.
    messages = [
        {
            "role": "system",
            "content": (
                "You are a price predictor. Output EXACTLY one line: Price is $<number>."
                " Do not include any other text, punctuation, or currency words."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    resp = completion(
        messages=messages,
        model=model_name,
        temperature=0,
        # Disable extra reasoning for the actual price request so content isn't empty/truncated.
        reasoning_effort=None,
        max_tokens=128,
    )

    choice = resp.choices[0]
    content = ""
    # litellm response shape can vary by provider; try common fields.
    if getattr(choice, "message", None) is not None:
        content = getattr(choice.message, "content", None) or ""
    if not content.strip() and getattr(choice, "text", None) is not None:
        content = getattr(choice, "text", None) or ""
    content = (content or "").strip()

    # 4) Parse numeric price from the most informative available text.
    # Sometimes `message.content` is empty and the model's text appears in `choice.reasoning`.
    reasoning = (
        getattr(choice, "reasoning", None)
        or getattr(getattr(choice, "message", None), "reasoning", None)
        or ""
    )
    text_to_parse = (content or "").strip() or str(reasoning).strip()
    if not text_to_parse:
        raise ValueError(f"Groq completion returned no usable text. Response: {resp!r}")

    return _extract_price(text_to_parse)

