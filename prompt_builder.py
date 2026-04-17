QUESTION = "What does this cost to the nearest dollar?"
PREFIX = "Price is $"


def build_prompt(structured_product_text: str) -> str:
    structured_product_text = (structured_product_text or "").strip()
    if not structured_product_text:
        raise ValueError("structured_product_text must be a non-empty string.")
    return f"{QUESTION}\n\n{structured_product_text}\n\n{PREFIX}"

