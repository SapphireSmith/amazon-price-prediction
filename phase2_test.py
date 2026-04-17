import math

from groq_predictor import predict_price as predict_price_groq
from modal_predictor import predict_price as predict_price_modal
from product_preprocessor import preprocess_description


def _assert_good_price(price: float, which: str) -> float:
    if not isinstance(price, float) or not math.isfinite(price):
        raise AssertionError(f"{which} returned non-finite/non-float price: {price!r}")
    if price <= 0:
        raise AssertionError(f"{which} returned non-positive price: {price!r}")
    return price


def main() -> None:
    sample = (
        """Sleek design meets powerful performance with the iPhone 17, featuring a stunning 120Hz display, A19 chip, and upgraded dual-camera system for smooth everyday use.
        A perfect balance of premium feel and performance, making it Apple's most refined base model yet."""
    )
    structured = preprocess_description(sample)

    groq_price = predict_price_groq(structured)
    groq_price = _assert_good_price(float(groq_price), "groq")

    modal_price = predict_price_modal(structured)
    modal_price = _assert_good_price(float(modal_price), "modal")

    print(f"groq_price={groq_price}")
    print(f"modal_price={modal_price}")


if __name__ == "__main__":
    main()

