import math

from groq_predictor import predict_price as predict_price_groq
from modal_predictor import predict_price as predict_price_modal


def _assert_good_price(price: float, which: str) -> float:
    if not isinstance(price, float) or not math.isfinite(price):
        raise AssertionError(f"{which} returned non-finite/non-float price: {price!r}")
    if price <= 0:
        raise AssertionError(f"{which} returned non-positive price: {price!r}")
    return price


def main() -> None:
    sample = (
        "Quadcast HyperX USB-C Condenser Mic connects via usb-c to your computer for crystal clear audio"
    )

    groq_price = predict_price_groq(sample)
    groq_price = _assert_good_price(float(groq_price), "groq")

    modal_price = predict_price_modal(sample)
    modal_price = _assert_good_price(float(modal_price), "modal")

    print(f"groq_price={groq_price}")
    print(f"modal_price={modal_price}")


if __name__ == "__main__":
    main()

