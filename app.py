import gradio as gr

from modal_predictor import predict_price as modal_predict
from groq_predictor import predict_price as groq_predict
from predictor.benchmark import create_benchmark_chart

# --- add this import ---
from product_preprocessor import preprocess_description


# --- structured examples (NO actual price sent to models) ---
EXAMPLES = [
    [
        """Title: STARMOON DOT-Certified Full Face Helmet
Category: Motorcycle Helmets
Brand: STARMOON
Description: A lightweight, matte black full-face helmet offering DOT-approved protection with a built-in transparent sun visor.
Details: Features an ABS shell, high-density ventilated liner, aerodynamic design, and UV-resistant topcoat for enhanced comfort and safety.""",
        145.00,
    ],
    [
        """Title: Compact Portable Foot Rest
Category: Accessories
Brand: EWH
Description: A fold-able foot rest that elevates one foot for back and hip relief while standing or sitting.
Details: Made from durable ABS, it measures 4in high, folds to 8½×4×¾ inches, and fits easily in a purse or pocket.""",
        30.00,
    ],
    [
        """Title: Inwalltech M525.1LCR 5 1/4" In-Wall LCR Speaker
Category: Home Theater Audio
Brand: Inwalltech
Description: A 5 1/4" in-wall LCR speaker delivering 125 W of powerful, well-balanced sound for home theater use.
Details: Features lightweight gold Apolymer-cone woofers, silk-dome tweeters for clear highs, easy drop-in installation, and paint-matchable grill for a discreet setup.""",
        139.00,
    ],
    [
        """Title: Rockford Fosgate P600X4 600-Watt 4-Channel Amplifier
Category: Audio Amplifiers
Brand: Rockford Fosgate
Description: Powerful 4-channel amplifier delivering up to 600 W RMS for a full-range car audio system.
Details: Features Class A/B circuitry, versatile 4- or bridged-channel operation, RCA inputs, onboard EQ with 45 Hz boost, and 1-year warranty.""",
        339.00,
    ],
    [
        """Title: Camoo Adjustable Gland Nut Wrench
Category: Tools & Equipment
Brand: Camoo
Description: An adjustable wrench for hydraulic gland nuts from 1" to 3-3/4" with interchangeable 1/4" or 7/32" pin holes.
Details: Features a 1/2" square drive, corrosion-resistant steel construction, and anti-wear design for heavy-duty use on construction and farm equipment.""",
        18.00,
    ],
    [
        """Title: Braven BRV-X/2 Rugged Waterproof Bluetooth Speaker
Category: Electronics
Brand: Braven
Description: A rugged, waterproof Bluetooth speaker that delivers powerful bass and crystal-clear highs for outdoor adventures.
Details: 20 W output, 18-hour battery life, USB charging port, and IPX7 waterproof rating.""",
        138.00,
    ],
    [
        """Title: Powerbuilt 2.5-Ton Low-Profile Fast Lift Floor Jack
Category: Automotive Tools
Brand: Powerbuilt
Description: A compact 2.5-ton floor jack that lifts up to 19.5 inches with a 3-inch clearance for low-profile vehicles.
Details: Features a dual-piston foot pump for quick, precise lifts, durable construction, and meets ASME standards with a one-year warranty.""",
        300.00,
    ],
]


# # --- detect if already structured ---
# def is_structured(text: str) -> bool:
#     keys = ["Title:", "Category:", "Brand:", "Description:"]
#     return all(k in text for k in keys)


# --- new helper ---
def get_actual_price(description):
    for desc, price in EXAMPLES:
        if description.strip() == desc.strip():
            return price
    return None


# --- prediction function ---
def predict(description):
    modal_price, groq_price, diff = "", "", ""

    try:
        # if is_structured(description):
        #     structured = description
        # else:
        structured = preprocess_description(description)
        print(f"structured={structured}")
    except Exception as e:
        return f"Error: {e}", f"Error: {e}", "N/A"

    try:
        modal_price = float(modal_predict(structured))
    except Exception as e:
        modal_price = f"Error: {e}"

    try:
        groq_price = float(groq_predict(structured))
    except Exception as e:
        groq_price = f"Error: {e}"

    if isinstance(modal_price, float) and isinstance(groq_price, float):
        diff = round(abs(modal_price - groq_price), 2)
    else:
        diff = "N/A"

    return modal_price, groq_price, diff


# --- UI changes ---
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🧠 AI Price Predictor
    ### Compare Fine-tuned vs Frontier Models in Real-Time

    Predict product prices using two different AI approaches and see which performs better.
    """)

    with gr.Tabs():
        with gr.Tab("🔍 Predict"):
            inp = gr.Textbox(label="Product Description", lines=8)

            example_inputs = [[x[0]] for x in EXAMPLES]
            gr.Examples(example_inputs, inputs=inp)

            actual_out = gr.Number(label="💰 Actual Price")

            # trigger actual price when example selected / input changes
            inp.change(fn=get_actual_price, inputs=inp, outputs=actual_out)

            btn = gr.Button("Predict Price")

            with gr.Row():
                modal_out = gr.Number(
                    label="🔴 Fine-tuned Model (meta-llama/Llama-3.2-3B)"
                )
                groq_out = gr.Number(
                    label="🔵 Frontier Model (groq/openai/gpt-oss-20b)"
                )

            diff_out = gr.Number(label="📊 Price Difference")

            btn.click(
                fn=predict,
                inputs=inp,
                outputs=[modal_out, groq_out, diff_out],
            )

        # -------- TAB 2: BENCHMARK --------
        with gr.Tab("📈 Benchmark"):
            gr.Markdown("### Model Performance Comparison (Lower MAE = Better)")
            chart = gr.Plot(value=create_benchmark_chart())

app.launch()
