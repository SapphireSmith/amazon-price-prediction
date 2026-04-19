import gradio as gr

from modal_predictor import predict_price as modal_predict
from groq_predictor import predict_price as groq_predict
from predictor.benchmark import create_benchmark_chart

# --- add this import ---
from product_preprocessor import preprocess_description


# --- structured examples (NO actual price sent to models) ---
EXAMPLES = [
    [
        """Title: VONADO Space Wars Colonial Viper MK1 Building Block Model
Category: Building Sets
Brand: VONADO
Description: 604-piece ABS plastic spaceship building kit that allows kids to construct the iconic Space Wars Colonial Viper MK1 fighter.
Details: Features smooth, durable bricks, QR-code instructions, and compatibility with other major building-brick brands for expanded play.""",
        39.99,
    ],
    [
        """Title: Panasonic KX-DECT 6.0 PLUS Digital Cordless Phone
Category: Electronics
Brand: Panasonic
Description: A single‑handset cordless phone with DECT 6.0 PLUS technology for clear, extended-range communication.
Details: Includes a headset jack, non‑slip handset design, and eco‑friendly power‑saving mode.""",
        79.2,
    ],
    [
        """Title: Bam France 2002XL Contoured Hightech 4/4 Violin Case
Category: Musical Instruments
Brand: Bam France
Description: Ultra‑light, shock‑resistant case for 4/4 violins featuring a sleek black carbon‑look exterior and contoured interior for maximum protection.
Details: 3.5‑lb, 30×12×9 in, three‑layer construction (AIREX, ABS, PVC), foam suspension, two bow holders, accessory pouch, and anti‑sk""",
        713.0,
    ],
    [
        """Title: Empava 24" Electric Single Wall Oven
Category: Kitchen Appliances
Brand: Empava
Description: 24‑inch electric single wall oven with 10 cooking functions, built‑in convection, rotisserie, and a touch LED digital display.
Details: Features a 2.3‑cubic‑foot capacity, stainless‑steel finish, 3200‑W power, ETL certification, and a 2‑year limited warranty.
""",
        654.14,
    ],
    [
        """Title: Carbon Fiber Racing Steering Wheel with Nappa Leather for GR Supra A90
Category: Automotive Accessories
Brand: Akozon
Description: A high‑performance steering wheel featuring a carbon fiber core, full Nappa leather finish, and a built‑in OLED display for real‑time data.
Details: Equipped with paddle shifters, wireless OBD‑II connectivity, a flat‑bottomed D‑shaped design, and hand‑stitched leather, it offers speed, RPM, lap, and engine telemetry display in""",
        447.74,
    ],
    [
        """Title: BMW 335i/135i AC Compressor Kit
Category: Automotive Parts
Brand: BuyAutoParts
Description: OEM‑grade AC compressor and component kit for BMW 335i, 335xi, 335is, 135i, and 135is models.
Details: Includes compressor with clutch, drier, expansion valve, PAG oil, and O‑ring seals, built to OE standards with a 2‑year unlimited‑mileage warranty.""",
        314.41,
    ],
    [
        """Title: 4GB DDR2 1066MHz Desktop Memory Kit
Category: Computer Components
Brand: Komputerbay
Description: Dual 2GB DDR2 1066MHz modules in a 4GB kit for desktop systems.
Details: Heatspreaders on each DIMM, 240-pin PC2-8500, dual‑channel support, lifetime warranty.""",
        80.34,
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
                    label="🔵 Frontier Model (groq/llama-3.3-70b-versatile)"
                )

            diff_out = gr.Number(label="📊 Price Difference Between both models")

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
