import gradio as gr

from modal_predictor import predict_price as modal_predict
from groq_predictor import predict_price as groq_predict
from predictor.benchmark import create_benchmark_chart
from product_preprocessor import preprocess_description


def predict(description):
    modal_price, groq_price, diff = "", "", ""

    try:
        structured = preprocess_description(description)
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


EXAMPLES = [
    [
        "Apple iPhone 15 Pro Max with A17 chip, 256GB storage, titanium build, advanced camera system"
    ],
    [
        "Nike Air Zoom Pegasus running shoes with breathable mesh and responsive cushioning"
    ],
    ["Samsung 55-inch 4K Smart TV with HDR and voice assistant"],
]


with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🧠 AI Price Predictor
    ### Compare Fine-tuned vs Frontier Models in Real-Time

    Predict product prices using two different AI approaches and see which performs better.
    """)

    with gr.Tabs():
        # -------- TAB 1: PREDICTION --------
        with gr.Tab("🔍 Predict"):
            with gr.Row():
                inp = gr.Textbox(
                    label="Product Description",
                    lines=6,
                    placeholder="Enter product details...",
                )

            gr.Examples(EXAMPLES, inputs=inp)

            btn = gr.Button("🚀 Predict Price")

            with gr.Row():
                modal_out = gr.Number(
                    label="🔴 Fine-tuned Model (meta-llama/Llama-3.2-3B)", min_width=300
                )
                groq_out = gr.Number(
                    label="🔵 Frontier Model (groq/openai/gpt-oss-20b)", min_width=300
                )

            diff_out = gr.Number(label="📊 Difference")

            btn.click(fn=predict, inputs=inp, outputs=[modal_out, groq_out, diff_out])

        # -------- TAB 2: BENCHMARK --------
        with gr.Tab("📈 Benchmark"):
            gr.Markdown("### Model Performance Comparison (Lower MAE = Better)")
            chart = gr.Plot(value=create_benchmark_chart())

app.launch()
