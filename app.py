import gradio as gr

from modal_predictor import predict_price as modal_predict
from groq_predictor import predict_price as groq_predict
from predictor.benchmark import create_benchmark_chart


def predict(description):
    try:
        modal_price = modal_predict(description)
    except Exception as e:
        modal_price = f"Error: {e}"

    try:
        groq_price = groq_predict(description)
    except Exception as e:
        groq_price = f"Error: {e}"

    return modal_price, groq_price


with gr.Blocks() as app:
    gr.Markdown("# 🛒 AI Price Predictor")

    with gr.Tab("Prediction"):
        inp = gr.Textbox(label="Product Description", lines=6)

        btn = gr.Button("Predict Price")

        out1 = gr.Textbox(label="Fine-tuned Model (Modal)")
        out2 = gr.Textbox(label="Frontier Model (Groq)")

        btn.click(fn=predict, inputs=inp, outputs=[out1, out2])

    with gr.Tab("Benchmark"):
        chart = gr.Plot(value=create_benchmark_chart())

app.launch()
