# predictor/benchmark.py

import plotly.graph_objects as go

BENCHMARK_RESULTS = [
    # ("Constant", "baseline", 106.18),
    # ("Linear Regression", "baseline", 101.56),
    # ("NLP + LR", "baseline", 76.81),
    # ("Random Forest", "baseline", 72.28),
    # ("XGBoost", "baseline", 68.23),
    # ("Human (Ed)", "human", 87.62),
    # ("Neural Network", "baseline", 63.97),
    ("GPT 4.1 Nano", "frontier", 62.51),
    ("Grok 4.1 Fast", "frontier", 57.62),
    ("Gemini 3 Pro", "frontier", 50.54),
    ("Claude 4.5 Sonnet", "frontier", 47.10),
    ("GPT 5.1", "frontier", 44.74),
    ("GPT 4.1 Nano (Fine-tuned)", "baseline", 75.91),
    # ("Deep Neural Network", "baseline", 46.49),
    ("Base Llama 3.2 4 bit", "baseline", 110.72),
    ("Fine-tuned Lite", "your_model", 65.40),
    ("Fine-tuned Full", "your_model", 39.85),
]


def _get_color(category):
    return {
        "your_model": "red",
        "frontier": "blue",
        "human": "black",
        "baseline": "gray",
    }.get(category, "gray")


def create_benchmark_chart():
    labels = [x[0] for x in BENCHMARK_RESULTS]
    values = [x[2] for x in BENCHMARK_RESULTS]
    colors = [_get_color(x[1]) for x in BENCHMARK_RESULTS]

    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))

    fig.update_layout(
        title="Model Comparison (MAE ↓)",
        xaxis=dict(tickangle=-45),
        yaxis_title="Error",
        margin=dict(b=120),  # allow space for rotated x-axis labels
        height=600,  # allow height to be responsive to width if needed, but keeping height to maintain aspect ratio
    )

    return fig
