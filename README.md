---
title: Amazon Price Predictor
emoji: 📈
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 5.23.0
app_file: app.py
pinned: false
---

# 🧠 AI Price Predictor: Fine-Tuned vs. Frontier Model Comparison

Welcome to the **AI Price Predictor**, a state-of-the-art comparison platform designed to evaluate the performance of specialized fine-tuned LLMs against massive frontier models in the context of Amazon product pricing.

This project demonstrates how a smaller, specifically trained model (Llama 3.2-3B) can compete with—and sometimes outperform—generic "frontier" models (Llama 3.3-70B) when tasked with niche domain knowledge like product valuation.

---

## ✨ Core Features

### 🔍 Real-Time Price Comparison
Input any Amazon product description and get instantaneous price predictions from two distinct AI architectures:
*   **The Specialist:** A fine-tuned `Llama-3.2-3B` model optimized for Amazon product data, hosted on **Modal**.
*   **The Generalist:** A `Llama-3.3-70B-Versatile` frontier model accessed via **Groq** for high-speed inference.

### 🤖 Intelligent Preprocessing Agent
Raw product descriptions are often messy and filled with irrelevant noise (part numbers, shipping info, etc.). Our integrated **Preprocessing Agent** uses LiteLLM to:
*   Reconstruct raw text into a structured format (Title, Brand, Category, Features).
*   Remove data noise to ensure the prediction models receive clean, high-signal input.
*   Improve prediction accuracy by focusing only on value-driving product attributes.

### 📈 Interactive Performance Benchmarking
The app includes a dedicated **Benchmark** tab that visualizes:
*   **Model Comparison (MAE ↓):** A live Plotly chart comparing the Mean Absolute Error of different models.
*   **Error Trends:** Visual analysis of how models perform across different price brackets.
*   **Prediction vs. Actual:** Scatter plots showcasing the correlation between AI guesses and real Amazon prices.

### 🎨 Premium Gradio Interface
A sleek, responsive UI built with Gradio's modern themes, featuring:
*   **Example Gallery:** Quick-load curated examples to see the models in action.
*   **Real-Time Difference Calculation:** Instant calculation of the delta between both models.
*   **Visual Feedback:** Clear, color-coded indicators for different model outputs.

---

## 🏗️ Architecture Overview

The project follows a modular "Agent-Inference" architecture designed to minimize noise and maximize prediction accuracy.

### 1. The Preprocessing Layer (LiteLLM + Agent)
Before any prediction occurs, the raw user input (often a messy copy-paste from an Amazon page) is passed to the **Preprocessing Agent** (`agents/preprocessor.py`). 
*   **Role:** Acts as a data cleaner and structurer.
*   **Logic:** Uses a dedicated system prompt to extract relevant features (Title, Brand, Category, Features) and discard irrelevant text (ASINs, shipping notes, etc.).
*   **Engine:** Powered by **LiteLLM**, allowing for easy swapping of the underlying model (defaulting to Llama 3.2 or GPT-4o-mini).

### 2. The Dual Inference Engine
Once structured, the cleaned description is sent simultaneously to two distinct endpoints:

#### **🔴 The Specialist (Fine-Tuned Llama 3.2-3B)**
*   **Hosting:** Deployed as a serverless function on **Modal** (`modal_predictor.py`).
*   **Training:** This model was fine-tuned on a curated dataset of Amazon products to understand the specific pricing correlations between product attributes and historical market values.
*   **Advantage:** Low latency and high precision for its specific domain, despite being a much smaller model.

#### **🔵 The Generalist (Frontier Llama 3.3-70B)**
*   **Hosting:** Accessed via the **Groq** API (`groq_predictor.py`).
*   **Capability:** A massive "Frontier" model that relies on its vast general knowledge and zero-shot reasoning to estimate prices.
*   **Advantage:** Serves as the gold standard baseline to evaluate whether fine-tuning a smaller model can bridge the performance gap with massive LLMs.

### 3. The Presentation Layer (Gradio)
The results are orchestrated in `app.py`, which handles:
*   **Asynchronous Calls:** Fetching predictions from both Modal and Groq.
*   **Comparison Logic:** Calculating the price difference in real-time.
*   **Visualization:** Rendering the Plotly benchmark charts from the `predictor/benchmark.py` module.

---

## 🚀 Setup & Installation

Follow these steps to get the AI Price Predictor running locally.

### 1. Prerequisites
*   **Python 3.10+**
*   **Groq API Key:** Get it from the [Groq Console](https://console.groq.com/).
*   **Modal Account:** Required if you want to deploy or call the fine-tuned model. Set up at [modal.com](https://modal.com/).

### 2. Clone & Install
```bash
# Clone the repository
git clone https://github.com/SapphireSmith/amazon-price-prediction.git
cd amazon-price-prediction

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory by copying the `example.env` file:
```bash
cp example.env .env
```
Fill in the following essential variables:
*   `GROQ_API_KEY`: Your personal Groq API key.
*   `PRICER_PREPROCESSOR_MODEL`: The model used for structuring descriptions (e.g., `groq/llama-3.3-70b-versatile`).
*   `MODAL_PRICE_API_URL`: The endpoint URL for your deployed Modal fine-tuned model.
*   *Note: Other variables in `example.env` are related to Phase 1 (Fine-tuning) and are only necessary if you are re-training the model.*

