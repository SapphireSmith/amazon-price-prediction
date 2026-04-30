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