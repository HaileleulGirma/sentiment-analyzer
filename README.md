# TinyBERT Sentiment Analyzer 🧠💬

A lightweight, cloud-deployable web app for binary sentiment classification using the `AdamCodd/bert-tiny` transformer model from Hugging Face. Built with Gradio, containerized with Docker, and deployed on Render.com.

---

## 🚀 Project Overview

This project demonstrates how a distilled transformer model can be used to perform sentiment analysis on user input text, delivering fast and accurate results with minimal computational overhead.

- 🔍 **Model**: `AdamCodd/bert-tiny` from Hugging Face
- 🌐 **UI & Backend**: Gradio
- 🐳 **Containerization**: Docker
- ☁️ **Deployment**: Render.com

---

## 🎯 Objectives

- ✅ Build a real-time, user-friendly sentiment analysis app
- ✅ Optimize for low latency and memory usage
- ✅ Deploy to a free-tier cloud platform
- ✅ Ensure reproducibility using Docker
- ✅ Showcase cloud-native NLP microservice design

---

## 🧰 Tech Stack

| Component     | Tool                     |
|--------------|--------------------------|
| NLP Model     | Hugging Face Transformers |
| UI & Backend | Gradio                   |
| Deployment   | Render.com               |
| Container    | Docker                   |
| Lang         | Python 3.10+             |

---

## 🏗️ System Architecture

1. User inputs a sentence in the Gradio web interface.
2. The sentence is tokenized using the model's tokenizer.
3. The `bert-tiny` model returns a sentiment prediction.
4. Confidence score is calculated and shown on the page.


---

## 📦 Installation

### Prerequisites

- Python 3.10+
- Docker (optional for containerized runs)

### Clone the repo

```
