# 🚀 TinyBERT Sentiment Analyzer

This project is a simple web app and REST API for performing sentiment analysis using a lightweight BERT model, ideal for low-latency environments. It's powered by:

- 🤗 Transformers (`AdamCodd/tinybert-sentiment-amazon`)
- 🧠 PyTorch
- 🌐 FastAPI (REST API backend)
- 🎛️ Gradio (interactive web UI)
- ☁️ Deployed on Render

---

## 🔍 Demo

🌐 **Live Gradio App**: [https://your-service.onrender.com/gradio](https://your-service.onrender.com/gradio)  
🛠 **API Endpoint**: `POST https://your-service.onrender.com/predict`

---

## 🧪 Example Input & Output

### Request (API)

```json
POST /predict
Content-Type: application/json

{
  "text": "This product is amazing! I love it."
}
