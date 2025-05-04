from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import gradio as gr
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.responses import RedirectResponse

app = FastAPI()

# CORS settings for Render (optional, but helpful for testing from frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer
model_name = "AdamCodd/tinybert-sentiment-amazon"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

class SentimentRequest(BaseModel):
    text: str

# Prediction function
def predict_sentiment_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    label = model.config.id2label[pred]
    confidence = round(probs[0][pred].item(), 4)
    return f"Label: {label}, Confidence: {confidence}"

# REST API endpoint
@app.post("/predict")
def predict_sentiment(request: SentimentRequest):
    return predict_sentiment_text(request.text)

# Gradio interface
gradio_interface = gr.Interface(
    fn=predict_sentiment_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter text..."),
    outputs="text",
    title="TinyBERT Sentiment Classifier"
)

# Mount Gradio app at /gradio
@app.get("/", response_class=HTMLResponse)
def root():
    return RedirectResponse("/gradio")

from fastapi.responses import Response
from starlette.middleware.wsgi import WSGIMiddleware

gradio_app = gradio_interface.server_app()
app.mount("/gradio", WSGIMiddleware(gradio_app))
