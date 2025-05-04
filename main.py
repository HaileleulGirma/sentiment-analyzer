from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import gradio as gr
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from gradio.routes import mount_gradio_app

app = FastAPI()

# CORS (optional)
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

# Data model
class SentimentRequest(BaseModel):
    text: str

# Inference
def predict_sentiment_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    label = model.config.id2label[pred]
    confidence = round(probs[0][pred].item(), 4)
    return f"Label: {label}, Confidence: {confidence}"

# REST endpoint
@app.post("/predict")
def predict_sentiment(request: SentimentRequest):
    return predict_sentiment_text(request.text)

# Gradio Interface
gradio_interface = gr.Interface(
    fn=predict_sentiment_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
    outputs="text",
    title="TinyBERT Sentiment Analyzer",
)

# Mount Gradio UI at /gradio
@app.get("/", response_class=HTMLResponse)
def root():
    return RedirectResponse("/gradio")

mount_gradio_app(app, gradio_interface, path="/gradio")
