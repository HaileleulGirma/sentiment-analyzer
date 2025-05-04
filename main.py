# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = FastAPI()

# Load pretrained sentiment model
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

labels = ["negative", "neutral", "positive"]

class TextIn(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextIn):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return {
        "label": labels[pred],
        "confidence": round(probs[0][pred].item(), 4)
    }
