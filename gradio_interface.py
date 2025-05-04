# gradio_interface.py

import gradio as gr
import requests

# FastAPI server URL
api_url = "http://localhost:8000/predict"  # If deployed on Render, use that URL here

def predict_sentiment(text):
    response = requests.post(api_url, json={"text": text})
    result = response.json()
    return result["label"], result["confidence"]

# Gradio interface
iface = gr.Interface(fn=predict_sentiment,
                     inputs="text",
                     outputs=["text", "number"],
                     title="Real-Time Sentiment Analysis",
                     description="Enter any text to analyze its sentiment as positive, neutral, or negative.")

iface.launch()
