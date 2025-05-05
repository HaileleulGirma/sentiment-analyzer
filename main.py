import os
import re
import torch
import gradio as gr
import praw
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, redirect

# === Load Model ===
model_name = "AdamCodd/tinybert-sentiment-amazon"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# === Flask Setup for OAuth ===
app = Flask(__name__)

CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDDIT_REDIRECT_URI")  # e.g. "http://localhost:8080/authorize_callback"
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    user_agent=USER_AGENT
)

# Store user's Reddit session globally (not secure for production)
authenticated_reddit = {}

@app.route("/")
def login():
    auth_url = reddit.auth.url(["identity", "read"], state="random123", duration="permanent")
    return f"<a href='{auth_url}'>Login with Reddit</a>"

@app.route("/authorize_callback")
def callback():
    code = request.args.get("code")
    refresh_token = reddit.auth.authorize(code)
    # Save Reddit instance with refresh token
    authenticated_reddit["session"] = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        refresh_token=refresh_token,
        user_agent=USER_AGENT
    )
    return redirect("/gradio")

# === Gradio Sentiment Interface ===
def get_reddit_post_text(post_url):
    try:
        post_id = re.findall(r"comments/([a-zA-Z0-9]+)/", post_url)[0]
        session = authenticated_reddit.get("session")
        if not session:
            return "Error: You must log in with Reddit first."
        submission = session.submission(id=post_id)
        return f"Title: {submission.title}\n\nText: {submission.selftext}"
    except Exception as e:
        return f"Error: {str(e)}"

def predict_sentiment(post_url):
    post_text = get_reddit_post_text(post_url)
    if "Error" in post_text:
        return post_text
    inputs = tokenizer(post_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs).item()
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        confidence = probs[0][predicted_class].item()
    return f"Sentiment: {sentiment} (Confidence: {round(confidence * 100, 2)}%)\n\n{post_text}"

@app.route("/gradio")
def launch_gradio():
    interface = gr.Interface(
        fn=predict_sentiment,
        inputs=gr.Textbox(lines=2, placeholder="Enter a Reddit post URL..."),
        outputs="text",
        title="Sentiment Analyzer for Reddit",
        description="Analyze Reddit post sentiment after logging in."
    )
    interface.launch(share=False, server_name="0.0.0.0", server_port=7860, inbrowser=True)
    return "Launching Gradio..."
