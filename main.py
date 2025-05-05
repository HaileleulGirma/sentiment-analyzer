import os
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import praw
import re

# Load tokenizer and model
model_name = "AdamCodd/tinybert-sentiment-amazon"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
  # e.g. "http://localhost:8080/authorize_callback"
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Reddit API setup using PRAW
reddit = praw.Reddit(
    client_id=CLIENT_ID,  # Replace with your Reddit API client_id
    client_secret=CLIENT_SECRET,  # Replace with your Reddit API client_secret
    user_agent=USER_AGENT  # Replace with a user agent string
)


# Function to scrape Reddit post text
def get_reddit_post_text(post_url):
    try:
        # Extract post ID from URL
        post_id = re.findall(r"comments/([a-zA-Z0-9]+)/", post_url)[0]
        submission = reddit.submission(id=post_id)

        # Get title and content (selftext)
        title = submission.title
        text = submission.selftext

        return f"Title: {title}\n\nText: {text}"
    except Exception as e:
        return f"Error: Unable to retrieve the post. {str(e)}"


# Gradio interface function for sentiment prediction
def predict_sentiment(post_url):
    # Get Reddit post text using the URL
    post_text = get_reddit_post_text(post_url)

    if "Error" in post_text:  # If there was an issue with the Reddit post retrieval
        return post_text

    # Perform sentiment analysis on the post text
    inputs = tokenizer(post_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs).item()
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        confidence = probs[0][predicted_class].item()

    return f"Sentiment: {sentiment} (Confidence: {round(confidence * 100, 2)}%)\n\n{post_text}"


# Gradio interface setup
interface = gr.Interface(fn=predict_sentiment,
                         inputs=gr.Textbox(lines=2, placeholder="Enter a Reddit post URL..."),
                         outputs="text",
                         title="Sentiment Analyzer for Reddit",
                         description="Enter a Reddit post URL, and this tool will analyze its sentiment using DistilRoBERTa.")

# Launch Gradio interface
interface.launch(share=True, server_name="0.0.0.0", server_port=7860, inbrowser=False)
