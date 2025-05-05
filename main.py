import os
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import praw
import re


model_name = "AdamCodd/tinybert-sentiment-amazon"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
  
USER_AGENT = os.getenv("REDDIT_USER_AGENT")


reddit = praw.Reddit(
    client_id=CLIENT_ID,  
    client_secret=CLIENT_SECRET,  
    user_agent=USER_AGENT  
)



def get_reddit_post_text(post_url):
    try:
        
        post_id = re.findall(r"comments/([a-zA-Z0-9]+)/", post_url)[0]
        submission = reddit.submission(id=post_id)

        
        title = submission.title
        text = submission.selftext

        return f"Title: {title}\n\nText: {text}"
    except Exception as e:
        return f"Error: Unable to retrieve the post. {str(e)}"



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



interface = gr.Interface(fn=predict_sentiment,
                         inputs=gr.Textbox(lines=2, placeholder="Enter a Reddit post URL..."),
                         outputs="text",
                         title="Sentiment Analyzer for Reddit",
                         description="Enter a Reddit post URL, and this tool will analyze its sentiment using TinyBERT.")


interface.launch(share=True, server_name="0.0.0.0", server_port=7860, inbrowser=False)
