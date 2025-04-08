from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import os

app = FastAPI()

# Hugging Face model settings
SUMMARY_MODEL = "philschmid/bart-large-cnn-samsum"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
HF_TOKEN = os.getenv("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# Hugging Face API caller
def query_hf(model: str, inputs: str):
    url = f"https://api-inference.huggingface.co/models/{model}"
    payload = {"inputs": inputs}
    response = requests.post(url, headers=HEADERS, json=payload)
    try:
        return response.json()
    except Exception:
        return {"error": "Invalid JSON response", "raw": response.text}

# Input models
class Turn(BaseModel):
    speaker: str
    text: str

class Transcript(BaseModel):
    transcript: List[Turn]

# Main endpoint
@app.post("/analyze")
async def analyze_call(data: Transcript):
    try:
        turns = data.transcript
        dialogue = "\n".join([f"{t.speaker}: {t.text}" for t in turns])

        # 🔎 Summarization
        summary_response = query_hf(SUMMARY_MODEL, dialogue)
        if isinstance(summary_response, list) and "summary_text" in summary_response[0]:
            summary_text = summary_response[0]["summary_text"]
        else:
            return {"error": "Summarization failed", "details": summary_response}

        # 🧠 Sentiment analysis
        sentiment_results = []
        for turn in turns:
            sentiment_raw = query_hf(SENTIMENT_MODEL, turn.text)

            try:
                top_label = None
                top_score = 0.0
                if isinstance(sentiment_raw, list):
                    raw = sentiment_raw[0] if isinstance(sentiment_raw[0], list) else sentiment_raw
                    top = max(raw, key=lambda x: x["score"])
                    top_label = top["label"]
                    top_score = top["score"]

                sentiment_results.append({
                    "speaker": turn.speaker,
                    "text": turn.text,
                    "sentiment": top_label,
                    "confidence": round(top_score, 3)
                })

            except Exception as e:
                sentiment_results.append({
                    "speaker": turn.speaker,
                    "text": turn.text,
                    "sentiment": "UNKNOWN",
                    "confidence": 0.0,
                    "details": sentiment_raw
                })

        return {
            "summary": summary_text,
            "sentiment_analysis": sentiment_results
        }

    except Exception as e:
        return {
            "error": "Something went wrong in /analyze",
            "message": str(e)
        }
