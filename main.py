from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import os

app = FastAPI()

# Hugging Face API endpoints & headers
SUMMARY_MODEL = "philschmid/bart-large-cnn-samsum"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_hf(model: str, inputs: str):
    url = f"https://api-inference.huggingface.co/models/{model}"
    payload = {"inputs": inputs}
    response = requests.post(url, headers=HEADERS, json=payload)
    try:
        return response.json()
    except:
        return {"error": "Invalid JSON response", "raw": response.text}

# Input models
class Turn(BaseModel):
    speaker: str
    text: str

class Transcript(BaseModel):
    transcript: List[Turn]

@app.post("/analyze")
async def analyze_call(data: Transcript):
    turns = data.transcript
    dialogue = "\n".join([f"{t.speaker}: {t.text}" for t in turns])
    
    # 🧠 Run summarization
    summary_response = query_hf(SUMMARY_MODEL, dialogue)
    summary_text = summary_response[0]["summary_text"] if isinstance(summary_response, list) else "Summarization failed"

    # 💬 Run sentiment per turn
    sentiment_results = []
    for turn in turns:
        sentiment_raw = query_hf(SENTIMENT_MODEL, turn.text)
        if isinstance(sentiment_raw, list):
            top = sentiment_raw[0]
            sentiment_results.append({
                "speaker": turn.speaker,
                "text": turn.text,
                "sentiment": top["label"],
                "confidence": round(top["score"], 3)
            })
        else:
            sentiment_results.append({
                "speaker": turn.speaker,
                "text": turn.text,
                "sentiment": "UNKNOWN",
                "confidence": 0.0
            })

    return {
        "summary": summary_text,
        "sentiment_analysis": sentiment_results
    }
