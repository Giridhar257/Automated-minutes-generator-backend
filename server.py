# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# import os
# from generator import generate_minutes_from_file

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/generate-minutes/")
# async def generate_minutes(
#     file: UploadFile = File(...),
#     participants: str = Form("")
# ):
#     file_path = f"temp_{file.filename}"
#     with open(file_path, "wb") as f:
#         f.write(await file.read())

#     participants_list = [p.strip() for p in participants.split(",") if p.strip()]

#     try:
#         result = generate_minutes_from_file(file_path, participants_list)
#     finally:
#         os.remove(file_path)

#     return result
# server.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
from datetime import datetime
import os
from generator import (
    read_meeting_file,
    load_summarizer,
    summarize_text,
    extract_action_items,
    format_minutes,
)

app = FastAPI()


origins = [
    "https://automated-minutes-generator-5.vercel.app",
    "http://localhost:3000",  # optional for local testing
]
# Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # ⚠️ replace with your frontend URL (e.g. http://localhost:3000) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate-minutes/")
async def generate_minutes(
    file: UploadFile = File(...),
    participants: str = Form(""),
    summarizer_model: str = Form("facebook/bart-large-cnn"),
    max_len: int = Form(180),
    min_len: int = Form(30),
):
    """
    Endpoint to generate meeting minutes from an uploaded file.
    Accepts TXT, PDF, MP3, WAV.
    """

    # Save uploaded file temporarily
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())

    try:
        # Read transcript from the file (handles txt/pdf/audio)
        transcript = read_meeting_file(temp_file_path)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)  # Clean up temp file

    # Load summarizer
    summarizer = load_summarizer(summarizer_model)

    # Summarize transcript
    summary = summarize_text(
        summarizer,
        transcript,
        max_length=max_len,
        min_length=min_len
    )

    # Extract action items
    actions = extract_action_items(transcript)

    # Process participants string → list
    participants_list = [p.strip() for p in participants.split(",") if p.strip()]

    # Generate formatted minutes text
    title = datetime.now().strftime("%Y-%m-%d %H:%M")
    minutes_text = format_minutes(title, summary, actions, participants_list)

    # ✅ Return JSON response for frontend
    return JSONResponse(content={
        "title": title,
        "summary": summary,
        "participants": participants_list,
        "minutes": minutes_text,
        "actions": actions,
    })
