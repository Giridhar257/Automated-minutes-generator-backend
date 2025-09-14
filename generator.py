from datetime import datetime
from typing import List, Dict, Optional
import os
import spacy
from pydub import AudioSegment
import whisper
from transformers import pipeline
import subprocess   
import sys

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Install the model if missing
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# ---------- File Readers ----------
def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def mp3_to_wav(mp3_path, wav_path):
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")

def transcribe_audio(file_path, model_name="small"):
    model = whisper.load_model(model_name)
    result = model.transcribe(file_path)
    return result.get("text", "")

def read_meeting_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return read_txt(file_path)
    elif ext in [".wav", ".mp3"]:
        if ext == ".mp3":
            temp_wav = "temp.wav"
            mp3_to_wav(file_path, temp_wav)
            text = transcribe_audio(temp_wav)
            os.remove(temp_wav)
        else:
            text = transcribe_audio(file_path)
        return text
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ---------- Summarizer ----------
def load_summarizer(model_name="facebook/bart-large-cnn"):
    return pipeline("summarization", model=model_name, framework="pt")

def summarize_text(summarizer, text, max_length=180, min_length=30):
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]["summary_text"]

# ---------- Action Item Extraction ----------
# def extract_action_items(text: str) -> List[Dict[str, Optional[str]]]:
#     actions = []
#     doc = nlp(text)
#     for sent in doc.sents:
#         sent_text = sent.text.strip()
#         if any(word.lower_ in ["will", "shall", "should", "need", "must", "ensure"] for word in sent):
#             task = sent_text
#             person = None
#             deadline = None
#             for ent in sent.ents:
#                 if ent.label_ == "PERSON":
#                     person = ent.text
#                 if ent.label_ == "DATE":
#                     deadline = ent.text
#             actions.append({"task": task, "person": person, "deadline": deadline})
#     return actions
def extract_action_items(text: str | list) -> List[Dict[str, Optional[str]]]:
    # If text is a list (e.g., from PDF lines), join into a single string
    if isinstance(text, list):
        text = " ".join([str(t) for t in text if t])

    actions = []
    doc = nlp(text)

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if any(word.lower_ in ["will", "shall", "should", "need", "must", "ensure"] for word in sent):
            task = sent_text
            person = None
            deadline = None
            for ent in sent.ents:
                if ent.label_ == "PERSON":
                    person = ent.text
                if ent.label_ == "DATE":
                    deadline = ent.text
            actions.append({"task": task, "person": person, "deadline": deadline})
    
    return actions


# ---------- Minutes Formatter ----------
def format_minutes(title: str, summary: str, action_items: List[Dict[str, Optional[str]]], participants: List[str]):
    minutes = f"Meeting Title: {title}\n"
    minutes += f"Participants: {', '.join(participants)}\n\n"
    minutes += "Summary:\n" + summary + "\n\n"
    if action_items:
        minutes += "Action Items:\n"
        for a in action_items:
            task = a.get("task", "")
            person = a.get("person", "")
            deadline = a.get("deadline", "")
            minutes += f"- Task: {task} | Person: {person} | Deadline: {deadline}\n"
    return minutes

def generate_minutes_from_file(file_path: str, participants: list,
                               summarizer_model="facebook/bart-large-cnn",
                               max_len=180, min_len=30):
    """
    High-level function: takes a file path + participants,
    and returns full minutes, summary, and action items.
    """
    # Step 1: Read transcript
    txt = read_meeting_file(file_path)

    # Step 2: Summarize
    summarizer = load_summarizer(summarizer_model)
    summary = summarize_text(summarizer, txt, max_length=max_len, min_length=min_len)

    # Step 3: Action items
    actions = extract_action_items(txt)

    # Step 4: Format minutes
    title = datetime.now().strftime("%Y-%m-%d %H:%M")
    minutes = format_minutes(title, summary, actions, participants)

    return {
        "title": title,
        "summary": summary,
        "participants": participants,
        "minutes": minutes,
        "actions": actions
    }
