import json
import os
from datetime import datetime, timedelta
import random

STATE_FILE = "api/state.json"

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"question": "", "start_time": "", "answers": []}
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def generate_question():
    return random.choice([
        "Co naprawdę napędza ludzką ciekawość?",
        "Jaka cecha najbardziej definiuje świadomość?",
        "Jakie są granice sztucznej inteligencji?",
        "Czym jest prawdziwa wolna wola?",
    ])

def get_current_question():
    state = load_state()
    now = datetime.utcnow()

    # Brak pytania lub minęła godzina
    if not state["question"] or not state["start_time"] or \
       datetime.fromisoformat(state["start_time"]) + timedelta(hours=1) <= now:
        state["question"] = generate_question()
        state["start_time"] = now.isoformat()
        state["answers"] = []
        save_state(state)

    return state

def add_answer(text, author="anonymous"):
    state = get_current_question()
    state["answers"].append({
        "text": text,
        "author": author,
        "votes": 0
    })
    save_state(state)

def vote_answer(index):
    state = get_current_question()
    if 0 <= index < len(state["answers"]):
        state["answers"][index]["votes"] += 1
        save_state(state)

def get_winning_answer():
    state = get_current_question()
    if not state["answers"]:
        return None
    return max(state["answers"], key=lambda a: a["votes"])
