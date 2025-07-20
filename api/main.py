from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.scheduler import RoundManager

app = FastAPI()
rounds = RoundManager()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sieć neuronowa + pytania
@app.get("/network")
async def get_network():
    return {
        "neurons": rounds.net.neurons,
        "rhythm_freq": rounds.net.rhythm_freq,
        "core": rounds.net.core_neuron["label"],
        "branches": rounds.net.branches,
    }

# Dane rundy (pytanie, odpowiedzi, czas, zwycięzca)
@app.get("/round")
async def get_current_round():
    return rounds.get_round_data()

# Dodaj odpowiedź
@app.post("/answer")
async def submit_answer(request: Request):
    body = await request.json()
    user = body.get("user", "Anonim")
    content = body.get("answer", "").strip()

    if not content:
        return {"error": "Pusta odpowiedź"}

    rounds.submit_answer(user, content)
    return {"message": "Odpowiedź dodana"}

# Głosowanie z user_id
@app.post("/vote")
async def vote_answer(request: Request):
    body = await request.json()
    answer_id = body.get("answer_id")
    user_id = body.get("user_id")

    if rounds.vote(answer_id, user_id):
        return {"message": "Głos dodany"}
    return {"error": "Nie można zagłosować ponownie lub odpowiedź nie istnieje"}

# Zwycięzca rundy
@app.get("/round/winner")
async def get_round_winner():
    winner = rounds.get_winner()
    return {"winner": winner} if winner else {"message": "Jeszcze brak zwycięzcy"}

# Endpoint dla zapytań AI
@app.post("/ask")
async def ask_ai(request: Request):
    body = await request.json()
    question = body.get("question", "").strip()
    if not question:
        return {"error": "Brak pytania"}

    # Symulacja odpowiedzi AI (zastąpimy Puter.js w przyszłości, jeśli zmienisz zdanie)
    response = {"answer": f"AI odpowiada: {question} (przykładowa odpowiedź)"}
    rounds.submit_ai_response(response["answer"])  # Tworzy neuron z odpowiedzią AI
    return response