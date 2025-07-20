import threading
import time
from datetime import datetime, timedelta
from api.network import NeuromindNetwork
import uuid

class RoundManager:
    def __init__(self):
        self.current_question = self._generate_question()
        self.start_time = datetime.now()
        self.answers = []
        self.voted_users = set()
        self.winner = None
        self.net = NeuromindNetwork()
        self.net.load_branches()
        self._start_timer()

    def _generate_question(self):
        sample = [
            "Co sprawia, że coś jest prawdziwe?",
            "Czy AI może mieć świadomość?",
            "Jakie emocje kierują ludźmi najczęściej?",
            "Czym jest cel w życiu?",
            "Jak rozpoznać dobrą decyzję?"
        ]
        return {
            "id": str(uuid.uuid4()),
            "text": sample[int(time.time()) % len(sample)],
            "timestamp": datetime.now().isoformat()
        }

    def _start_timer(self):
        def run():
            while True:
                now = datetime.now()
                if now - self.start_time >= timedelta(hours=1):
                    self._end_round()
                    self._new_round()
                time.sleep(10)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def _end_round(self):
        if self.answers:
            self.winner = max(self.answers, key=lambda a: a["votes"])
            self.net.grow_network(self.winner["content"])
            self.net.save_branches()

    def _new_round(self):
        self.answers.clear()
        self.voted_users.clear()
        self.current_question = self._generate_question()
        self.start_time = datetime.now()
        self.winner = None

    def submit_answer(self, user, content):
        self.answers.append({
            "id": str(uuid.uuid4()),
            "user": user,
            "content": content.strip(),
            "votes": 0
        })

    def vote(self, answer_id, user_id):
        if not user_id or user_id in self.voted_users:
            return False
        for answer in self.answers:
            if answer["id"] == answer_id:
                answer["votes"] += 1
                self.voted_users.add(user_id)
                return True
        return False

    def submit_ai_response(self, content):
        # Tworzy neuron na podstawie odpowiedzi AI
        self.net.grow_network(content)
        self.net.save_branches()

    def get_round_data(self):
        return {
            "question": self.current_question,
            "answers": self.answers,
            "winner": self.winner,
            "time_left_seconds": max(0, int(3600 - (datetime.now() - self.start_time).total_seconds()))
        }

    def get_winner(self):
        return self.winner