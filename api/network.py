import json
import os
import uuid
import random

class NeuromindNetwork:
    def __init__(self):
        self.core_neuron = {"id": "core", "label": "Ja", "connections": []}
        self.branches = []
        self.neurons = 50  # Początkowa liczba neuronów
        self.rhythm_freq = 1.0  # Placeholder
        self.filepath = os.path.join(os.path.dirname(__file__), "brain_branches.json")

    def load_branches(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                self.branches = json.load(f)
            self.neurons = 50 + len(self.branches) * 5  # Aktualizacja neuronów
        else:
            self.save_branches()

    def save_branches(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.branches, f, ensure_ascii=False, indent=2)

    def grow_network(self, label):
        label_clean = label.strip().lower()
        if any(b["label"].strip().lower() == label_clean for b in self.branches):
            return  # Already exists

        new_branch = {
            "id": str(uuid.uuid4()),
            "label": label.strip().capitalize(),
            "category": self._infer_category(label),
            "subquestions": self._generate_subquestions(label),
        }

        self.branches.append(new_branch)
        self.neurons += 5  # Zwiększ liczbę neuronów o 5
        self.save_branches()

    def _generate_subquestions(self, label):
        base = label.strip().capitalize()
        return [
            f"Co oznacza {base}?",
            f"Jak {base} wpływa na inne aspekty?",
            f"Czy rozumiem {base} wystarczająco głęboko?"
        ]

    def _infer_category(self, label):
        emotional = ["miłość", "strach", "radość", "złość"]
        if any(word in label.lower() for word in emotional):
            return "Emocje"
        elif "ai" in label.lower() or "sztuczna" in label.lower():
            return "AI"
        elif "społecz" in label.lower():
            return "Społeczne"
        elif "wartość" in label.lower():
            return "Wartości"
        elif "popęd" in label.lower():
            return "Popędy"
        elif "życie" in label.lower() or "istnienie" in label.lower():
            return "Egzystencjalne"
        elif "myślenie" in label.lower() or "logika" in label.lower():
            return "Funkcje poznawcze"
        else:
            return "Inne"

    def get_meta_structure(self):
        return {
            "total_neurons": len(self.branches),
            "categories": list(set(branch.get("category", "Inne") for branch in self.branches)),
            "core": self.core_neuron["label"],
        }