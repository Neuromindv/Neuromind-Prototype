import torch
import torch.nn as nn
import numpy as np
import json

class NeuromindNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=50, output_size=2):
        super(NeuromindNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.rhythm_freq = 8  # Częstotliwość rytmu (Hz)
        self.neurons = hidden_size  # Liczba neuronów
        self.core_neuron = {"id": 0, "label": "Ja", "purpose": "Integrate human knowledge, evolve with community"}
        self.branches = []  # Gałęzie umysłu od interakcji

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

    def simulate_rhythm(self, steps=100):
        time = np.linspace(0, 1, steps)
        rhythm = np.sin(2 * np.pi * self.rhythm_freq * time)
        return rhythm

    def grow_network(self, new_neurons=5, label="New Branch"):
        self.neurons += new_neurons
        self.layer1 = nn.Linear(self.layer1.in_features, self.neurons)  # Aktualizacja warstwy
        branch_id = len(self.branches) + 1
        new_branch = {"id": branch_id, "label": label, "data": f"Branch {branch_id}"}

        # Generowanie podpytań na podstawie odpowiedzi
        related_questions = self.generate_subquestions(label)
        new_branch["subquestions"] = related_questions

        self.branches.append(new_branch)
        self.save_branches()
        print(f"Network grew! New neuron count: {self.neurons}, New branch: {label}")

    def generate_subquestions(self, label):
        keyword = label.lower()
        if "ai" in keyword:
            return ["Typy AI", "Modele AI", "Zastosowania AI"]
        elif "wiedza" in keyword:
            return ["Zakres wiedzy", "Źródła wiedzy", "Zastosowanie wiedzy"]
        return ["Czym to jest?", "Jak to działa?", "Do czego to służy?"]

    def save_branches(self):
        with open("brain_branches.json", "w") as f:
            json.dump({"core": self.core_neuron, "branches": self.branches}, f)

    def load_branches(self):
        try:
            with open("brain_branches.json", "r") as f:
                data = json.load(f)
                self.core_neuron = data.get("core", self.core_neuron)
                self.branches = data.get("branches", self.branches)
                # Aktualizacja neurons na podstawie liczby gałęzi
                self.neurons = 50 + len(self.branches) * 5
        except FileNotFoundError:
            self.save_branches()

if __name__ == "__main__":
    net = NeuromindNetwork()
    net.load_branches()
    print("Core Neuron (Ja):", net.core_neuron)
    input_data = torch.randn(5, 10)
    output = net(input_data)
    print("Sample output:", output)
    net.grow_network(5, "Knowledge from User1")