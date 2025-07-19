import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Prosta sieć neuronowa inspirowana ludzkim mózgiem
class NeuromindNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=2):
        super(NeuromindNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.rhythm_freq = 8  # Początkowa częstotliwość rytmu (Hz, np. alfa)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

    def simulate_rhythm(self, steps=100):
        # Symulacja rytmów mózgowych (pulsowanie 4-100 Hz)
        time = np.linspace(0, 1, steps)
        rhythm = np.sin(2 * np.pi * self.rhythm_freq * time)
        return rhythm

# Test sieci
if __name__ == "__main__":
    # Utwórz sieć
    net = NeuromindNetwork()
    print("Neuromind Network created with:", net)

    # Generuj dane testowe
    input_data = torch.randn(5, 10)  # 5 próbek, 10 cech
    output = net(input_data)
    print("Sample output:", output)

    # Symuluj rytm i wizualizuj
    rhythm = net.simulate_rhythm()
    plt.plot(rhythm)
    plt.title("Rhythm of Neuromind (8 Hz)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()