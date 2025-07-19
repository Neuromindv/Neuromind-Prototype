import torch
from network import NeuromindNetwork
import json
import http.server
import socketserver
import os

PORT = 8000
WEB_DIR = os.path.join(os.path.dirname(__file__), "")
BRAIN_FILE = "brain_branches.json"

# Globalny singleton dla sieci
class NeuromindServer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NeuromindServer, cls).__new__(cls)
            cls._instance.net = NeuromindNetwork()
            cls._instance.net.load_branches()
        return cls._instance

# Serwer z dostępem do `net`
class CustomTCPServer(socketserver.TCPServer):
    def __init__(self, server_address, RequestHandlerClass, bind_and_activate=True):
        self.net = NeuromindServer().net  # Przekazanie singletona
        super().__init__(server_address, RequestHandlerClass, bind_and_activate=bind_and_activate)

class NeuromindHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        if self.path == '/network':
            data = {
                "neurons": self.server.net.neurons,
                "rhythm_freq": self.server.net.rhythm_freq,
                "core": self.server.net.core_neuron["label"],
                "questions": ["What knowledge do I need?", "In which directions should I grow?"]
            }
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/answer':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            answer = data.get("answer", "Unknown").lower().strip()
            # Sprawdzenie duplikatów
            is_duplicate = any(branch["label"].lower() == f"response: {answer}" for branch in self.server.net.branches)
            if not is_duplicate:
                self.server.net.grow_network(5, f"Response: {answer.capitalize()}")
                self.server.net.save_branches()  # Zapis stanu
            # Pobierz ostatnie podpytania (tylko dla nowych gałęzi)
            subquestions = self.server.net.branches[-1].get("subquestions", []) if not is_duplicate else []
            response = {
                "branch": f"Response: {answer.capitalize()}",
                "neurons": self.server.net.neurons,
                "subquestions": subquestions
            }
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            super().do_GET()

if __name__ == "__main__":
    with CustomTCPServer(("", PORT), NeuromindHandler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()