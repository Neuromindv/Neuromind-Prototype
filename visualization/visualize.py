import torch
from network import NeuromindNetwork
import json
import http.server
import socketserver

PORT = 8000

class NeuromindHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/network':
            net = NeuromindNetwork()
            net.grow_network(5)  # Symulacja wzrostu
            data = {
                "neurons": net.neurons,
                "rhythm_freq": net.rhythm_freq,
                "output": net(torch.randn(1, 10)).tolist()
            }
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.writelines(json.dumps(data).encode())
        else:
            super().do_GET()

with socketserver.TCPServer(("", PORT), NeuromindHandler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()