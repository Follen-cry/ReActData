"""Small transport doubles shared by CPU tests."""

class Slot:
    def __init__(self, name, node, gpus, port):
        self.name, self.node, self.gpus, self.port, self.job = name, node, gpus, port, None

    def to_json(self):
        return {"name": self.name, "node": self.node, "gpus": self.gpus, "port": self.port}
