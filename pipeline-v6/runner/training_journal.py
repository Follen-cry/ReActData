"""Per-step streams with checkpoint-aligned truncation; no training/model dependencies."""
import os
import json

class TrainingJournal:
    def __init__(self, root):
        self.root, self.files = root, {}

    def open(self, name, step):
        path = os.path.join(self.root, name)
        if os.path.exists(path):
            with open(path) as f:
                keep = [line for line in f if json.loads(line)["step"] < step]
            with open(path, "w") as f:
                f.writelines(keep)
        self.files[name] = open(path, "a")

    def write(self, name, record):
        self.files[name].write(json.dumps(record, ensure_ascii=False) + "\n")

    def flush(self):
        for stream in self.files.values():
            stream.flush()

    def close(self):
        for stream in self.files.values():
            stream.close()
