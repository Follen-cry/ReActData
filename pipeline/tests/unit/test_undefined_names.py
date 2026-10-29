"""Static check: no undefined names in the trusted implementation (regression: tasks/reactdata/gpu_probes.py used
``self.collator`` and ``crc`` after being moved out of DynTrainer; the CPU fakes never reach that GPU path)."""
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PACKAGES = ("runner", "tasks", "experiment", "api", "scorers")


class UndefinedNames(unittest.TestCase):
    def test_no_undefined_names(self):
        try:
            from pyflakes import api, reporter, messages
        except ImportError:
            self.skipTest("pyflakes not installed")
        bad = []

        class Collect(reporter.Reporter):
            def __init__(self):
                pass

            def flake(self, m):
                if isinstance(m, (messages.UndefinedName, messages.UndefinedLocal)):
                    bad.append(str(m))

            def unexpectedError(self, filename, msg):
                bad.append(f"{filename}: {msg}")

            def syntaxError(self, filename, msg, lineno, offset, text):
                bad.append(f"{filename}:{lineno}: {msg}")

        rep = Collect()
        for pkg in PACKAGES:
            for d, _, files in os.walk(os.path.join(ROOT, pkg)):
                for f in files:
                    if f.endswith(".py"):
                        api.checkPath(os.path.join(d, f), rep)
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
