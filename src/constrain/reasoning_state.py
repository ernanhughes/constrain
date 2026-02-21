from typing import List

class ReasoningState:
    def __init__(self, prompt):
        self.prompt = prompt
        self.history = []
        self.current = prompt
        self.temperature = None

    def accept(self, reasoning):
        self.history.append(reasoning)
        self.current = reasoning

    def revert(self):
        if self.history:
            self.current = self.history[-1]

    def reset(self):
        self.history = []
        self.current = self.prompt
