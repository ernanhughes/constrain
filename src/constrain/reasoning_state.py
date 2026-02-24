from constrain.config import get_config

class ReasoningState:
    def __init__(self, prompt, temperature: float | None = None):
        cfg = get_config()

        self.prompt = prompt
        self.history = []
        self.current = prompt
        self.temperature = (
            temperature if temperature is not None
            else cfg.initial_temperature
        )

    def accept(self, reasoning):
        self.history.append(reasoning)
        self.current = reasoning

    def revert(self):
        if self.history:
            self.current = self.history[-1]

    def reset(self):
        self.history = []
        self.current = self.prompt