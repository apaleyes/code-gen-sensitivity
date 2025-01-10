from .base import BaseModel

class Dummy(BaseModel):
    def __init__(self):
        super().__init__("Dummy")
        self.temperature = 0.0

    def __call__(self, prompt):
        return "def calc(x, y):\n    return x + y"
