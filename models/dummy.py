from .base import BaseModel

class Dummy(BaseModel):
    def __init__(self):
        super().__init__("Dummy")
    
    def get_code(self, prompt):
        return "def calc(x, y):\n    return x + y"
