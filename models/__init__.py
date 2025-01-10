from .base import BaseModel
from .model_caller import ModelCaller

def get_model(model_name):
    if model_name.lower() == "dummy":
        from .dummy import Dummy
        return Dummy()

    if model_name.lower() == "gemini":
        from .gemini import Gemini
        return Gemini()

    if model_name.lower() == "openai":
        from .openai import OpenAI
        return OpenAI()

    raise ValueError(f"Unknown model name {model_name}")
