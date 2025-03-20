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
        from .chatgpt import OpenAI

        return OpenAI()

    if model_name.lower() == "claude":
        from .claude import Claude

        return Claude()
    
    if model_name.lower() == "llama":
        from .llama import Llama
        return Llama()

    if model_name.lower() == "deepseek":
        from .deepseek import DeepSeek
        return DeepSeek
    
    raise ValueError(f"Unknown model name {model_name}")
