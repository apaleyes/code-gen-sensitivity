import os
import openai  # OpenRouter uses OpenAI's API format
import ollama  # For local inference
import subprocess
from .base import BaseModel


class Llama(BaseModel):
    def __init__(self):
        super().__init__("Llama")
        self.client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1"  # Explicitly route to OpenRouter
        )
        self.model = "meta-llama/llama-3.3-70b-instruct"

        # Set temperature similar to Gemini
        self.temperature = 0.0

    def __call__(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    @property
    def call_timeout(self):
        return 5


class LocalLlama(BaseModel):
    def __init__(self):
        super().__init__("LocalLlama")
        self.model = "llama3"  # Default model

        # Ensure Ollama is installed and the model is available
        self.ensure_model_downloaded()

    def ensure_model_downloaded(self):
        """Check if the model is available, otherwise download it."""
        try:
            # List all available models
            available_models = ollama.list()
            model_names = [model["name"] for model in available_models["models"]]

            # If model is missing, download it
            if self.model not in model_names:
                print(f"Model '{self.model}' not found. Downloading...")
                subprocess.run(["ollama", "pull", self.model], check=True, shell=True)
                print(f"Model '{self.model}' downloaded successfully.")
            else:
                print(f"Model '{self.model}' is already available.")
        except Exception as e:
            print(f"Error checking or downloading model: {e}")

    def __call__(self, prompt):
        """Generate a response from the Llama model."""
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    @property
    def call_timeout(self):
        return 5


def install_ollama():
    # this is only designed for Linux/macOS
    # alternatively https://ollama.ai/download
    try:
        subprocess.run(["curl", "-fsSL", "https://ollama.ai/install.sh", "|", "sh"], check=True, shell=True)
        print("Ollama installed successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to install Ollama:", e)
