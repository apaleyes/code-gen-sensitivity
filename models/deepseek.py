import os

import openai  # OpenRouter uses OpenAI's API format

from .base import BaseModel


class DeepSeek(BaseModel):
    def __init__(self):
        super().__init__("DeepSeek")
        self.client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",  # Explicitly route to OpenRouter
        )
        self.model = "deepseek/deepseek-r1:free"

        # Set temperature similar to Gemini
        self.temperature = 0.0

    def __call__(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content
