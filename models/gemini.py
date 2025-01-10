# for access to Gemini, see https://ai.google.dev/gemini-api/docs/quickstart?lang=python

import os
import google.generativeai as genai

from .base import BaseModel

class Gemini(BaseModel):
    def __init__(self):
        super().__init__("Gemini")
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.client = genai.GenerativeModel("gemini-1.5-flash")

        # possible values for temperature range from 0 to 2
        # default is 1
        # we use 0 to minimise variation
        self.temperature = 0.0

    def make_model_call(self, prompt):
        response = self.client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
            )
        )

        return response.text

    @property
    def call_timeout(self):
        # This is to stay within free tier for Gemini
        return 5
