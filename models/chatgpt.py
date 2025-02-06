# renamed this file because it used to be same as import below, and that confused python
from openai import OpenAI as OpenAIClient
from .base import BaseModel


class OpenAI(BaseModel):
    def __init__(self):
        super().__init__("OpenAI")
        self.client = OpenAIClient()

        # possible values for temperature range from 0 to 2
        # default is 1
        # we use 0 to minimise variation
        self.temperature = 0.0

    def __call__(self, prompt):
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a coding assistant."},
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.temperature
        )

        response_message = completion.choices[0].message
        return response_message.content

    @property
    def call_timeout(self):
        return 1
