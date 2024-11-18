# for access to Gemini, see https://ai.google.dev/gemini-api/docs/quickstart?lang=python

import os
from openai import OpenAI as OpenAIClient

from .base import BaseModel

class OpenAI(BaseModel):
    def __init__(self):
        super().__init__("OpenAI")
        self.client = OpenAIClient()

    def get_code(self, prompt):
        # possible values for temperature range from 0 to 2
        # default is 1
        # we use 0 to minimise variation
        temperature = 0.0

        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a coding assistant."},
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0
        )

        response_message = completion.choices[0].message

        # if it happens to wrap responses in ```python ... ```
        # here is a hacky way to remove these, so that we only deal with code itself
        # it will always be first line and either one or two last lines (it may also add an empty line)
        lines = response_message.content.split("\n")
        if len(lines[-1]) == 0:
            lines = lines[:-1]
        
        if "```" in lines[0]:
            lines = lines[1:]
        
        if "```" in lines[-1]:
            lines = lines[:-1]

        code_text = "\n".join(lines)
        return code_text
