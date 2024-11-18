# for access to Gemini, see https://ai.google.dev/gemini-api/docs/quickstart?lang=python

import os
import google.generativeai as genai

from .base import BaseModel

class Gemini(BaseModel):
    def __init__(self):
        super().__init__("Gemini")
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel("gemini-1.5-flash")

        # Sometimes requests to Gemini seem to fail with a reason completely beyond user's control:
        #############
        #  ValueError: Invalid operation: The `response.text` quick accessor requires the response to contain a valid `Part`, but none were returned. The candidate's [finish_reason](https://ai.google.dev/api/generate-content#finishreason) is 4. Meaning that the model was reciting from copyrighted material.
        ################
        # it normally works fine on retry, but sometimes multiple retries are necesasary
        self.n_retries = 10
    
    def get_code(self, prompt):
        # possible values for temperature range from 0 to 2
        # default is 1
        # we use 0 to minimise variation
        temperature = 0.0

        # see note in __init__ on retries
        for _ in range(self.n_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                    )
                )
            except ValueError:
                print("Request failed, retrying")
            else:
                break
        

        # annoyingly, Gemini seems to always wrap responses in ```python ... ```
        # here is a hacky way to remove these, so that we only deal with code itself
        # it will always be first line and either one or two last lines (it may also add an empty line)
        lines = response.text.split("\n")
        if len(lines[-1]) == 0:
            lines = lines[:-1]
        
        if "```" in lines[0]:
            lines = lines[1:]
        
        if "```" in lines[-1]:
            lines = lines[:-1]

        code_text = "\n".join(lines)
        return code_text
