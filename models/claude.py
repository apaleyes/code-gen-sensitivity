# some docs are here: https://github.com/anthropics/anthropic-sdk-python

import os
from anthropic import Anthropic

from .base import BaseModel


class Claude(BaseModel):
    def __init__(self):
        super().__init__("Claude")
        self.client = Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

        # possible values for temperature range from 0 to 1
        # default is 1
        # we use 0 to minimise variation
        self.temperature = 0.0

    def __call__(self, prompt):
        message = self.client.messages.create(
            # max declared for 3 Haiku model, see https://docs.anthropic.com/en/docs/about-claude/models
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="claude-3-haiku-20240307",
        )

        return message.content[0].text

    @property
    def call_timeout(self):
        return 1


if __name__ == "__main__":
    # to try this model, go to repo's root and run
    # python -m models.claude
    model = Claude()
    output = model("Write Python code to sum two numbers.")
    print(output)
