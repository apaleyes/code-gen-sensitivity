import time

from .base import BaseModel

def sanitise_response(response_string):
    # annoyingly, some LLMs seems wrap responses in ```python ... ```
    # here is a hacky way to remove these, so that we only deal with code itself
    # it will always be first line and either one or two last lines (it may also add an empty line)
    lines = response_string.split("\n")
    if len(lines[-1]) == 0:
        lines = lines[:-1]

    if "```" in lines[0]:
        lines = lines[1:]
    
    if "```" in lines[-1]:
        lines = lines[:-1]

    code_string = "\n".join(lines)
    return code_string

def is_valid_python_code(code_string):
    # verify that response from LLM is a valid Python code
    import ast

    try:
        ast.parse(code_string, filename='<unknown>', mode='exec')
    except (SyntaxError, ValueError) as _:
        return False

    return True

class ModelCaller:
    def __init__(self, model: BaseModel):
        self.model = model

        # Sometimes requests to Gemini seem to fail with a reason completely beyond user's control.
        ################
        #  ValueError: Invalid operation: The `response.text` quick accessor requires the response to contain a valid `Part`, but none were returned. The candidate's [finish_reason](https://ai.google.dev/api/generate-content#finishreason) is 4. Meaning that the model was reciting from copyrighted material.
        ################
        # Also LLMs may occasionally output invalid code
        # To account for it, we want to retry a few times
        # it normally works fine on first retry, but sometimes multiple retries are necesasary
        self.n_retries = 10

    def get_code(self, prompt):
        # see note in __init__ on retries
        for _ in range(self.n_retries):
            try:
                # if model requires delay between calls,
                # e.g. to stay under certain call rate,
                # do it right before we made a call
                time.sleep(self.model.call_timeout)
                response_string = self.model(prompt)
            except ValueError:
                print("Request failed, retrying")
            else:
                code_string = sanitise_response(response_string)
                if is_valid_python_code(code_string):
                    # no more retries necessary
                    # break out of the retry loop
                    break
                else:
                    print("Request failed, retrying")
        else:
            raise ValueError(f"Failed to get valid code from {self.name}")

        return code_string
