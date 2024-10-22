# for access to Gemini, see https://ai.google.dev/gemini-api/docs/quickstart?lang=python

import google.generativeai as genai
import os
import time

import nlpaug.augmenter.char as nac
from TSED import TSED

import pandas as pd

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

original_prompt = "Write a Calculator class in Python. It shall contain common operations, such as addition or multiplication, but also more advanced operations, such as logarithm (of variable bases), factorial, trigonometry. Output only Python code and nothing else."

def get_code_dummy(prompt):
    return "def calc(x, y):\n    return x + y"

def get_code(prompt):
    response = model.generate_content(prompt)

    # annoyingly, it seems to always wrap responses in ```python ... ```
    # here is a hacky way to remove these, so that we only deal with code itself
    # it will always be first line and either one or two last lines (it may also add an empty line)
    lines = response.text.split("\n")
    if len(lines[-1]) == 0:
        code_lines = lines[1:-2]
    else:
        code_lines = lines[1:-1]

    code_text = "\n".join(code_lines)
    return code_text

# get_code = get_code_dummy

original_code = get_code(original_prompt)
typo_percentages = range(0, 105, 5)

n_repeats = 20
# Sometimes requests to Gemini seem to fail with a reason completely beyond user's control:
#############
#  ValueError: Invalid operation: The `response.text` quick accessor requires the response to contain a valid `Part`, but none were returned. The candidate's [finish_reason](https://ai.google.dev/api/generate-content#finishreason) is 4. Meaning that the model was reciting from copyrighted material.
################
# it normally works fine on retry, but sometimes multiple retries are necesasary
n_retries = 5
measurements = []

for typo_percentage in typo_percentages:
    typo_rate = typo_percentage / 100.0
    augmenter = nac.KeyboardAug(aug_char_p=typo_percentage / 100.0, aug_char_min=0)

    for _ in range(n_repeats):
        augmented_prompt = augmenter.augment(original_prompt, n=1)[0]

        for attempt in range(n_retries):
            try:
                new_code = get_code(augmented_prompt)
            except ValueError:
                print("Request failed, retrying")
            else:
                break

        similarity_score = TSED.Calaulte("python", original_code, new_code, 1.0, 0.8, 1.0)
        print(f"Typo percentage: {typo_percentage}, similarity score: {similarity_score}")

        measurements.append({
            "method": "KeyboardAug",
            "augmentation_rate": typo_rate,
            "code_similarity": similarity_score
        })
        time.sleep(5)

df = pd.DataFrame(measurements)
df.to_csv("gemini_results.csv", sep=",", header=True, index=False)
