# for access to Gemini, see https://ai.google.dev/gemini-api/docs/quickstart?lang=python
# for TSED, see test_manual_paraphrasings.py
# for nlpaug, see augmentation_test.py

import google.generativeai as genai
import os
import time

import nlpaug.augmenter.char as nac
from TSED import TSED

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

original_prompt = "Write a Calculator class in Python. It shall contain common operations, such as addition or multiplication, but also more advanced operations, such as logarithm (of variable bases), factorial, trigonometry. Output only Python code and nothing else."

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

original_code = get_code(original_prompt)
typo_percentages = range(0, 100, 5)
distances = []

for typo_percentage in typo_percentages:
    augmenter = nac.KeyboardAug(aug_char_p=typo_percentage / 100.0, aug_char_min=0)
    augmented_prompt = augmenter.augment(original_prompt, n=1)[0]
    new_code = get_code(augmented_prompt)
    similarity_score = TSED.Calaulte("python", original_code, new_code, 1.0, 0.8, 1.0)
    print(f"Typo percentage: {typo_percentage}, similarity score: {similarity_score}")
    time.sleep(5)
    break # this plus is here to avoid accidental requests