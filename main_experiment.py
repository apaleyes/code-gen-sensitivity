# for access to Gemini, see https://ai.google.dev/gemini-api/docs/quickstart?lang=python

import os
import time
import json
import datetime

import nlpaug.augmenter.char as nac
from TSED import TSED

from models.dummy import Dummy
from models.gemini import Gemini
from models.openai import OpenAI

prompt_title = "calculator"
original_prompt = "Write a Calculator class in Python. It shall contain common operations, such as addition or multiplication, but also more advanced operations, such as logarithm (of variable bases), factorial, trigonometry."
original_prompt = original_prompt + " " +  "Output only Python code and nothing else. CRITICAL:Do not include any markdown _or_ code block indicators."

#########################
# configuration
#########################
# model = Dummy()
# model = Gemini()
model = OpenAI()
original_code = model.get_code(original_prompt)
typo_percentages = range(0, 105, 5)
n_repeats = 10

# set to 5 to stay within free tier for Gemini
# maybe worth making this a model-level hard-coded setting
timeout = 1


experiment_data = {}
experiment_data["llm_model"] = model.name
experiment_data["prompt_title"] = prompt_title
experiment_data["original_prompt"] = original_prompt
experiment_data["augmentation_method"] = "Keyboard"
experiment_data["parameters"] = {
    "temperature": 0.0,
    "n_repeats": n_repeats,
}
experiment_data["measurements"] = []

for typo_percentage in typo_percentages:
    typo_rate = typo_percentage / 100.0
    augmenter = nac.KeyboardAug(aug_char_p=typo_percentage / 100.0, aug_char_min=0)

    for i in range(n_repeats):
        augmented_prompt = augmenter.augment(original_prompt, n=1)[0]

        new_code = model.get_code(augmented_prompt)

        similarity_score = TSED.Calaulte("python", original_code, new_code, 1.0, 0.8, 1.0)
        print(f"Typo percentage: {typo_percentage}, similarity score: {similarity_score}")

        experiment_data["measurements"].append({
            "n_repeat": i,
            "augmentation_rate": typo_rate,
            "code_similarity": similarity_score
        })
        time.sleep(timeout)

timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
experiment_data["id"] = f"{experiment_data['llm_model']}-{experiment_data['prompt_title']}-{timestamp_now}"

with open(experiment_data["id"] + ".json", 'w') as f:
    json.dump(experiment_data, f, indent=4)
