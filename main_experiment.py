import time
import json
import datetime

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from TSED import TSED

from models import get_model

model_name = "dummy" # possible values 'openai', 'gemini', 'dummy'
augmentation_method = "synonym" # possible values 'keyboard', 'synonym'
prompt_title = "calculator"
original_prompt = "Write a Calculator class. It shall contain common operations, such as addition or multiplication, but also more advanced operations, such as logarithm (of variable bases), factorial, trigonometry."
def get_full_prompt(prompt):
    prefix = "Write Python code."
    postfix = "Output only Python code and nothing else. CRITICAL:Do not include any markdown _or_ code block indicators."
    return prefix + prompt + postfix

#########################
# configuration
#########################
model = get_model(model_name)
original_code = model.get_code(get_full_prompt(original_prompt))
all_aug_rates = [x / 100.0 for x in range(0, 105, 5)] 
n_repeats = 1


experiment_data = {}
experiment_data["llm_model"] = model.name
experiment_data["prompt_title"] = prompt_title
experiment_data["original_prompt"] = original_prompt
experiment_data["augmentation_method"] = augmentation_method
experiment_data["parameters"] = {
    "temperature": model.temperature,
    "n_repeats": n_repeats,
}
experiment_data["measurements"] = []

for aug_rate in all_aug_rates:
    if experiment_data["augmentation_method"].lower() == "keyboard":
        # work out a proper way to calculate these rates
        # aug_char_p and aug_word_p in combination should give aug_rate
        augmenter = nac.KeyboardAug(aug_char_p=aug_rate, aug_word_p=aug_rate, aug_char_min=0, aug_word_max=len(original_prompt))
    elif experiment_data["augmentation_method"].lower() == "synonym":
        augmenter = naw.SynonymAug(aug_p=aug_rate, aug_max=len(original_prompt))
    else:
        raise ValueError(f"Unknown augmentation method {experiment_data['augmentation_method']}")

    for i in range(n_repeats):
        augmented_prompt = augmenter.augment(original_prompt, n=1)[0]

        new_code = model.get_code(get_full_prompt(augmented_prompt))


        similarity_score = TSED.Calaulte("python", original_code, new_code, 1.0, 0.8, 1.0)
        print(f"Augmentation percentage: {aug_rate * 100}, similarity score: {similarity_score}")

        experiment_data["measurements"].append({
            "n_repeat": i,
            "augmentation_rate": aug_rate,
            "code_similarity": similarity_score
        })
        time.sleep(model.call_timeout)

timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
experiment_data["id"] = f"{experiment_data['llm_model']}-{experiment_data['prompt_title']}-{experiment_data['augmentation_method']}-{timestamp_now}"

with open(experiment_data["id"] + ".json", 'w') as f:
    json.dump(experiment_data, f, indent=4)
