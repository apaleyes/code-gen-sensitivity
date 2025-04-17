import datetime
import json

from augmenters import get_augmenter
from models import ModelCaller, get_model
from TSED import TSED
from code_utils import ensure_python_code_prompt

#########################
# experiment configuration
# TODO: this should probably be passed in as arguments or json file
#########################
model_name = "dummy"  # possible values 'openai', 'gemini', 'dummy', 'claude'
augmentation_method = "keyboard"  # possible values 'keyboard', 'synonym'
prompt_title = "calculator"
original_prompt = "Write a Calculator class. It shall contain common operations, such as addition or multiplication, but also more advanced operations, such as logarithm (of variable bases), factorial, trigonometry."
n_repeats = 2


model = get_model(model_name)
model_caller = ModelCaller(model, prompt_transform=ensure_python_code_prompt)
original_code = model_caller.get_code(original_prompt)
all_aug_rates = [x / 100.0 for x in range(0, 105, 5)]


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
    augmenter = get_augmenter(augmentation_method, aug_rate)

    for i in range(n_repeats):
        augmented_prompt = augmenter.augment(original_prompt)

        new_code = model_caller.get_code(augmented_prompt)

        similarity_score = TSED.Calaulte(
            "python", original_code, new_code, 1.0, 0.8, 1.0
        )
        print(f"Augmentation rate: {aug_rate}, similarity score: {similarity_score}")

        experiment_data["measurements"].append(
            {
                "n_repeat": i,
                "augmentation_rate": aug_rate,
                "code_similarity": similarity_score,
            }
        )


timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
experiment_data[
    "id"
] = f"{experiment_data['llm_model']}-{experiment_data['prompt_title']}-{experiment_data['augmentation_method']}-{timestamp_now}"

with open(experiment_data["id"] + ".json", "w") as f:
    json.dump(experiment_data, f, indent=4)
