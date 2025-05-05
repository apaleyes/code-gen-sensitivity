import os
import json
from augmenters import get_augmenter
# import helper

datasets_dir = "experimental_setup/augmented_datasets"
output_dir = "experimental_setup/augmented_datasets"
os.makedirs(output_dir, exist_ok=True)

json_files = [f for f in os.listdir(datasets_dir) if f.endswith(".json")]

aug_methods = {
    # "keyboard": [x / 20.0 for x in range(0, 21)],
    # "synonym": [x / 20.0 for x in range(0, 21)],
    "paraphraser": [0.0] + [(x / 5 + 0.1) for x in range(0, 5)]
}

for filename in json_files:
    filepath = os.path.join(datasets_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    augmented_data = []
    for item in data:
        prompt = item["question"]
        augmented_versions = item.get("augmented_questions", {})

        for method in aug_methods:
            aug_rates = aug_methods[method]
            augmented_versions[method] = {}
            for rate in aug_rates:
                augmenter_kwargs = {"text_len": len(prompt), "paraphrases_file": "datasets/paraphrases.csv"}
                augmenter = get_augmenter(method, rate, **augmenter_kwargs)
                augmented_prompt = augmenter.augment(prompt)
                augmented_versions[method][str(round(rate, 2))] = augmented_prompt

        item["augmented_questions"] = augmented_versions
        augmented_data.append(item)

    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(augmented_data, out_f, indent=2)
