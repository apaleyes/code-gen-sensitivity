import os
import json
import helper

datasets_dir = "datasets"
output_dir = "augmented_datasets"
os.makedirs(output_dir, exist_ok=True)

json_files = [f for f in os.listdir(datasets_dir) if f.endswith(".json")]
aug_methods = ["keyboard", "synonym"]
aug_rates = [x / 20.0 for x in range(0, 21)]  # 0.0 to 1.0, step 0.05

for filename in json_files:
    filepath = os.path.join(datasets_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    augmented_data = []
    for item in data:
        prompt = item.get("question", "")
        augmented_versions = {}

        for method in aug_methods:
            augmented_versions[method] = {}
            for rate in aug_rates:
                augmenter = helper.get_augmenter(method, rate, len(prompt))
                aug_prompts = helper.generate_augmented_prompts(prompt, augmenter, 1)
                augmented_versions[method][str(round(rate, 2))] = aug_prompts[0]

        item["augmented_questions"] = augmented_versions
        augmented_data.append(item)

    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(augmented_data, out_f, indent=2)
