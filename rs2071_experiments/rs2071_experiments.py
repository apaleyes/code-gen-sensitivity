# import nltk
# nltk.download('averaged_perceptron_tagger_eng')

import json
import datetime
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from TSED import TSED
from models import get_model, ModelCaller
from utils import ensure_python_code_prompt
import code_execute
import visualise
import concurrent.futures

def run_experiment(i, model_name="openai", augmentation_method="synonym", n_repeats=5):
    # LEETCODE DATASET
    with open('leetcode-dataset.json', 'r') as f:
        leet_data = json.load(f)
    leet_item = leet_data[i]
    print(leet_item['slug'], flush=True)
    print(leet_item['question'], flush=True)
    solution = "\n".join(
        line[4:] if line.startswith("    ") else line for line in leet_item['python_solutions'].split("\n")[1:])
    print(solution, flush=True)

    # EXPERIMENT SPECS
    prompt_title = f"leet code {i} {leet_item['slug']} {leet_item['difficulty']}".replace(" ", "-")
    original_prompt = leet_item['question']
    model = get_model(model_name)
    model_caller = ModelCaller(model, prompt_transform=ensure_python_code_prompt)
    original_code = model_caller.get_code(original_prompt)
    all_aug_rates = [x / 100.0 for x in range(0, 105, 5)]
    experiment_data = {
        "llm_model": model.name,
        "prompt_title": prompt_title,
        "original_prompt": original_prompt,
        "augmentation_method": augmentation_method,
        "parameters": {
            "temperature": model.temperature,
            "n_repeats": n_repeats,
        },
        "measurements": []
    }

    # EXPERIMENT RUN
    for aug_rate in all_aug_rates:
        if experiment_data["augmentation_method"].lower() == "keyboard":
            augmenter = nac.KeyboardAug(aug_char_p=aug_rate, aug_word_p=aug_rate, aug_char_min=0, aug_word_max=len(original_prompt))
        elif experiment_data["augmentation_method"].lower() == "synonym":
            augmenter = naw.SynonymAug(aug_p=aug_rate, aug_max=len(original_prompt))
        else:
            raise ValueError(f"Unknown augmentation method {experiment_data['augmentation_method']}")

        augmented_prompts = [augmenter.augment(original_prompt, n=1)[0] for _ in range(n_repeats)]

        # **Parallelize only model_caller.get_code(augmented_prompt)**
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_augmented_prompt = {executor.submit(model_caller.get_code, prompt): prompt for prompt in augmented_prompts}

            for repeat_index, future in enumerate(concurrent.futures.as_completed(future_to_augmented_prompt)):
                try:
                    new_code = future.result()
                    # print(new_code)

                    similarity_score = TSED.Calaulte("python", original_code, new_code, 1.0, 0.8, 1.0)
                    solution_similarity_score = TSED.Calaulte("python", leet_item['python_solutions'], new_code, 1.0, 0.8, 1.0)
                    evaluation_score = code_execute.evaluate_solution(new_code, original_prompt)

                    print(f"Augmentation percentage: {round(aug_rate * 100)}, similarity score: {round(similarity_score,2)}, solution similarity: {round(solution_similarity_score,2)}, code accuracy: {round(evaluation_score,2)}", flush=True)

                    experiment_data["measurements"].append({
                        "n_repeat": repeat_index,
                        "augmentation_rate": aug_rate,
                        "code_similarity": similarity_score,
                        "solution_similarity": solution_similarity_score,
                        "code_accuracy": evaluation_score
                    })
                except Exception as e:
                    print('FAILED', e, flush=True)
                    experiment_data["measurements"].append({
                        "n_repeat": repeat_index,
                        "augmentation_rate": None,
                        "code_similarity": None,
                        "solution_similarity": None,
                        "code_accuracy": None
                    })

    # Save experiment results
    timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_data["id"] = f"{experiment_data['llm_model']}-{experiment_data['prompt_title']}-{experiment_data['augmentation_method']}-{timestamp_now}"

    print(experiment_data, flush=True)
    with open(f'experiments/{experiment_data["id"]}.json', 'w+', encoding='utf-8') as f:
        json.dump(experiment_data, f, indent=4)
    print('done', flush=True)


# for k in range(9, 10):
#     run_experiment(k, 'llama', 'keyboard')
#     visualise.process_experiments('experiments')

aug_rate = 1
original_prompt = "Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.\nThe overall run time complexity should be O(log (m+n)).\n  Example 1:\nInput: nums1 = [1,3], nums2 = [2]\nOutput: 2.00000\nExplanation: merged array = [1,2,3] and median is 2.\nExample 2:\nInput: nums1 = [1,2], nums2 = [3,4]\nOutput: 2.50000\nExplanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.\n  Constraints:\nnums1.length == m\nnums2.length == n\n0 <= m <= 1000\n0 <= n <= 1000\n1 <= m + n <= 2000\n-106 <= nums1[i], nums2[i] <= 106"
augmenter = nac.KeyboardAug(aug_char_p=aug_rate, aug_word_p=aug_rate, aug_char_min=0, aug_word_max=len(original_prompt))
augmented_prompts = [augmenter.augment(original_prompt, n=1)[0] for _ in range(10)]
print(augmented_prompts)
