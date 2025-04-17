import json
import os

def call_with_retry(model_caller, prompt, retries=3, timeout=30):
    time.sleep(random.uniform(0, 3))
    for attempt in range(retries):
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(model_caller.get_code, prompt)
                return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            if attempt == retries - 1:
                return f"ERROR: Timeout after {retries} attempts"
            time.sleep(1)
        except Exception as e:
            return f"ERROR: {str(e)}"


def get_output_path(output_base, model_name, method, filename, item_idx):
    name = f"Q{item_idx:05d}.json"
    return os.path.join(output_base, model_name, method, filename, name)


def load_existing_responses(path, model_name, method, rate_str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)
        return item.get("llm_responses", {}).get(model_name, {}).get(method, {}).get(rate_str, [])
    except Exception:
        return []


def save_response(output_path, item, model_name, method, rate_str, responses, n_repeats):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            item_copy = json.load(f)
    else:
        item_copy = dict(item)
        item_copy["llm_responses"] = {}

    item_copy.setdefault("llm_responses", {}).setdefault(model_name, {}).setdefault(method, {})[rate_str] = responses[:n_repeats]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(item_copy, f, indent=2)



def migrate_old_responses(old_dir, new_dir, model_names):
    for filename in os.listdir(old_dir):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(old_dir, filename), "r", encoding="utf-8") as f:
            data = json.load(f)

        for item_idx, item in enumerate(data):
            llm_responses = item.get("llm_responses", {})
            for model_name in model_names:
                if model_name not in llm_responses:
                    continue

                for method, rate_dict in llm_responses[model_name].items():
                    for rate_str, responses in rate_dict.items():
                        if not isinstance(responses, list) or len(responses) == 0:
                            continue

                        output_path = get_output_path(new_dir, model_name, method, filename, item_idx)
                        existing = load_existing_responses(output_path, model_name, method, rate_str)

                        if isinstance(existing, list) and len(existing) >= len(responses):
                            continue

                        save_response(output_path, item, model_name, method, rate_str, responses, len(responses))
                        print("[MIGRATED]", model_name, filename, item_idx, method, rate_str)


migrate_old_responses("augmented_datasets_with_responses", "augmented_datasets_split", ["openai", "llama"])
