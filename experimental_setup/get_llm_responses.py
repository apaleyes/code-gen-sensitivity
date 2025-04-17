import os
import json
import concurrent.futures
import time
import random
from tqdm import tqdm
from models import get_model, ModelCaller
from prompt_utils import ensure_python_code_prompt


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


def main():
    input_dir = "augmented_datasets"
    output_dir = "augmented_datasets_with_responses"
    os.makedirs(output_dir, exist_ok=True)

    model_names = ["openai", "llama"]
    n_repeats = 5
    request_buffer = 6  # send 6 requests, keep 5

    all_data = []
    file_map = {}

    for filename in os.listdir(input_dir):
        if not filename.endswith(".json"):
            continue

        output_path = os.path.join(output_dir, filename)
        input_path = os.path.join(input_dir, filename)
        load_path = output_path if os.path.exists(output_path) else input_path

        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.extend(data)
            file_map[filename] = data

    # dataset_pbar = tqdm(total=len(all_data), desc="Questions", position=0)

    for model_name in model_names:
        model = get_model(model_name)
        model_caller = ModelCaller(model, prompt_transform=ensure_python_code_prompt)

        for filename, data in file_map.items():
            for item_idx, item in enumerate(data):
                item.setdefault("llm_responses", {})
                item["llm_responses"].setdefault(model_name, {})

                augmented = item.get("augmented_questions", {})
                # level_pbar = tqdm(total=sum(len(v) for v in augmented.values()), desc="Levels", position=1, leave=False)

                for method, versions in augmented.items():
                    item["llm_responses"][model_name].setdefault(method, {})

                    for rate, prompt in versions.items():
                        rate_str = str(rate)
                        existing = item["llm_responses"][model_name][method].get(rate_str, [])
                        if isinstance(existing, list) and len(existing) >= n_repeats:
                            # tqdm.write(f"[SKIP] {filename} Q{item_idx} {model_name} {method} {rate_str}")
                            # level_pbar.update(1)
                            print(model_name, filename, item_idx, rate)
                            continue

                        responses = existing if isinstance(existing, list) else []
                        prompts = [prompt] * request_buffer

                        try:
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                futures = [executor.submit(call_with_retry, model_caller, p) for p in prompts]
                                for future in concurrent.futures.as_completed(futures):
                                    if len(responses) >= n_repeats:
                                        for f in futures:
                                            if not f.done():
                                                f.cancel()
                                        break
                                    try:
                                        result = future.result()
                                    except Exception as e:
                                        result = f"ERROR: {str(e)}"
                                    responses.append(result)

                            if len(responses) < n_repeats:
                                tqdm.write(f"[WARN] {filename} Q{item_idx} {model_name} {method} {rate_str} – only got {len(responses)}")
                                continue

                            item["llm_responses"][model_name][method][rate_str] = responses[:n_repeats]

                            output_path = os.path.join(output_dir, filename)
                            with open(output_path, "w", encoding="utf-8") as f:
                                json.dump(data, f, indent=2)

                            # tqdm.write(f"[SAVED] {filename} Q{item_idx} {model_name} {method} {rate_str}")
                        except Exception as e:
                            tqdm.write(f"[FAIL] {filename} Q{item_idx} {model_name} {method} {rate_str} – {str(e)}")

                        # level_pbar.update(1)
                        print(model_name, filename, item_idx, rate)

                # level_pbar.close()
                # dataset_pbar.update(1)
                print(model_name, filename, item_idx)

    # dataset_pbar.close()


if __name__ == "__main__":
    main()
