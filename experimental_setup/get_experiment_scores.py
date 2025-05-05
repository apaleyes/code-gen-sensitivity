import concurrent.futures
import os
import json
import csv
from TSED import TSED
import code_execute
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def run_and_save_all_repeats_to_csv(data_dir="augmented_datasets_split", out_dir="experiments_csvs", n_repeats=5):
    for model in os.listdir(data_dir):
        model_dir = os.path.join(data_dir, model)
        for method in os.listdir(model_dir):
            method_dir = os.path.join(model_dir, method)
            for filename in os.listdir(method_dir):
                file_dir = os.path.join(method_dir, filename)

                for item_file in os.listdir(file_dir):
                    rel_dir = os.path.join(model, method, filename)
                    out_subdir = os.path.join(out_dir, rel_dir)
                    out_csv = os.path.join(out_subdir, f"{item_file.replace('.json', '')}.csv")

                    if os.path.exists(out_csv):
                        print(f"[SKIP] {out_csv}")
                        continue

                    item_path = os.path.join(file_dir, item_file)

                    with open(item_path, "r", encoding="utf-8") as f:
                        item = json.load(f)

                    original_prompt = item["question"]
                    original_solution = item["python_solutions"]

                    method_responses = item.get("llm_responses", {}).get(model, {}).get(method, {})
                    original_codes = method_responses.get("0.0", [])
                    if not original_codes:
                        continue

                    rows = []
                    for rate_str, responses in method_responses.items():
                        aug_rate = float(rate_str)
                        for code in responses[:n_repeats]:
                            if not code or code.startswith("ERROR"):
                                continue
                            try:
                                code_sim = sum(
                                    TSED.Calaulte("python", ref, code, 1.0, 0.8, 1.0)
                                    for ref in original_codes
                                ) / len(original_codes)
                                sol_sim = TSED.Calaulte("python", original_solution, code, 1.0, 0.8, 1.0)
                                # with concurrent.futures.ThreadPoolExecutor() as executor:
                                #     future = executor.submit(code_execute.evaluate_solution, code, original_prompt)
                                #     try:
                                #         acc = future.result(timeout=5)  # total timeout per code snippet
                                #     except Exception as e:
                                #         print(f"[TIMEOUT] {item_path} – {e}")
                                #         acc = 0.0
                                rows.append([aug_rate, code_sim, sol_sim])
                                print(aug_rate, code_sim, sol_sim)
                            except Exception as e:
                                print(f"[ERROR] {item_path} – {e}")

                    # Save to CSV with same subdir structure
                    rel_dir = os.path.join(model, method, filename)
                    out_subdir = os.path.join(out_dir, rel_dir)
                    os.makedirs(out_subdir, exist_ok=True)

                    out_csv = os.path.join(out_subdir, f"{item_file.replace('.json', '')}.csv")
                    with open(out_csv, "w", newline="", encoding="utf-8") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["augmentation_rate", "code_similarity", "solution_similarity", "code_accuracy"])
                        writer.writerows(rows)
                    print(f"[SAVED] {out_csv}")

if __name__ == "__main__":
    run_and_save_all_repeats_to_csv()
