import os
import json
import csv
import torchmetrics
from TSED import TSED
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def run_and_save_all_repeats_to_csv(data_dir="augmented_datasets_split", out_dir="augmented_datasets_metrics", partial=False):
    bert_score_metric = torchmetrics.text.BERTScore(
        model_name_or_path="roberta-large",
        max_length=512,
        truncation=True
    )
    fieldnames = ["model", "method", "dataset", "task", "augmentation_rate", "tsed_score", "bert_score"]

    for model in os.listdir(data_dir):
        model_dir = os.path.join(data_dir, model)
        for method in os.listdir(model_dir):
            method_dir = os.path.join(model_dir, method)
            for dataset in os.listdir(method_dir):
                dataset_dir = os.path.join(method_dir, dataset)

                for item_file in os.listdir(dataset_dir):
                    task = item_file.replace(".json", "")
                    rel_dir = os.path.join(model, method, dataset)
                    out_subdir = os.path.join(out_dir, rel_dir)
                    os.makedirs(out_subdir, exist_ok=True)
                    out_csv = os.path.join(out_subdir, f"{task}.csv")

                    # Load previous rows
                    existing_rows = []
                    if os.path.exists(out_csv):
                        with open(out_csv, newline='', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            existing_rows = list(reader)

                    item_path = os.path.join(dataset_dir, item_file)
                    with open(item_path, "r", encoding="utf-8") as f:
                        item = json.load(f)

                    method_responses = item.get("llm_responses", {}).get(model, {}).get(method, {})
                    original_codes = method_responses.get("0.0", [])
                    if not original_codes:
                        continue

                    updated_rows = []
                    row_index = 0

                    for rate_str in sorted(method_responses.keys(), key=float):
                        aug_rate = float(rate_str)

                        if partial and (round(aug_rate * 100) % 20 != 0):
                            responses = method_responses[rate_str]
                            for _ in responses:
                                if row_index < len(existing_rows):
                                    updated_rows.append(existing_rows[row_index])
                                else:
                                    updated_rows.append({
                                        "model": model,
                                        "method": method,
                                        "dataset": dataset,
                                        "task": task,
                                        "augmentation_rate": str(aug_rate),
                                        "tsed_score": "",
                                        "bert_score": ""
                                    })
                                row_index += 1
                            continue

                        responses = method_responses[rate_str]
                        for i, code in enumerate(responses):
                            # In partial mode, only process first response per rate
                            if partial and i > 0:
                                if row_index < len(existing_rows):
                                    updated_rows.append(existing_rows[row_index])
                                else:
                                    updated_rows.append({
                                        "model": model,
                                        "method": method,
                                        "dataset": dataset,
                                        "task": task,
                                        "augmentation_rate": str(aug_rate),
                                        "tsed_score": "",
                                        "bert_score": ""
                                    })
                                row_index += 1
                                continue

                            if not code or code.startswith("ERROR"):
                                if row_index < len(existing_rows):
                                    updated_rows.append(existing_rows[row_index])
                                row_index += 1
                                continue

                            if row_index < len(existing_rows):
                                row = existing_rows[row_index]
                            else:
                                row = {
                                    "model": model,
                                    "method": method,
                                    "dataset": dataset,
                                    "task": task,
                                    "augmentation_rate": str(aug_rate),
                                    "tsed_score": "",
                                    "bert_score": ""
                                }

                            try:
                                if not row["tsed_score"]:
                                    tsed_score = sum(
                                        TSED.Calaulte("python", ref, code, 1.0, 0.8, 1.0)
                                        for ref in original_codes
                                    ) / len(original_codes)
                                    row["tsed_score"] = tsed_score
                                    print(f"[TSED] {task} @ {aug_rate}: {tsed_score}")

                                if not row["bert_score"]:
                                    preds = [code] * len(original_codes)
                                    refs = original_codes
                                    score = bert_score_metric(preds, refs)["f1"].mean().item()
                                    row["bert_score"] = score
                                    print(f"[BERT] {task} @ {aug_rate}: {score}")
                            except Exception as e:
                                print(f"[ERROR] {item_path} – {e}")

                            updated_rows.append(row)
                            row_index += 1

                    updated_rows += existing_rows[row_index:]

                    with open(out_csv, "w", newline="", encoding="utf-8") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(updated_rows)
                    print(f"[SAVED] {out_csv}")

if __name__ == "__main__":
    run_and_save_all_repeats_to_csv(partial=True)