import os
import json
import csv
import torchmetrics
from TSED import TSED
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def calculate_metrics_to_csv(data_dir="augmented_datasets_split", out_dir="augmented_datasets_metrics", partial=False):
    bert_score_metric = torchmetrics.text.BERTScore(
        model_name_or_path="roberta-large",
        max_length=512,
        truncation=True
    )

    for model in os.listdir(data_dir):
        model_dir = os.path.join(data_dir, model)
        for method in os.listdir(model_dir):
            method_dir = os.path.join(model_dir, method)
            for dataset in os.listdir(method_dir):
                dataset_dir = os.path.join(method_dir, dataset)

                for item_file in os.listdir(dataset_dir):
                    changed = False

                    task = item_file.replace(".json", "")
                    rel_dir = os.path.join(model, method, dataset)
                    out_subdir = os.path.join(out_dir, rel_dir)
                    os.makedirs(out_subdir, exist_ok=True)
                    out_csv = os.path.join(out_subdir, f"{task}.csv")

                    item_path = os.path.join(dataset_dir, item_file)
                    with open(item_path, "r", encoding="utf-8") as f:
                        item = json.load(f)

                    method_responses = item.get("llm_responses", {}).get(model, {}).get(method, {})
                    original_codes = method_responses.get("0.0", [])
                    if not original_codes:
                        continue

                    existing_rows = []
                    if os.path.exists(out_csv):
                        with open(out_csv, newline='', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            existing_rows = list(reader)
                    else:
                        changed = True
                        for rate_str, responses in sorted(method_responses.items(), key=lambda x: float(x[0])):
                            for _ in responses:
                                existing_rows.append({
                                    "model": model,
                                    "method": method,
                                    "dataset": dataset,
                                    "task": task,
                                    "augmentation_rate": rate_str,
                                    "tsed_score": "",
                                    "bert_score": ""
                                })

                    updated_rows = []
                    aug_counts = {}

                    for i, row in enumerate(existing_rows):
                        # print(row)
                        try:
                            aug_rate = row["augmentation_rate"]
                            rate_responses = method_responses.get(aug_rate, [])
                            index_within_rate = aug_counts.get(aug_rate, 0)

                            if not partial or i % 20 < 5:
                                code = rate_responses[index_within_rate]
                                aug_counts[aug_rate] = index_within_rate + 1

                                if code and not code.startswith("ERROR"):
                                    if not row["tsed_score"]:
                                        changed = True
                                        tsed_score = sum(
                                            TSED.Calaulte("python", ref, code, 1.0, 0.8, 1.0)
                                            for ref in original_codes
                                        ) / len(original_codes)
                                        row["tsed_score"] = tsed_score
                                        print(f"{model} {method} {dataset} {task} [TSED] @ {aug_rate} #{index_within_rate}: {tsed_score}")

                                    # if not row["bert_score"]:
                                    #     changed = True
                                    #     preds = [code] * len(original_codes)
                                    #     refs = original_codes
                                    #     score = bert_score_metric(preds, refs)["f1"].mean().item()
                                    #     row["bert_score"] = score
                                    #     print(f"{model} {method} {dataset} {task} [BERT] @ {aug_rate} #{index_within_rate}: {score}")
                        except Exception as e:
                            print(f"[ERROR] {item_path} – {e}")
                        # print(row)
                        updated_rows.append(row)

                    if updated_rows and changed:
                        with open(out_csv, "w", newline='', encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=["model", "method", "dataset", "task", "augmentation_rate", "tsed_score", "bert_score"])
                            writer.writeheader()
                            writer.writerows(updated_rows)
                        print(f"[SAVED] {out_csv}")

if __name__ == "__main__":
    calculate_metrics_to_csv(partial=True)

# def flag_bad_csv_files(base_dir="augmented_datasets_metrics"):
#     bad_files = []
#     for root, _, files in os.walk(base_dir):
#         for file in files:
#             if not file.endswith(".csv"):
#                 continue
#             path = os.path.join(root, file)
#             with open(path, newline="", encoding="utf-8") as f:
#                 reader = csv.DictReader(f)
#                 rows = list(reader)
#
#             if len(rows) != 105:
#                 bad_files.append((path, "Incorrect row count"))
#                 continue
#
#             aug_rate_counts = {}
#             aug_rates = []
#             for row in rows:
#                 rate = float(row["augmentation_rate"])
#                 aug_rate_counts[rate] = aug_rate_counts.get(rate, 0) + 1
#                 aug_rates.append(rate)
#
#             if any(count != 5 for count in aug_rate_counts.values()):
#                 bad_files.append((path, "Not exactly 5 rows per aug_rate"))
#                 continue
#
#             if aug_rates != sorted(aug_rates):
#                 bad_files.append((path, "Rows not sorted by aug_rate"))
#
#     print("[FLAGGED FILES]")
#     for path, reason in bad_files:
#         print(f"{path} – {reason}")
#         os.remove(path)
#
# flag_bad_csv_files()