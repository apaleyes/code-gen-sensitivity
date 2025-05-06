import os
import csv


def process_csvs(in_dir="experiments_csvs", out_dir="augmented_datasets_metrics"):
    for root, _, files in os.walk(in_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue
            in_path = os.path.join(root, file)

            rel_path = os.path.relpath(in_path, in_dir)
            parts = rel_path.split(os.sep)
            if len(parts) < 4:
                continue
            model, method, dataset, task_file = parts
            task = task_file.replace(".csv", "")

            out_path = os.path.join(out_dir, model, method, dataset, task_file)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            with open(in_path, newline='', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                rows = [
                    {
                        "model": model,
                        "method": method,
                        "dataset": dataset,
                        "task": task,
                        "augmentation_rate": row["augmentation_rate"],
                        "tsed_score": row["code_similarity"],
                        "bert_score": ""
                    }
                    for row in reader
                ]

            with open(out_path, "w", newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=["model", "method", "dataset", "task", "augmentation_rate",
                                                             "tsed_score", "bert_score"])
                writer.writeheader()
                writer.writerows(rows)
            print(f"[UPDATED] {out_path}")


if __name__ == "__main__":
    process_csvs()
