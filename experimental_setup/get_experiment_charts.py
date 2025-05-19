import os
import pandas as pd
import matplotlib.pyplot as plt

metrics_dir = "./augmented_datasets_metrics"
charts_dir = "./augmented_dataset_charts"
agg_dir = os.path.join(charts_dir, "aggregated")
os.makedirs(charts_dir, exist_ok=True)
os.makedirs(agg_dir, exist_ok=True)

def plot_metric_chart(tsed_data, bert_data, tsed_sem=None, bert_sem=None, title="", save_path=None):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    if tsed_data.notna().any():
        tsed_min = tsed_data.min(skipna=True)
        tsed_max = tsed_data.max(skipna=True)
        tsed_range = tsed_max - tsed_min if pd.notna(tsed_min) and pd.notna(tsed_max) else 0
        ax1.errorbar(
            tsed_data.index, tsed_data,
            yerr=tsed_sem if tsed_sem is not None and tsed_sem.notna().any() else None,
            label="TSED", marker='o', capsize=4, color="tab:blue"
        )
        ax1.set_ylabel("TSED Score", color="tab:blue")
        ax1.set_ylim(tsed_min - 0.1 * tsed_range, tsed_max + 0.1 * tsed_range)
        ax1.tick_params(axis='y', labelcolor="tab:blue")

    if bert_data.notna().any():
        bert_min = bert_data.min(skipna=True)
        bert_max = bert_data.max(skipna=True)
        bert_range = bert_max - bert_min if pd.notna(bert_min) and pd.notna(bert_max) else 0
        ax2.errorbar(
            bert_data.index, bert_data,
            yerr=bert_sem if bert_sem is not None and bert_sem.notna().any() else None,
            label="BERT", marker='o', capsize=4, color="tab:orange"
        )
        ax2.set_ylabel("BERT Score", color="tab:orange")
        ax2.set_ylim(bert_min - 0.1 * bert_range, bert_max + 0.1 * bert_range)
        ax2.tick_params(axis='y', labelcolor="tab:orange")

    ax1.set_xlabel("Augmentation Rate")
    plt.title(title)
    plt.grid(True)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

# Store data per (model, method, dataset)
grouped_data = {}

for root, _, files in os.walk(metrics_dir):
    for file in files:
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(root, file)
        df = pd.read_csv(file_path)

        if df.empty or "augmentation_rate" not in df.columns:
            continue

        df["augmentation_rate"] = df["augmentation_rate"].astype(float)
        df.sort_values("augmentation_rate", inplace=True)

        valid_rates = df["augmentation_rate"].value_counts()
        valid_rates = valid_rates[valid_rates == 5].index
        df = df[df["augmentation_rate"].isin(valid_rates)]

        if df.empty:
            continue

        grouped_mean = df.groupby("augmentation_rate")[["tsed_score", "bert_score"]].mean()
        grouped_sem = df.groupby("augmentation_rate")[["tsed_score", "bert_score"]].sem()

        tsed_data = grouped_mean["tsed_score"]
        bert_data = grouped_mean["bert_score"]
        tsed_sem = grouped_sem["tsed_score"]
        bert_sem = grouped_sem["bert_score"]

        relative_path = os.path.relpath(root, metrics_dir)
        save_dir = os.path.join(charts_dir, relative_path)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file.replace(".csv", ".png"))

        plot_metric_chart(tsed_data, bert_data, tsed_sem, bert_sem, title=file.replace(".csv", ""), save_path=save_path)

        # Extract model/method/dataset for aggregation grouping
        parts = os.path.normpath(file_path).split(os.sep)
        model, method, dataset = parts[-4], parts[-3], parts[-2]
        group_key = f"{model}__{method}__{dataset}"

        if group_key not in grouped_data:
            grouped_data[group_key] = []
        grouped_data[group_key].append(grouped_mean.reset_index())

# Aggregated
for group_key, data_frames in grouped_data.items():
    combined_df = pd.concat(data_frames)
    agg_mean = combined_df.groupby("augmentation_rate")[["tsed_score", "bert_score"]].mean().interpolate(method="linear", limit_direction="both")
    agg_sem = combined_df.groupby("augmentation_rate")[["tsed_score", "bert_score"]].sem().interpolate(method="linear", limit_direction="both")

    if agg_mean.empty:
        continue

    tsed_data = agg_mean["tsed_score"]
    bert_data = agg_mean["bert_score"]
    tsed_sem = agg_sem["tsed_score"]
    bert_sem = agg_sem["bert_score"]

    save_path = os.path.join(agg_dir, f"{group_key}_aggregated.png")
    plot_metric_chart(tsed_data, bert_data, tsed_sem, bert_sem, title=group_key.replace('_', ' '), save_path=save_path)
