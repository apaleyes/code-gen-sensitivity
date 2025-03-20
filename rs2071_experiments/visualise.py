import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os

def get_plot_data(df, metric):
    aug_rates = np.sort(df["augmentation_rate"].unique())
    means, lows, highs = [], [], []

    for aug_rate in aug_rates:
        values = df[df["augmentation_rate"] == aug_rate][metric].tolist()
        mean, std = np.mean(values), np.std(values)
        ci = 1.96 * std / np.sqrt(len(values))

        means.append(mean)
        lows.append(mean - ci)
        highs.append(mean + ci)

    return aug_rates, means, lows, highs


def generate_and_save_plots(json_path, output_folder):
    if os.path.exists(output_folder):
        return

    os.makedirs(output_folder, exist_ok=True)

    with open(json_path, 'r') as f:
        data_json = json.load(f)

    data = pd.DataFrame(data_json["measurements"])
    metrics = [col for col in data.columns if col not in ["n_repeat", "augmentation_rate"]]

    model_name = os.path.basename(json_path).split('-')[0]
    aug_method = data_json.get('augmentation_method', 'Unknown')

    # Separate plots for each metric
    for metric in metrics:
        aug_rates, means, lows, highs = get_plot_data(data, metric)
        plt.figure()
        plt.plot(aug_rates, means, label=metric, color='b')
        plt.fill_between(aug_rates, lows, highs, color='b', alpha=0.15)
        plt.ylim(-0.05, 1.05)
        plt.xlabel(f"Augmentation rate ({aug_method})")
        plt.ylabel(metric)
        plt.legend()
        plt.title(f"{model_name} - {metric}")
        plt.savefig(os.path.join(output_folder, f"{metric}.png"))
        plt.close()


def generate_combined_plots(directory):
    combined_data = []
    augmentation_methods = {}

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            json_path = os.path.join(directory, filename)
            with open(json_path, 'r') as f:
                data_json = json.load(f)

            data = pd.DataFrame(data_json["measurements"])
            model_name = os.path.basename(json_path).split('-')[0]
            aug_method = data_json.get('augmentation_method', 'Unknown')
            pair_key = f"{model_name}_{aug_method}"

            data["model_aug_pair"] = pair_key
            combined_data.append(data)
            augmentation_methods[pair_key] = aug_method

    if not combined_data:
        return

    combined_df = pd.concat(combined_data, ignore_index=True)
    metrics = [col for col in combined_df.columns if col not in ["n_repeat", "augmentation_rate", "model_aug_pair"]]
    output_folder = os.path.join(directory, "combined_plots")
    os.makedirs(output_folder, exist_ok=True)

    # Separate plots for each model-augmentation pair
    for model_aug in combined_df["model_aug_pair"].unique():
        model_data = combined_df[combined_df["model_aug_pair"] == model_aug]
        model_folder = os.path.join(output_folder, model_aug)
        os.makedirs(model_folder, exist_ok=True)

        # Combined plot for all metrics for the model-augmentation pair
        plt.figure()
        for metric in metrics:
            aug_rates, means, lows, highs = get_plot_data(model_data, metric)
            plt.plot(aug_rates, means, label=metric)
            plt.fill_between(aug_rates, lows, highs, alpha=0.15)

        plt.ylim(-0.05, 1.05)
        plt.xlabel("Augmentation rate")
        plt.ylabel("Measurement values")
        plt.legend()
        plt.title(f"{model_aug} - All Metrics")
        plt.savefig(os.path.join(model_folder, "all_metrics.png"))
        plt.close()

        # Individual metric plots
        for metric in metrics:
            plt.figure()
            aug_rates, means, lows, highs = get_plot_data(model_data, metric)
            plt.plot(aug_rates, means, label=f"{model_aug}")
            plt.fill_between(aug_rates, lows, highs, alpha=0.15)

            plt.ylim(-0.05, 1.05)
            plt.xlabel("Augmentation rate")
            plt.ylabel(metric)
            plt.legend()
            plt.title(f"{model_aug} - {metric}")
            plt.savefig(os.path.join(model_folder, f"{metric}.png"))
            plt.close()


def process_experiments(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            json_path = os.path.join(directory, filename)
            output_folder = os.path.join(directory, filename[:-5])
            generate_and_save_plots(json_path, output_folder)


# Uncomment to run
# generate_combined_plots('experiments')
