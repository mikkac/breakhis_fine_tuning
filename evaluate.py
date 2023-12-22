from pathlib import Path
from datasets import load_dataset
import json
from PIL import Image
import os
import numpy as np
import pandas as pd
import pprint
import psutil
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

n_splits = 5

cwd = Path().absolute()
results_path = cwd / "results_good"

output_paths = [
    os.path.basename(f.path) for f in os.scandir(results_path) if f.is_dir()
]


def find_best_model_idx_and_acc(output_path):
    json_files = [
        results_path / output_path / f"model_{idx}" / "all_results.json"
        for idx in range(n_splits)
    ]
    # csv_files = [results_path / output_path / f'train_metrics_{idx}.csv' for idx in range(n_splits)]
    # csv_files = [results_path / f'train_metrics_{idx}.csv' for idx in range(n_splits)]
    # dataframes = [pd.read_csv(file) for file in csv_files]

    best_model_index = None
    best_f1_score = 0.0

    for i, json_files in enumerate(json_files):
        with open(json_files) as json_file:
            data = json.load(json_file)
            print(
                f"Model {i}: e_acc: {data['eval_accuracy']:.3f}, e_loss: {data['eval_loss']:.3f}, e_f1: {data['eval_f1']:.3f}"
            )
            f1_score = data["eval_f1"]
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_model_index = i

    print(f"Best model index: {best_model_index}, f1_score: {best_f1_score}")
    return best_model_index, best_f1_score


best_models = {}
for output_path in output_paths:
    best_models[str(results_path / output_path)] = find_best_model_idx_and_acc(
        output_path
    )

print(best_models)


def calculate_mean_metrics(output_path):
    numeric_columns = [
        "eval_accuracy",
        "eval_auc",
        "eval_f1",
        "eval_loss",
        "eval_precision",
        "eval_recall",
        "train_loss",
    ]

    json_files = [
        results_path / output_path / f"model_{idx}" / "all_results.json"
        for idx in range(n_splits)
    ]

    data_list = []
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
            data_list.append({k: v for k, v in data.items() if k in numeric_columns})

    df = pd.DataFrame(data_list)

    mean_metrics = df.mean()
    std_metrics = df.std()

    # print(df)

    metrics = {
        metric_name: {
            "mean": mean_metrics[metric_name],
            "std": std_metrics[metric_name],
        }
        for metric_name in mean_metrics.index
    }

    with open(
        results_path / output_path / "train_metrics_mean_with_std.json", "w"
    ) as f:
        json.dump(metrics, f, indent=4)

    metrics["output_path"] = str(results_path / output_path)

    return metrics


mean_metrics = []
for output_path in output_paths:
    mean_metrics.append(calculate_mean_metrics(output_path))


for m in mean_metrics:
    pprint.pprint(m)
max_f1_score = max(mean_metrics, key=lambda x: x["eval_f1"]["mean"])

# with open(results_path / f'model_info_{best_model_index}.json', 'r') as f:
best_model_output_path = max_f1_score["output_path"]
best_model_index = best_models[best_model_output_path][0]
with open(
    results_path / best_model_output_path / f"model_info_{best_model_index}.json", "r"
) as f:
    best_model_info = json.load(f)

input_path = cwd / f'breakhis_{best_model_info["zoom"]}x'


###############
def process_example(image, image_processor):
    """Preprocesses an image for the model"""
    inputs = image_processor(image, return_tensors="pt")
    return inputs["pixel_values"].squeeze(0)


def process_dataset(example, image_processor):
    """Preprocesses a dataset for the model"""
    image = Image.open(example["file_loc"]).convert("RGB")
    example["pixel_values"] = process_example(image, image_processor)
    return example


def load_test_data(test_csv, image_processor):
    dataset = load_dataset("csv", data_files={"test": test_csv})
    dataset = dataset.map(
        lambda e: process_dataset(e, image_processor), with_indices=False, num_proc=2
    )
    print(f"Loaded test dataset: {len(dataset['test'])} samples")

    return dataset


def compute_metrics(eval_pred):
    """Computes the metrics"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    # auc = roc_auc_score(labels, logits[:, 1])  # For binary classification
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # "auc": auc,
    }

def evaluate_model(model, test_dataset, batch_size, cpu_threads):
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
        args=TrainingArguments(
            output_dir="/tmp/eval",
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=cpu_threads,
        ),
    )
    metrics = trainer.evaluate(eval_dataset=test_dataset)
    return metrics


def main_evaluation(model_id, checkpoint_path, test_csv, batch_size, cpu_threads):
    model = AutoModelForImageClassification.from_pretrained(checkpoint_path)
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    test_dataset = load_test_data(test_csv, image_processor)
    metrics = evaluate_model(model, test_dataset["test"], batch_size, cpu_threads)
    print(metrics)


# Example usage
model_id = best_model_info["model_id"]
checkpoint_path = results_path / best_model_output_path / f"model_{best_model_index}"
test_csv = str(input_path / "test.csv")
batch_size = 16
cpu_threads = psutil.cpu_count(logical=True)

main_evaluation(model_id, checkpoint_path, test_csv, batch_size, cpu_threads)
