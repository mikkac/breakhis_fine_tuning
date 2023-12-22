import json
import os
import pprint
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)


def find_best_model_idx_and_acc(n_splits, results_path, output_path):
    json_files = [
        results_path / output_path / f"model_{idx}" / "all_results.json"
        for idx in range(n_splits)
    ]

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


def get_best_models(n_splits, results_path, output_paths):
    best_models = {}
    for output_path in output_paths:
        best_models[str(results_path / output_path)] = find_best_model_idx_and_acc(
            n_splits, results_path, output_path
        )

    return best_models


def calculate_mean_metrics(n_splits, results_path, output_path):
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


def get_all_mean_val_metrics(n_splits, results_path, output_paths):
    mean_metrics = []
    for output_path in output_paths:
        mean_metrics.append(calculate_mean_metrics(n_splits, results_path, output_path))

    return mean_metrics


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
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Compute basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )

    # Apply softmax to logits to get probabilities
    softmax_logits = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    # Compute ROC AUC
    roc_auc = roc_auc_score(labels, softmax_logits[:, 1])

    # Compute precision-recall curve and PR AUC
    precision_curve, recall_curve, _ = precision_recall_curve(
        labels, softmax_logits[:, 1]
    )
    pr_auc = auc(recall_curve, precision_curve)

    # Compute confusion matrix and specificity
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
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

    predictions = trainer.predict(test_dataset)
    metrics = trainer.evaluate(eval_dataset=test_dataset)
    logits = predictions.predictions
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    return metrics, probabilities


def save_test_info(best_model_info, val_metrics, test_metrics, model_path, output_path):
    info = {
        "best_model_info": best_model_info,
        "model_path": str(model_path),
        "average_val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(output_path, "w") as f:
        json.dump(info, f, indent=4)


def save_predictions(dataset, probabilities, output_path):
    predicted_labels = np.argmax(probabilities, axis=-1)

    df = pd.DataFrame(
        {
            "file_name": dataset["file_loc"],
            "label": dataset["label"],
            "label_predicted": predicted_labels,
            "label_0_probability": probabilities[:, 0],
            "label_1_probability": probabilities[:, 1],
        }
    )

    df.to_csv(output_path, index=False)


def main_evaluation(
    model_id,
    checkpoint_path,
    test_csv,
    batch_size,
    cpu_threads,
    results_path,
    best_model_info,
    best_mean_val_metrics,
):
    model = AutoModelForImageClassification.from_pretrained(checkpoint_path)
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    test_dataset = load_test_data(test_csv, image_processor)
    test_metrics, probabilities = evaluate_model(
        model, test_dataset["test"], batch_size, cpu_threads
    )
    save_predictions(
        test_dataset["test"],
        probabilities,
        results_path / f"test_predictions.csv",
    )
    save_test_info(
        best_model_info,
        best_mean_val_metrics,
        test_metrics,
        checkpoint_path,
        results_path / f"test_info.json",
    )


def main():
    # Constants
    n_splits = 5

    cwd = Path().absolute()
    results_path = cwd / "results_good"

    best_metric = "eval_f1"

    output_paths = [
        os.path.basename(f.path) for f in os.scandir(results_path) if f.is_dir()
    ]

    # Find best model
    best_models = get_best_models(n_splits, results_path, output_paths)
    mean_val_metrics = get_all_mean_val_metrics(n_splits, results_path, output_paths)

    best_mean_val_metrics = max(mean_val_metrics, key=lambda x: x[best_metric]["mean"])

    best_model_output_path = best_mean_val_metrics["output_path"]
    best_model_index = best_models[best_model_output_path][0]
    with open(
        results_path / best_model_output_path / f"model_info_{best_model_index}.json",
        "r",
    ) as f:
        best_model_info = json.load(f)

    # Run final evaluation on test dataset
    model_id = best_model_info["model_id"]
    checkpoint_path = (
        results_path / best_model_output_path / f"model_{best_model_index}"
    )
    test_csv = str(cwd / f'breakhis_{best_model_info["zoom"]}x' / "test.csv")
    batch_size = 16
    cpu_threads = psutil.cpu_count(logical=True)

    main_evaluation(
        model_id,
        checkpoint_path,
        test_csv,
        batch_size,
        cpu_threads,
        results_path,
        best_model_info,
        best_mean_val_metrics,
    )


if __name__ == "__main__":
    main()
