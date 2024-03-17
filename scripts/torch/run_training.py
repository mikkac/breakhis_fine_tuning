"""
Fine-tune a pretrained model on a BreakHis dataset using
PyTorch via HuggingFace's Transformers library.
"""

import json
import multiprocessing
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch
from datasets import config, load_dataset
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    EarlyStoppingCallback,
    PrinterCallback,
    Trainer,
    TrainingArguments,
)
from transformers.integrations import TensorBoardCallback

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set IN_MEMORY_MAX_SIZE to available RAM
config.IN_MEMORY_MAX_SIZE = int(psutil.virtual_memory().available * 0.9)
print(
    f"{config.IN_MEMORY_MAX_SIZE / (1024**3):.2f} GB of data will be stored in the RAM"
)

CPU_THREADS = psutil.cpu_count(logical=True)
print(f"Number of available CPU threads: {CPU_THREADS}")


# MODEL_ID, EXPERIMENT_PREFIX = "google/vit-base-patch16-224", "ViT"
MODEL_ID, EXPERIMENT_PREFIX = "facebook/convnext-base-224-22k", "ConvNext"
# MODEL_ID, EXPERIMENT_PREFIX = "microsoft/swin-base-patch4-window7-224-in22k", "Swin"

IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_ID)


def process_example(image):
    """Preprocesses an image for the model"""
    inputs = IMAGE_PROCESSOR(image, return_tensors="pt")
    return inputs["pixel_values"].squeeze(0)


def process_dataset(example):
    """Preprocesses a dataset for the model"""
    image = Image.open(example["file_loc"]).convert("RGB")
    example["pixel_values"] = process_example(image)
    return example


def print_first_patient_ids(dataset, n=10):
    """Prints the 'patient_id' for the first n rows of a Hugging Face dataset."""
    for i in range(n):
        print(dataset[i]["patient_id"])


def load_data(fold_index, input_path):
    """Loads the dataset"""
    train_csv = str(input_path / f"train_{fold_index}.csv")
    val_csv = str(input_path / f"val_{fold_index}.csv")
    dataset = load_dataset(
        "csv", data_files={"train": train_csv, "val": val_csv}, keep_in_memory=False
    )
    dataset = dataset.map(process_dataset, with_indices=False, num_proc=CPU_THREADS)
    dataset["train"] = dataset["train"].shuffle(seed=42)
    dataset["val"] = dataset["val"].shuffle(seed=42)

    print("First 10 rows of the train dataset:")
    print_first_patient_ids(dataset["train"])
    print("\nFirst 10 rows of the validation dataset:")
    print_first_patient_ids(dataset["val"])
    print(f"\nLoaded {fold_index} dataset: {dataset}")
    return dataset


id2label = {0: "benign", 1: "malignant"}
label2id = {v: k for k, v in id2label.items()}


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
        "f1": f1,
        "roc_auc": roc_auc,  # Area under the ROC curve
        "pr_auc": pr_auc,  # Area under the Precision-Recall curve
        "specificity": specificity,  # Specificity (True Negative Rate)
    }


# Define the Trainer and train the model
def train_model(
    fold_idx,
    train_dataset,
    val_dataset,
    learning_rate,
    weight_decay_rate,
    epochs,
    batch_size,
    output_path,
):  # pylint: disable=R0913
    """Trains the model"""
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(id2label),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    print(f"MODEL INFO: {model}")
    print(
        f"PARAMS NUM (M): {sum(param.nelement() for param in model.parameters()) / 1e6}"
    )
    training_args = TrainingArguments(
        output_dir=str(output_path / f"model_{fold_idx}"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # gradient_accumulation_steps=4, # slows down, saves memory
        # gradient_checkpointing=True, # slows down, saves memory
        dataloader_num_workers=CPU_THREADS,  # can this even help with current form of data loading?
        num_train_epochs=epochs,
        # eval_delay=4,    # evaluate after 4 epochs
        optim="adamw_bnb_8bit",
        learning_rate=learning_rate,
        weight_decay=weight_decay_rate,
        push_to_hub=False,
        remove_unused_columns=False,
        logging_strategy="steps",
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=250,
        save_steps=250,
        save_total_limit=1,
        # fp16=True,
        bf16=True,  # available only with +Ampere architecture (>=RTX 3000)
        tf32=True,
        torch_compile=True,
        do_train=True,
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="eval_loss",  # Choose the metric to monitor for early stopping
        greater_is_better=False,  # Early stop when the metric is decreasing
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=6, early_stopping_threshold=0.001
            ),
            PrinterCallback(),
        ],
    )

    train_result = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.save_metrics("eval", eval_metrics)
    return model, train_result.metrics, eval_metrics


def run_fold(
    fold_idx,
    learning_rate,
    weight_decay_rate,
    zoom,
    epochs,
    batch_size,
    input_path,
    output_path,
):
    """Runs the training process for one fold"""
    print(f"Starting experiment with fold {fold_idx}. Hyperparams:")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay rate: {weight_decay_rate}")

    dataset = load_data(fold_idx, input_path)

    train_dataset = dataset["train"]
    val_dataset = dataset["val"]

    _, train_metrics, eval_metrics = train_model(
        fold_idx,
        train_dataset,
        val_dataset,
        learning_rate,
        weight_decay_rate,
        epochs,
        batch_size,
        output_path,
    )
    print(f"Fold {fold_idx} finished")
    print(f"Train metrics: {train_metrics}")
    print(f"Evaluation metrics: {eval_metrics}")

    save_model_info(
        output_path,
        fold_idx,
        learning_rate,
        weight_decay_rate,
        zoom,
        epochs,
        batch_size,
    )


def save_model_info(
    output_path, fold_idx, learning_rate, weight_decay_rate, zoom, epochs, batch_size
):
    """Saves the model information"""
    model_info = {
        "idx": fold_idx,
        "model_id": MODEL_ID,
        "zoom": zoom,
        "n_splits": 5,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay_rate": weight_decay_rate,
    }
    with open(output_path / f"model_info_{fold_idx}.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4)
    print(json.dumps(model_info, indent=4))


def main():
    """Main function"""
    cwd = Path().absolute()
    scenarios = (
        # "original",
        # "patches_fixed",
        "patches_fixed_with_random",
        "patches_fixed_with_random_with_transformations",
        "patches_fixed_with_random_with_filtered_cells",
        "patches_fixed_with_random_with_filtered_background",
    )
    zoom = 200
    learning_rate = 3e-5
    weight_decay_rate = 5e-3
    # fold_indices = [4]
    fold_indices = range(5) # default

    epochs = 50
    # batch_size = 64
    batch_size = 16
    for scenario in scenarios:
        experiment_id = f"{EXPERIMENT_PREFIX}_{scenario}"
        lr_f = f"{learning_rate:.0e}".replace(".", "p")
        wdr_f = f"{weight_decay_rate:.0e}".replace(".", "p")
        cur_dt_f = datetime.now().strftime("%Y_%m_%d_%H_%M")
        input_path = cwd / "data" / f"{zoom}x" / scenario
        output_path = (
            cwd
            / "results"
            / f"{zoom}x_{experiment_id}_lr_{lr_f}_wdr_{wdr_f}_{cur_dt_f}"
        )

        os.makedirs(output_path, exist_ok=True)
        print(f"Directory {output_path} created.")
        for fold_idx in fold_indices:
            print(
                f"Starting experiment {experiment_id} with fold {fold_idx}. Hyperparams:"
            )
            print(f"Scenario: {scenario}")
            print(f"Learning rate: {learning_rate}")
            print(f"Weight decay rate: {weight_decay_rate}")
            p = multiprocessing.Process(
                target=run_fold,
                args=(
                    fold_idx,
                    learning_rate,
                    weight_decay_rate,
                    zoom,
                    epochs,
                    batch_size,
                    input_path,
                    output_path,
                ),
            )
            p.start()
            p.join()  # This will block until p finishes execution
            time.sleep(5)


if __name__ == "__main__":
    main()
