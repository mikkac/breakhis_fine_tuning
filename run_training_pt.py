"""
Fine-tune a pretrained model on a BreakHis dataset using
PyTorch via HuggingFace's Transformers library.
"""

import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    EarlyStoppingCallback,
    PrinterCallback,
    Trainer,
    TrainingArguments,
)

# Define the model architecture and image processor
# MODEL_ID = "microsoft/swin-base-patch4-window7-224-in22k"
# MODEL_ID = "google/vit-base-patch16-224"
MODEL_ID = "facebook/deit-base-distilled-patch16-224"
# MODEL_ID = "facebook/convnext-base-224-22k"
IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_ID)

ZOOM = 400

# Define file paths
CWD = Path().absolute()
INPUT_PATH = CWD / f"breakhis_{ZOOM}x"


def process_example(image):
    """Preprocesses an image for the model"""
    inputs = IMAGE_PROCESSOR(image, return_tensors="pt")
    return inputs["pixel_values"]


def process_dataset(example):
    """Preprocesses a dataset for the model"""
    example["pixel_values"] = process_example(
        Image.open(example["file_loc"]).convert("RGB")
    )
    example["pixel_values"] = np.squeeze(example["pixel_values"], axis=0)
    return example


def remove_extra_dim(example):
    """Removes the extra dimension from the dataset"""
    example["pixel_values"] = np.squeeze(example["pixel_values"], axis=0)
    return example


def load_data(fold_index):
    """Loads the dataset"""
    train_csv = str(INPUT_PATH / f"train_{fold_index}.csv")
    val_csv = str(INPUT_PATH / f"val_{fold_index}.csv")
    dataset = load_dataset("csv", data_files={"train": train_csv, "val": val_csv})

    dataset = dataset.map(process_dataset, with_indices=False, num_proc=4)

    print(f"Loaded {fold_index} dataset: {dataset}")

    return dataset


EPOCHS = 50
BATCH_SIZE = 16

id2label = {0: "benign", 1: "malignant"}
label2id = {v: k for k, v in id2label.items()}


def compute_metrics(eval_pred):
    """Computes the metrics"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    auc = roc_auc_score(labels, logits[:, 1])  # For binary classification
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


# Define the Trainer and train the model
def train_model(
    fold_idx, train_dataset, val_dataset, learning_rate, weight_decay_rate, output_path
):  # pylint: disable=R0913
    """ " Trains the model"""
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(id2label),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(output_path / f"model_{fold_idx}"),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=learning_rate,
        weight_decay=weight_decay_rate,
        push_to_hub=False,
        remove_unused_columns=False,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        use_cpu=False,
        fp16=True,
        torch_compile=True,
        do_train=True,
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="eval_loss",  # Choose the metric to monitor for early stopping
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5, early_stopping_threshold=0.0
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


def run_fold(fold_idx, learning_rate, weight_decay_rate, output_path):
    """Runs the training process for one fold"""
    print(f"Starting experiment with fold {fold_idx}. Hyperparams:")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay rate: {weight_decay_rate}")

    dataset = load_data(fold_idx)

    train_dataset = dataset["train"]
    val_dataset = dataset["val"]

    _, train_metrics, eval_metrics = train_model(
        fold_idx,
        train_dataset,
        val_dataset,
        learning_rate,
        weight_decay_rate,
        output_path,
    )
    print(f"Fold {fold_idx} finished")
    print(f"Train metrics: {train_metrics}")
    print(f"Evaluation metrics: {eval_metrics}")

    save_model_info(output_path, fold_idx, learning_rate, weight_decay_rate)


def save_model_info(output_path, fold_idx, learning_rate, weight_decay_rate):
    """Saves the model information"""
    model_info = {
        "idx": fold_idx,
        "MODEL_ID": MODEL_ID,
        "ZOOM": ZOOM,
        "n_splits": 5,
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "learning_rate": learning_rate,
        "weight_decay_rate": weight_decay_rate,
    }
    with open(output_path / f"model_info_{fold_idx}.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4)
    print(json.dumps(model_info, indent=4))


def main():
    """Main function"""
    experiment_id = "DeiT_PT"
    fold_idx = 0
    learning_rate = 3e-5
    weight_decay_rate = 5e-3

    output_path = CWD / "results" / f"{ZOOM}x_{experiment_id}"

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        print(f"Directory {output_path} already exists. Skipping creation.")

    only_one_fold = False

    if only_one_fold:
        fold_idx = 0
        print(f"Starting experiment {experiment_id} with fold {fold_idx}. Hyperparams:")
        print(f"Learning rate: {learning_rate}")
        print(f"Weight decay rate: {weight_decay_rate}")
        run_fold(fold_idx, learning_rate, weight_decay_rate, output_path)
    else:
        for fold_idx in range(5):
            p = multiprocessing.Process(
                target=run_fold,
                args=(
                    fold_idx,
                    learning_rate,
                    weight_decay_rate,
                    output_path,
                ),
            )
            p.start()
            p.join()  # This will block until p finishes execution
            time.sleep(5)


if __name__ == "__main__":
    main()
