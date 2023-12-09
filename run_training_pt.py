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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
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


# MODEL_ID = "google/vit-base-patch16-224"
MODEL_ID = "facebook/convnext-base-224-22k"
# MODEL_ID = "facebook/deit-base-patch16-224"
# MODEL_ID = "facebook/deit-base-distilled-patch16-224"
# MODEL_ID = "microsoft/swin-base-patch4-window7-224-in22k"

IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_ID)
ZOOM = 400

# Define file paths
CWD = Path().absolute()
INPUT_PATH = CWD / f"breakhis_{ZOOM}x"

_transforms = transforms.Compose(
    [
        # transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomVerticalFlip(),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        # transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
    ]
)

to_tensor = transforms.ToTensor()
to_image = transforms.ToPILImage()


def process_example(image):
    """Preprocesses an image for the model"""
    inputs = IMAGE_PROCESSOR(
        to_image(_transforms(to_tensor(image))), return_tensors="pt"
    )
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


def load_data(fold_index):
    """Loads the dataset"""
    train_csv = str(INPUT_PATH / f"train_{fold_index}.csv")
    val_csv = str(INPUT_PATH / f"val_{fold_index}.csv")
    dataset = load_dataset(
        "csv", data_files={"train": train_csv, "val": val_csv}, keep_in_memory=True
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


EPOCHS = 50
# BATCH_SIZE = 64
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
    # auc = roc_auc_score(labels, logits[:, 1])  # For binary classification
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # "auc": auc,
    }


# Define the Trainer and train the model
def train_model(
    fold_idx, train_dataset, val_dataset, learning_rate, weight_decay_rate, output_path
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
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        # gradient_accumulation_steps=4, # slows down, saves memory
        # gradient_checkpointing=True, # slows down, saves memory
        dataloader_num_workers=CPU_THREADS,  # can this even help with current form of data loading?
        num_train_epochs=EPOCHS,
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
        # fp16=True,
        bf16=True,  # available only with +Ampere architecture (>=RTX 3000)
        tf32=True,
        torch_compile=True,
        do_train=True,
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="eval_loss",  # Choose the metric to monitor for early stopping
        greater_is_better=False,  # Early stop when the metric is decreasing
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=7, early_stopping_threshold=0.0
            ),
            PrinterCallback(),
            TensorBoardCallback(SummaryWriter()),
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
        "model_id": MODEL_ID,
        "zoom": ZOOM,
        "n_splits": 5,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": learning_rate,
        "weight_decay_rate": weight_decay_rate,
    }
    with open(output_path / f"model_info_{fold_idx}.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4)
    print(json.dumps(model_info, indent=4))


def main():
    """Main function"""
    # experiment_id = "ViT_PT_patches224_grid"
    experiment_id = "ConvNext_PT_patches224_grid"

    # learning_rate_values = [1e-5, 3e-5, 1e-4, 3e-4]
    # weight_decay_values = [1e-3, 5e-3, 1e-2]
    learning_rate_values = [3e-5]
    weight_decay_values = [5e-3]

    for learning_rate in learning_rate_values:
        for weight_decay_rate in weight_decay_values:
            lr_f = f"{learning_rate:.0e}".replace(".", "p")
            wdr_f = f"{weight_decay_rate:.0e}".replace(".", "p")
            cur_dt_f = datetime.now().strftime("%Y_%m_%d_%H_%M")
            output_path = (
                CWD
                / "results"
                / f"{ZOOM}x_{experiment_id}_lr_{lr_f}_wdr_{wdr_f}_{cur_dt_f}"
            )

            os.makedirs(output_path, exist_ok=True)
            print(f"Directory {output_path} created.")

            only_one_fold = False
            if only_one_fold:
                fold_idx = 0
                print(
                    f"Starting experiment {experiment_id} with fold {fold_idx}. Hyperparams:"
                )
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
