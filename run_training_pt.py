import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageClassification, AutoImageProcessor, TrainingArguments, Trainer
from datasets import load_dataset
from datetime import datetime
import json
from pathlib import Path
import os
import argparse
from PIL import Image
import numpy as np
import time
import multiprocessing

# Define the model architecture and image processor
# model_id = "microsoft/swin-base-patch4-window7-224-in22k"
# model_id = "google/vit-base-patch16-224"
# model_id = "facebook/deit-base-distilled-patch16-224"
model_id = "facebook/convnext-base-224-22k"
model_arch = AutoModelForImageClassification
image_processor = AutoImageProcessor.from_pretrained(model_id)

zoom = 400

# Define file paths
cwd = Path().absolute()
input_path = cwd / f'breakhis_{zoom}x'


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {"pixel_values": item['pixel_values'], "labels": item['label']}

# Assume train_data and val_data are your data, formatted as lists of dictionaries


# Load the data

def process_example(image):
    inputs = image_processor(image, return_tensors='pt')
    return inputs['pixel_values']


def process_dataset(example):
    example['pixel_values'] = process_example(Image.open(example['file_loc']).convert("RGB"))
    example['pixel_values'] = np.squeeze(example['pixel_values'], axis=0)
    # print(f"Shape of pixel_values: {example['pixel_values'].shape}")
    # example['label'] = to_categorical(example['label'], num_classes=2)
    return example

def remove_extra_dim(example):
    example['pixel_values'] = np.squeeze(example['pixel_values'], axis=0)
    return example


def load_data(fold_idx):
    train_csv = str(input_path / f"train_{fold_idx}.csv")
    val_csv = str(input_path / f"val_{fold_idx}.csv")
    dataset = load_dataset(
        'csv', data_files={'train': train_csv, 'val': val_csv})

    dataset = dataset.map(process_dataset, with_indices=False, num_proc=4)

    print(f"Loaded {fold_idx} dataset: {dataset}")

    return dataset


# def load_data(fold_idx):
#     train_csv = str(input_path / f"train_{fold_idx}.csv")
#     val_csv = str(input_path / f"val_{fold_idx}.csv")
#     dataset = load_dataset('csv', data_files={'train': train_csv, 'val': val_csv})
#     train_dataset = CustomDataset(dataset["train"])
#     val_dataset = CustomDataset(dataset["val"])
#     return train_dataset, val_dataset

# Define the training arguments
num_train_epochs = 50
batch_size = 16
num_warmup_steps = 0

id2label = {0: "benign", 1: "malignant"}
label2id = {v: k for k, v in id2label.items()}

# Define the Trainer and train the model
def train_model(fold_idx, train_dataset, val_dataset, learning_rate, weight_decay_rate, output_path):
    model = model_arch.from_pretrained(
        model_id,
        num_labels=len(id2label),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes = True,
    )

    # model.classifier = nn.Linear(model.classifier.in_features, 2)  # Assuming binary classification

    training_args = TrainingArguments(
        output_dir=str(output_path / f'model_{fold_idx}'),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay_rate,
        push_to_hub=False,
        logging_dir=str(output_path / f'logs_{fold_idx}'),
        log_level="debug",
        remove_unused_columns=False,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        use_cpu=False,
        fp16=True,
        # auto_find_batch_size=True,
        torch_compile=True,
        do_train=True,
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model='eval_loss',  # Choose the metric to monitor for early stopping
    )

    from transformers import EarlyStoppingCallback, PrinterCallback

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.0
    )
    printer_callback = PrinterCallback()

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        auc = roc_auc_score(labels, logits[:, 1])  # For binary classification
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback, printer_callback]
    )

    train_result = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # train_metrics = trainer.train()
    # trainer.log_metrics("train", train_metrics)
    # trainer.save_metrics("train", train_metrics)
    
    # trainer.save_model(output_path / f'model_{fold_idx}')
    eval_metrics = trainer.evaluate()
    trainer.save_metrics("eval", eval_metrics)
    return model, train_result.metrics, eval_metrics

# Define the main function to run the training process
def run_fold(fold_idx, learning_rate, weight_decay_rate, output_path):
    
    print(f"Starting experiment with fold {fold_idx}. Hyperparams:")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay rate: {weight_decay_rate}")

    dataset = load_data(fold_idx)

    train_dataset = dataset["train"]
    val_dataset = dataset["val"]
    
    # train_dataset = dataset["train"].map(remove_extra_dim)
    # val_dataset = dataset["val"].map(remove_extra_dim)
    model, train_metrics, eval_metrics = train_model(fold_idx, train_dataset, val_dataset, learning_rate, weight_decay_rate, output_path)
    print(f'Fold {fold_idx} finished')
    print(f'Train metrics: {train_metrics}')
    print(f'Evaluation metrics: {eval_metrics}')
    
    save_model_info(output_path, fold_idx, learning_rate, weight_decay_rate)

# Define the function to save the model information
def save_model_info(output_path, fold_idx, learning_rate, weight_decay_rate):
    model_info = {
        "idx": fold_idx,
        "model_id": model_id,
        "zoom": zoom,
        "n_splits": 5,
        "num_train_epochs": num_train_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay_rate": weight_decay_rate,
        "num_warmup_steps": num_warmup_steps,
    }
    with open(output_path / f'model_info_{fold_idx}.json', 'w') as f:
        json.dump(model_info, f, indent=4)
    print(json.dumps(model_info, indent=4))

# Define the argument parser and run the training process
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--experiment_id', type=str, help='Experiment ID', default='convnextXXXXXXXXXX')
    # parser.add_argument('--experiment_id', type=str, help='Experiment ID', default='ViT_PT_folds4')
    parser.add_argument('--experiment_id', type=str, help='Experiment ID', default='ConvNext_PT_folds')
    parser.add_argument('--fold_idx', type=int, help='Fold index', default=0)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=3e-5)
    parser.add_argument('--weight_decay_rate', type=float, help='Weight decay rate', default=5e-3)

    args = parser.parse_args()

    output_path = cwd / 'results' / f'{zoom}x_{args.experiment_id}'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        print(f"Directory {output_path} already exists. Skipping creation.")

    only_one_fold = False

    if only_one_fold:
        fold_idx = 0
        print(f"Starting experiment {args.experiment_id} with fold {fold_idx}. Hyperparams:")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Weight decay rate: {args.weight_decay_rate}")
        run_fold(fold_idx, args.learning_rate, args.weight_decay_rate, output_path)
    else:
        for fold_idx in range(5):
            p = multiprocessing.Process(target=run_fold, args=(fold_idx, args.learning_rate, args.weight_decay_rate, output_path))
            p.start()
            p.join()  # This will block until p finishes execution
            time.sleep(5)