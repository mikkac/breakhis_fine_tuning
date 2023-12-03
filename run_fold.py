from transformers import TFAutoModelForImageClassification, AutoImageProcessor

# model_id = "facebook/convnext-base-224-22k"
model_id = "microsoft/swin-base-patch4-window7-224-in22k"

model_arch = TFAutoModelForImageClassification
image_processor = AutoImageProcessor.from_pretrained(model_id)

zoom = 400

image_processor

from datasets import load_dataset
from datetime import datetime
import json
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import shutil

import tensorflow as tf
import tensorflow_addons as tfa
from transformers import create_optimizer, DefaultDataCollator, ViTImageProcessor

cwd = Path().absolute()
input_path = cwd / f'breakhis_{zoom}x'

def process_example(image):
    inputs = image_processor(image, return_tensors='tf')
    return inputs['pixel_values']

def process_dataset(example):
    example['pixel_values'] = process_example(Image.open(example['file_loc']).convert("RGB"))

    # example['label'] = to_categorical(example['label'], num_classes=2)
    return example

def load_data(fold_idx):
    train_csv = str(input_path / f"train_{fold_idx}.csv")
    val_csv = str(input_path / f"val_{fold_idx}.csv")
    dataset = load_dataset(
        'csv', data_files={'train': train_csv, 'val': val_csv})

    dataset = dataset.map(process_dataset, with_indices=False, num_proc=1)

    print(f"Loaded {fold_idx} dataset: {dataset}")

    return dataset

id2label = {"0": "benign", "1": "malignant"}
label2id = {v: k for k, v in id2label.items()}

num_train_epochs = 150
batch_size = 40
batch_size = 40
num_warmup_steps = 0
fp16 = True

if fp16:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")


def get_loss():
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)


def get_metrics():
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name='auc', from_logits=True),
        # tf.keras.metrics.AUC(name='auc_multi', from_logits=True,
                            #  num_labels=2, multi_label=True),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision'),
        tfa.metrics.F1Score(name='f1_score', num_classes=2, threshold=0.5),
    ]


def get_callbacks(output_path, fold_idx):
    return [
        EarlyStopping(monitor="val_loss", patience=3),
        CSVLogger(output_path / f'train_metrics_{fold_idx}.csv')
    ]


def get_optimizer(learning_rate, weight_decay_rate, num_warmup_steps, num_train_steps):
    optimizer, _ = create_optimizer(
        init_lr=learning_rate,
        num_train_steps=num_train_steps,
        weight_decay_rate=weight_decay_rate,
        num_warmup_steps=num_warmup_steps,
    )

    return optimizer


num_train_steps_list = []
def train_model(fold_idx, train, val, learning_rate, weight_decay_rate, output_path):
    num_train_steps = len(train) * num_train_epochs
    num_train_steps_list.append(num_train_steps)
    print(f"num_train_steps = {num_train_steps}")
    optimizer = get_optimizer(
        learning_rate, weight_decay_rate, num_warmup_steps, num_train_steps)

    # load pre-trained ViT model
    model = model_arch.from_pretrained(
        model_id,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    # compile model
    model.compile(optimizer=optimizer, loss=get_loss(), metrics=get_metrics())
    history = model.fit(
        train,
        validation_data=val,
        callbacks=get_callbacks(output_path, fold_idx),
        epochs=num_train_epochs,
    )

    return model, history


def remove_extra_dim(example):
    example['pixel_values'] = np.squeeze(example['pixel_values'], axis=0)
    return example

def save_model(idx, model, output_path):
    model.save_pretrained(output_path / f'model_{idx}', from_tf=True)
    
def save_history(idx, history, output_path):
    np.save(output_path / f'train_history_{idx}.npy', history.history)

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def run_fold(fold_idx, learning_rate, weight_decay_rate, output_path):
    tf.keras.backend.clear_session()
    dataset = load_data(fold_idx)

    # Check patient ids uniqueness
    train_dataset = dataset["train"].map(remove_extra_dim)
    val_dataset = dataset["val"].map(remove_extra_dim)

    # Create datasets and train model
    data_collator = DefaultDataCollator(return_tensors="tf")

    train_dataset_tf = train_dataset.to_tf_dataset(
        columns=['pixel_values'],
        label_cols=['label'],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator
    )

    val_dataset_tf = val_dataset.to_tf_dataset(
        columns=['pixel_values'],
        label_cols=['label'],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator
    )
    print(train_dataset_tf)
    print(val_dataset_tf)

    model, history = train_model(fold_idx, train_dataset_tf, val_dataset_tf, learning_rate, weight_decay_rate, output_path)
    save_model(fold_idx, model, output_path)
    save_history(fold_idx, history, output_path)

    print(f'Fold {fold_idx} finished')


def save_model_info(output_path, fold_idx, learning_rate, weight_decay_rate):
    model_info = {"idx": fold_idx,
                    "model_id": model_id,
                    "zoom": zoom,
                    "n_splits": 5,
                    "num_train_epochs": num_train_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay_rate": weight_decay_rate,
                    "num_warmup_steps": num_warmup_steps,
                    "num_train_steps": num_train_steps_list[0]}

    with open(output_path / f'model_info_{fold_idx}.json', 'w') as f:
        json.dump(model_info, f, indent=4)

    print(json.dumps(model_info, indent=4))

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_id', type=str, help='Experiment ID')
    parser.add_argument('--fold_idx', type=int, help='Fold index')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--weight_decay_rate', type=float, help='Weight decay rate')

    args = parser.parse_args()

    output_path = cwd / 'results' / f'{zoom}x_{args.experiment_id}'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        print(f"Directory {output_path} already exists. Skipping creation.")

    print(f"Starting experiment {args.experiment_id} with fold {args.fold_idx}. Hyperparams:")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay rate: {args.weight_decay_rate}")
    run_fold(args.fold_idx, args.learning_rate, args.weight_decay_rate, output_path)
    save_model_info(output_path, args.fold_idx, args.learning_rate, args.weight_decay_rate)



