from datasets import Dataset, load_dataset
import glob
import os

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

import torch
import gc

import numpy as np


def _find_latest_subdir(parent_dir: str) -> str:
    if not os.path.isdir(parent_dir):
        raise FileNotFoundError(f"Directory not found: {parent_dir}")

    subdirs = [
        os.path.join(parent_dir, d)
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in: {parent_dir}")

    return max(subdirs, key=os.path.getctime)

# Legacy load datasets, if pre-split
def load_datasets(train_path, val_path, test_path, label2id):
    train_dataset = Dataset.from_file(train_path)
    val_dataset = Dataset.from_file(val_path)
    test_dataset = Dataset.from_file(test_path)

    print(f"Loaded datasets (train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)})")

    id2label = {v: k for k, v in label2id.items()}

    def encode_labels(example):
        example["label"] = label2id[example["label"]]
        return example

    train_dataset = train_dataset.map(encode_labels)
    val_dataset = val_dataset.map(encode_labels)
    test_dataset = test_dataset.map(encode_labels)

    return train_dataset, val_dataset, test_dataset, id2label


def load_datasets_from_hf(
    dataset_name: str,
    split: str,
    label2id: dict,
    train_size: float = 0.8,
    val_size: float = 0.1,
    seed: int = 42,
):

    dataset = load_dataset(dataset_name, split=split)

    split_1 = dataset.train_test_split(test_size=(1.0 - train_size), seed=seed)
    train_dataset = split_1["train"]
    rest = split_1["test"]

    rest_total = 1.0 - train_size
    if rest_total <= 0:
        raise ValueError("train_size must be < 1.0")

    test_fraction_of_rest = (1.0 - train_size - val_size) / rest_total
    if not (0.0 < test_fraction_of_rest < 1.0):
        raise ValueError("train_size and val_size must leave a positive test split.")

    split_2 = rest.train_test_split(test_size=test_fraction_of_rest, seed=seed)
    val_dataset = split_2["train"]
    test_dataset = split_2["test"]

    print(f"Loaded datasets (train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)})")

    id2label = {v: k for k, v in label2id.items()}

    def encode_labels(example):
        example["label"] = label2id[example["label"]]
        return example

    train_dataset = train_dataset.map(encode_labels)
    val_dataset = val_dataset.map(encode_labels)
    test_dataset = test_dataset.map(encode_labels)

    return train_dataset, val_dataset, test_dataset, id2label


def tokenize_dataset(dataset, tokenizer, max_length=128):
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    return dataset.map(tokenize, batched=True)

def tokenize_datasets(train_dataset, val_dataset, test_dataset, tokenizer, max_length=128):
    train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length)
    val_dataset = tokenize_dataset(val_dataset, tokenizer, max_length)
    test_dataset = tokenize_dataset(test_dataset, tokenizer, max_length)
    return train_dataset, val_dataset, test_dataset

def clear_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()

def load_model_and_tokenizer(model_name: str, num_labels: int):
    clear_cuda_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    return model, tokenizer


def load_trained_model_and_tokenizer(
    trained_model_dir: str,
    num_labels: int,
    device_map: str | None = None,
):
    """Load a *fine-tuned* model + tokenizer from disk."""
    clear_cuda_cache()

    tokenizer = AutoTokenizer.from_pretrained(trained_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        trained_model_dir,
        num_labels=num_labels,
        device_map=device_map,
    )
    return model, tokenizer


def load_latest_trained_model_and_tokenizer(
    trained_models_root: str = "./trained_models",
    num_labels: int = 2,
    device_map: str | None = None,
):
    """Load the most recently saved model from `trained_models_root`."""
    latest_dir = _find_latest_subdir(trained_models_root)
    model, tokenizer = load_trained_model_and_tokenizer(
        latest_dir,
        num_labels=num_labels,
        device_map=device_map,
    )
    return model, tokenizer, latest_dir


@torch.inference_mode()
def predict_labels(trainer: Trainer, dataset):
    """Run prediction and return (preds, labels)."""
    output = trainer.predict(dataset)
    logits = output.predictions
    labels = output.label_ids
    preds = np.argmax(logits, axis=-1)
    return preds, labels

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def build_trainer(model, train_dataset, val_dataset,
                  learning_rate=2e-5,
                  num_epochs=3,
                  per_device_train_batch_size=4,
                  per_device_eval_batch_size=4):

    args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    )

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

def clear_cache(path):
    for filename in glob.glob(f"{path}/cache-*"):
        os.remove(filename)

def clear_folder(path):
    for filename in glob.glob(f"{path}/*"):
        if os.path.isdir(filename):
            clear_folder(filename)
            os.rmdir(filename)
        else:
            os.remove(filename)