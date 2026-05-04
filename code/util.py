from datasets import Dataset
import glob
import os

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

import torch
import gc

import numpy as np


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

def tokenize_dataset(dataset, tokenizer, max_length=128):
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    return dataset.map(tokenize, batched=True)

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
        report_to="none"  # keeps notebook clean
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
        os.remove(filename)
