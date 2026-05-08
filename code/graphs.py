import json
import matplotlib.pyplot as plt
import os
import csv


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError("numpy is required for confusion matrix plotting") from e

    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def load_logs(result_dir="./results"):
    log_data = []
    
    checkpoint_dirs = [result_dir] + [os.path.join(result_dir, f) for f in os.listdir(result_dir) if f.startswith("checkpoint")]

    last_checkpoint_dir = max(checkpoint_dirs, key=os.path.getctime)

    log_file = os.path.join(last_checkpoint_dir, "trainer_state.json")

    with open(log_file, "r") as f:
        data = json.load(f)
        log_data.extend(data.get("log_history", []))  # Append log history if exists

    return log_data


def plot_loss_and_metrics(log_history, save_path="./graphs"):
    os.makedirs(save_path, exist_ok=True)

    steps = []
    train_loss = []
    eval_loss = []
    accuracy = []
    f1 = []

    for log in log_history:
        if "loss" in log and "eval_loss" not in log:
            steps.append(log["step"])
            train_loss.append(log["loss"])

        if "eval_loss" in log:
            eval_loss.append((log["step"], log["eval_loss"]))

        if "eval_accuracy" in log:
            accuracy.append((log["step"], log["eval_accuracy"]))

        if "eval_f1" in log:
            f1.append((log["step"], log["eval_f1"]))

    plt.figure()

    plt.plot(steps, train_loss, label="train_loss")

    if eval_loss:
        x, y = zip(*eval_loss)
        plt.plot(x, y, label="eval_loss")

    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(f"{save_path}/loss.png")
    plt.close()


    def plot_metric(metric, name):
        if metric:
            x, y = zip(*metric)
            plt.figure()
            plt.plot(x, y)
            plt.title(name)
            plt.savefig(f"{save_path}/{name}.png")
            plt.close()

    plot_metric(accuracy, "accuracy")
    plot_metric(f1, "f1")

def generate_all_plots(result_dir="./results", save_path="./graphs"):
    logs = load_logs(result_dir)
    plot_loss_and_metrics(logs, save_path)


def plot_fewshot_comparison(results_csv, save_path="./graphs", metric="f1"):
    """Plot a few-shot comparison curve from the CSV written by the notebook.

    Expected columns:
    few_shot_k, accuracy, precision, recall, f1, train_size, val_size, checkpoint_dir
    """
    os.makedirs(save_path, exist_ok=True)

    if metric not in {"accuracy", "precision", "recall", "f1"}:
        raise ValueError("metric must be one of: accuracy, precision, recall, f1")

    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Few-shot results CSV not found: {results_csv}")

    rows = []
    with open(results_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append(
                    {
                        "few_shot_k": int(row["few_shot_k"]),
                        "accuracy": float(row["accuracy"]),
                        "precision": float(row["precision"]),
                        "recall": float(row["recall"]),
                        "f1": float(row["f1"]),
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row in few-shot CSV: {row}") from exc

    if not rows:
        raise ValueError(f"No rows found in few-shot CSV: {results_csv}")

    rows.sort(key=lambda item: item["few_shot_k"])
    x = [item["few_shot_k"] for item in rows]
    y = [item[metric] for item in rows]

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.title(f"Few-shot comparison ({metric})")
    plt.xlabel("few_shot_k")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(save_path, f"fewshot_comparison_{metric}.png")
    plt.savefig(out_path)
    plt.close()

    return out_path