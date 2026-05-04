import json
import matplotlib.pyplot as plt
import os

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