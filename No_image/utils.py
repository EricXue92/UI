import torch
from pathlib import Path
import random
import os
import numpy as np
import engine, model_builder, data_setup
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def save_model(model, target_dir, model_name):
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    model_save_path = target_dir_path / model_name
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

def save_results_to_csv(results, result_file_path):
    os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
    results = {key: (value.cpu().numpy() if isinstance(value, torch.Tensor) else value if isinstance(value, (list, pd.Series)) else [value])
               for key, value in results.items() }
    df = pd.DataFrame(results)
    if not os.path.isfile(result_file_path):
        if len(df) == 3:
            df.index = ["Test", "Shift", "OOD"]
            df.to_csv(result_file_path, index=True, header=True)
        elif len(df) == 4:
            df.index = ["Test", "Shift", "OOD", "All over"]
            df.to_csv(result_file_path, index=True, header=True)
        else:
            df.to_csv(result_file_path, index=False, header=True)
    else:
        df.to_csv(result_file_path, mode='a', index=False, header=False)

def distance_helper(x1, x2, k=10):
    dist = torch.zeros((x2.shape[0], k), device=x1.device)
    # Define the chunk size (affects computational efficiency)
    chunk_size = 50
    for i in range(0, x2.shape[0], chunk_size):
        s, t = i, min(i + chunk_size, x2.shape[0])
        dist[s:t, :] = compute_distance(x1, x2[s:t, :], k=k)
    return dist

# For each testing point, we choose 10 nearest points from training data
def compute_distance(train_hidden, test_hidden, k=10):
    distances = torch.norm(test_hidden.unsqueeze(1) - train_hidden.unsqueeze(0), dim=-1)
    topk_distances, _ = torch.topk(-distances, k=k, dim=1)
    return -topk_distances

def mc_dropout(model, dataloader, n_samples=5, return_hidden=False):
    results = {}
    predictions, hiddens, all_labels = [], [], []
    model.train()  # Ensure dropout is active
    with torch.no_grad():
        for _ in range(n_samples):
            batch_preds, batch_hiddens = [], []
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                all_labels.append(y)
                logits, hidden = model(X, return_hidden=True) if return_hidden else (model(X), None)
                batch_preds.append(logits)
                if return_hidden:
                    batch_hiddens.append(hidden)
            predictions.append(torch.cat(batch_preds, dim=0))
            if return_hidden:
                hiddens.append(torch.cat(batch_hiddens, dim=0))

        predictions = torch.stack(predictions, dim=0)
        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0).mean(dim=1)
        pred_y = mean_prediction.argmax(dim=1)
        y_true = torch.cat(all_labels, dim=0)[:len(pred_y)]
        acc = (pred_y == y_true).float().mean().item()

        results = {"acc": round(acc, 4), "uncertainty": uncertainty}
        if return_hidden:
            results["hiddens"] = torch.stack(hiddens, dim=0).mean(dim=0)
        return results

def plot_distance_variance(test_mean_sngp, test_var_sngp, shift_mean_sngp, shift_var_sngp, OOD_mean_sngp, OOD_var_sngp,
                           test_mean_mc, test_var_mc, shift_mean_mc, shift_var_mc, OOD_mean_mc, OOD_var_mc,
                           test_mean_deep, test_var_deep, shift_mean_deep, shift_var_deep, OOD_mean_deep, OOD_var_deep,
                           filename='Dist-Var-comb.pdf'):
    methods = [
        ("(a) Proposed method", test_mean_sngp, test_var_sngp, shift_mean_sngp, shift_var_sngp, OOD_mean_sngp,
         OOD_var_sngp),
        ("(b) MC dropout", test_mean_mc, test_var_mc, shift_mean_mc, shift_var_mc, OOD_mean_mc, OOD_var_mc),
        ("(c) Deep ensemble", test_mean_deep, test_var_deep, shift_mean_deep, shift_var_deep, OOD_mean_deep,
         OOD_var_deep),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout='constrained')

    for ax, (title, test_mean, test_var, shift_mean, shift_var, OOD_mean, OOD_var) in zip(axes, methods):
        ax.scatter(test_mean, test_var, alpha=0.5, rasterized=True, label='Normal')
        ax.scatter(shift_mean, shift_var, s=10, alpha=0.2, rasterized=True, label='Shift')
        ax.scatter(OOD_mean, OOD_var, alpha=0.5, rasterized=True, label='OOD')
        ax.legend(fontsize=14)
        ax.set_xlabel('Distance in the hidden space', fontsize=16)
        ax.set_ylabel('Variance', fontsize=16)
        ax.set_title(title, fontweight="bold")

    plt.savefig(filename)
    plt.show()

def plot_hidden_distance_histograms(test_mean_inp, shift_mean_inp, OOD_mean_inp,
                                    test_mean_sngp, shift_mean_sngp, OOD_mean_sngp,
                                    test_mean_mc, shift_mean_mc, OOD_mean_mc,
                                    test_mean_deep, shift_mean_deep, OOD_mean_deep,
                                    filename='Hidden-Distance-comb.pdf'):
    sns.set(style="whitegrid", font_scale=1.5)

    methods = [
        ("(a) Histogram of distance", test_mean_inp, shift_mean_inp, OOD_mean_inp, 'Distance in the input space'),
        ("(b) Proposed method", test_mean_sngp, shift_mean_sngp, OOD_mean_sngp, 'Distance in the hidden space'),
        ("(c) MC dropout", test_mean_mc, shift_mean_mc, OOD_mean_mc, 'Distance in the hidden space'),
        ("(d) Deep ensemble", test_mean_deep, shift_mean_deep, OOD_mean_deep, 'Distance in the hidden space'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), layout='constrained')

    for ax, (title, test_mean, shift_mean, OOD_mean, xlabel) in zip(axes, methods):
        ax.hist(test_mean, bins=20, alpha=0.5, label='Normal')
        ax.hist(shift_mean, bins=20, alpha=0.5, label='Shift')
        ax.hist(OOD_mean, bins=20, alpha=0.5, label='OOD')
        ax.set_yscale('log', base=10)
        ax.legend(fontsize=14)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel('Frequency', fontsize=16)
        ax.set_title(title, fontweight="bold")

    plt.savefig(filename)
    plt.show()

def plot_variance_histograms(test_var_sngp, shift_var_sngp, OOD_var_sngp,
                             test_var_mc, shift_var_mc, OOD_var_mc,
                             test_var_deep, shift_var_deep, OOD_var_deep,
                             filename='Var-hist-comb.pdf'):
    methods = [
        ("(a) Proposed method", test_var_sngp, shift_var_sngp, OOD_var_sngp),
        ("(b) MC dropout", test_var_mc, shift_var_mc, OOD_var_mc),
        ("(c) Deep ensemble", test_var_deep, shift_var_deep, OOD_var_deep),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout='constrained')

    for ax, (title, test_var, shift_var, OOD_var) in zip(axes, methods):
        ax.hist(test_var, bins=20, alpha=0.5, label='Normal')
        ax.hist(shift_var, bins=20, alpha=0.5, label='Shift')
        ax.hist(OOD_var, bins=20, alpha=0.5, label='OOD')
        ax.set_yscale('log', base=10)
        ax.legend(fontsize=12)
        ax.set_xlabel('Variance', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.set_title(title, fontweight="bold")

    plt.savefig(filename)
    plt.show()

# Plot loss curves of a model
def plot_loss_curves(results):
    loss = results["train_loss"]
    val_loss = results["val_loss"]
    acc = results["train_acc"]
    val_acc = results["val_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, label="train_accuracy")
    plt.plot(epochs, val_acc, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.savefig('loss_curves.pdf')
    plt.show()

def main():
    pass


if __name__ == "__main__":
    main()