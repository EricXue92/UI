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
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

def save_results_to_csv(results, result_file_path):
    os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
    df = pd.DataFrame([results])
    if not os.path.isfile(result_file_path):
        df.to_csv(result_file_path, index=False, header=True)  # Add header if file doesn't exist
    else:
        df.to_csv(result_file_path, mode='a', index=False, header=False)

# Compute the hidden representations of the data (test, shift, OOD)
def hidden_helper(x, model, chunk_size=100):
    batches = int(np.ceil(x.shape[0] / chunk_size))
    num_hidden = model.num_hidden
    classes = model.classifier.out_features
    hidden = torch.zeros((x.shape[0], num_hidden))
    logits = torch.zeros((x.shape[0], classes))
    for i in range(batches):
        s = i * chunk_size
        t = min((i + 1) * chunk_size, x.shape[0])
        input_chunk = x[s:t, :]
        logits_temp, hidden_temp = model(input_chunk, return_hidden=True)
        hidden[s:t, :] = hidden_temp
        logits[s:t, :] = logits_temp
    return logits, hidden

# Compute the distance between the hidden representations of the training and (testing, shift, OOD )
def distance_helper(x1, x2, k=10):
    dists = torch.zeros((x2.shape[0], k), dtype=torch.float32)
    chunk_size = 100
    batches = int(x2.shape[0] / chunk_size)
    for i in range(batches + 1):
        if i != batches:
            s = i * chunk_size
            t = (i + 1) * chunk_size
        else:
            s = i * chunk_size
            t = x2.shape[0]
        # Compute distances for the current chunk
        temp = compute_distance(x1, x2[s:t, :], k=k)
        dists[s:t, :] = temp
    return dists

# For each testing point, we choose 10 nearest points from training data
def compute_distance(train_hidden, test_hidden, k=10):
    distances = torch.norm(test_hidden.unsqueeze(1) - train_hidden.unsqueeze(0), dim=-1)
    # Get the top k smallest distances (use negative distances to get smallest)
    topk_distances, topk_indices = torch.topk(-distances, k=k, dim=1)
    return -topk_distances

def cal_correlation(x1, x2):
    correlation_matrix = np.corrcoef(x1, x2)
    correlation_coefficient = correlation_matrix[0, 1]
    return correlation_coefficient

def concat_data(x1, x2, x3):
    return np.concatenate((x1, x2, x3), axis=0)


def mc_dropout(model, x, n_samples=5, return_hidden=False):
    results = {}
    predictions, hiddens, all_labels = [], [], []
    model.to(device)
    model.train()  # Ensure dropout is active

    with torch.no_grad():
        for _ in range(n_samples):
            batch_preds, batch_hiddens = [], []
            for batch, (X, y) in enumerate(x):
                X, y = X.to(device), y.to(device)
                all_labels.append(y)
                if return_hidden:
                    logits, hidden = model(X, return_hidden=True)
                    batch_hiddens.append(hidden)
                else:
                    logits = model(X)
                batch_preds.append(logits)
            batch_preds = torch.cat(batch_preds, dim=0)  # Combine batches
            predictions.append(batch_preds)
            if return_hidden:
                batch_hiddens = torch.cat(batch_hiddens, dim=0)
                hiddens.append(batch_hiddens)

        predictions = torch.stack(predictions, dim=0)
        mean_prediction = predictions.mean(dim=0)
        mean_std = predictions.std(dim=0)
        uncertainty = torch.mean(mean_std, dim=1)

        pred_y = mean_prediction.argmax(dim=1)
        y_true = torch.cat(all_labels, dim=0)
        y_true = y_true[:len(pred_y)]
        acc = (pred_y == y_true).float().mean().item()

        results["acc"], results["uncertainty"]= acc, uncertainty
        print(f"Test Accuracy: {acc:.4f} | Uncertainty shape: {uncertainty.shape}")
        if return_hidden:
            hiddens = torch.stack(hiddens, dim=0).mean(dim=0)
            results["hiddens"] = hiddens.cpu().numpy()
        else:
            results["hiddens"] = None
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


def cal_correlation(x1, x2):
    correlation_matrix = np.corrcoef(x1, x2)
    correlation_coefficient = correlation_matrix[0, 1]
    return correlation_coefficient

def concat_data(x1, x2, x3):
    return np.concatenate((x1, x2, x3), axis = 0)

def analyze_model_uncertainty(model_name, test_mean, shift_mean, OOD_mean, test_var, shift_var, OOD_var):

    print(f'{model_name}:')
    model_dist = concat_data(test_mean, shift_mean, OOD_mean)

    # Handle potential shape differences for variance in Deep Ensemble
    if model_name == 'Deep ensemble':
        model_var = concat_data(test_var.ravel(), shift_var.ravel(), OOD_var.ravel())
    else:
        model_var = concat_data(test_var, shift_var, OOD_var)

    print(cal_correlation(model_dist, model_var))
    print(cal_correlation(test_mean,  test_var))
    print(cal_correlation(shift_mean, shift_var))
    print(cal_correlation(OOD_mean,   OOD_var))


# Compute the uncertainty (variance) and hidden representations of data (test, shift, OOD)
def compute_variance(x, sngp_model, flag=False, chunk_size=100):
    num_samples = x.shape[0]
    batches = int(np.ceil(num_samples / chunk_size))
    num_hidden = sngp_model.num_hidden
    classes = sngp_model.classifier.out_features
    hidden = torch.zeros((num_samples, num_hidden), dtype=torch.float32)
    variance = torch.zeros((num_samples, 1), dtype=torch.float32)
    logits = torch.zeros((num_samples, classes), dtype=torch.float32)
    for i in range(batches):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_samples)
        x_batch = x[start:end]
        (logits_temp, covmat_temp), hidden_temp = sngp_model(
            x_batch, kwargs={"update_precision_matrix": False, "return_covariance": True}, return_hidden=True
        )
        covmat_diag = torch.diag(covmat_temp).unsqueeze(1)
        # Store intermediate results
        hidden[start:end] = hidden_temp
        variance[start:end] = covmat_diag
        logits[start:end] = logits_temp
    # Compute mean variance
    mean_var = torch.mean(variance).item()
    print(f'Mean variance -- {mean_var}')
    return hidden, logits, variance


def main():
    pass


if __name__ == "__main__":
    main()