import torch
from pathlib import Path
import random
import os
import numpy as np
import engine, model_builder, data_setup
from torch.utils.data import DataLoader

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

# Compute the hidden representations of the data (test, shift, OOD) with resnet model
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

# Compute the uncertainty (variance) and hidden representations of data (test, shift, OOD)
def compute_variance(x, sngp_model, flag=False, chunk_size=100):
    num_samples = x.shape[0]
    batches = int(np.ceil(num_samples / chunk_size))
    num_hidden = sngp_model.num_hidden
    classes = sngp_model.classifier.out_features

    hidden = torch.zeros((num_samples, num_hidden), dtype=torch.float32)
    vars_ = torch.zeros((num_samples, 1), dtype=torch.float32)
    logits = torch.zeros((num_samples, classes), dtype=torch.float32)
    covmat = torch.zeros((num_samples, num_samples), dtype=torch.float32) if flag else None

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
        vars_[start:end] = covmat_diag
        logits[start:end] = logits_temp

        if flag:
            covmat[start:end, start:end] = covmat_diag.squeeze(-1)

    # Compute mean variance
    mean_var = torch.mean(vars_).item()
    print(f'Mean variance -- {mean_var}')

    return (hidden, logits, vars_, covmat) if flag else (hidden, logits, vars_)

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

def cal_correlation(x1, x2):
    correlation_matrix = np.corrcoef(x1, x2)
    correlation_coefficient = correlation_matrix[0, 1]
    return correlation_coefficient

def concat_data(x1, x2, x3):
    return np.concatenate((x1, x2, x3), axis=0)

def get_all_test_dataloaders():
    _, test_data = data_setup.get_train_test_mnist()
    shift_data = data_setup.get_shifted_mnist(rotate_degs=2, roll_pixels=2)
    ood_data = data_setup.get_ood_mnist()
    test_loader = DataLoader(test_data, batch_size=1024, shuffle=False, drop_last=True)
    shift_loader = DataLoader(shift_data , batch_size=1024, shuffle=False, drop_last=True)
    ood_loader = DataLoader(ood_data, batch_size=1024, shuffle=False, drop_last=True)
    return test_loader, shift_loader, ood_loader

def mc_dropout(model, x, n_samples=5, return_hidden=False):
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

        if return_hidden:
            hiddens = torch.stack(hiddens, dim=0).mean(dim=0)
            return acc, uncertainty, hiddens
        else:
            return acc, uncertainty


