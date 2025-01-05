import random
import os
import torch
import numpy as np
from typing import Tuple, Union

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
    target_dir_path = os.path.join(target_dir)
    os.makedirs(target_dir_path, exist_ok=True)
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pth' or '.pt'"
    model_filepath = os.path.join(target_dir_path, model_name)
    torch.save(model.state_dict(), model_filepath)
    print(f"[INFO] Saved model to: {model_filepath}")


def hidden_helper(x, model, return_hidden=True, chunk_size=100):
    batches = int(np.ceil(x.shape[0] / chunk_size))
    num_hidden = model.num_hidden
    classes = model.classifier.out_features

    hidden = torch.zeros((x.shape[0], num_hidden))
    logits = torch.zeros((x.shape[0], classes))
    for i in range(batches):
        s = i * chunk_size
        t = min((i + 1) * chunk_size, x.shape[0])
        input_chunk = x[s:t, :]
        if return_hidden:
            logits_temp, hidden_temp = model(input_chunk, return_hidden=True)
            hidden[s:t, :] = hidden_temp
            logits[s:t, :] = logits_temp
        else:
            logits_temp = model(input_chunk)
            logits[s:t, :] = logits_temp
    if return_hidden:
        return logits, hidden
    else:
        return logits

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
            x_batch, return_covmat=True, return_hidden=True
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



# For each testing point, we choose 10 nearest points from training data
def compute_distance(train_hidden, test_hidden, k=10):
    distances = torch.norm(test_hidden.unsqueeze(1) - train_hidden.unsqueeze(0), dim=-1)
    # Get the top k smallest distances (use negative distances to get smallest)
    topk_distances, topk_indices = torch.topk(-distances, k=k, dim=1)
    return -topk_distances

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


# Example Usage:
if __name__ == '__main__':
    # Create some dummy data
# Example usage:
    x1 = torch.randn(1000, 128)  # 1000 training points with 128 features
    x2 = torch.randn(500, 128)   # 500 test points with 128 features
    distances = distance_helper(x1, x2, k=10)
    print(distances)
