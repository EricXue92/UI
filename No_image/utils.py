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


# For each testing point, we choose 10 nearest points from training data
def compute_distance(train_hidden, test_hidden, k=10):
    distances = torch.norm(test_hidden.unsqueeze(1) - train_hidden.unsqueeze(0), dim=-1)
    # Get the top k smallest distances (use negative distances to get smallest)
    topk_distances, topk_indices = torch.topk(-distances, k=k, dim=1)
    return -topk_distances

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


def save_dict_to_csv(data_dict, file_path):
    df = pd.DataFrame(data_dict)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)


