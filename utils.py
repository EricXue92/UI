import torch
from pathlib import Path
import random
import os
import numpy as np
import engine, model_builder
from torch.utils.data import DataLoader
import data_setup

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

