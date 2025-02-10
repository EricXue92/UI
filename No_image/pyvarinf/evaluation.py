from __future__ import print_function
import argparse
import numpy as np
import torch
from pathlib import Path
from data_setup import get_all_test_dataloaders
import pyvarinf
from utils import set_seed
from build_model import Build_MNISTClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

ROTATE_DEGS, ROLL_PIXELS = 2, 4
test_loader, shift_loader, ood_loader = get_all_test_dataloaders(batch_size=512, rotate_degs=ROTATE_DEGS, roll_pixels=ROLL_PIXELS) # 4


kwargs = {'num_workers': 4, 'pin_memory': True}
#
def shift_acc(model, data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y= X.to(device), y.to(device)
            logits = model(X)
            pred_y = torch.argmax(logits, dim=1)
            print(f"pred_y: {pred_y}")
            correct += (pred_y == y).sum().item()
    acc = correct / len(data_loader.dataset)
    print(f"Shifted Test Accuracy: {acc * 100:.2f}%")
    return acc


def evaluate_bnn_shift(model, data_loader, num_samples=50):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total_samples = 0
    uncertainties = []
    with torch.no_grad():  # Disable gradient computation for evaluation
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            batch_size = y.size(0)
            # Perform Monte Carlo sampling for BNN
            predictions = []
            for _ in range(num_samples):
                # Forward pass with weight sampling (BNN-specific)
                logits = model(X)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs)
            predictions = torch.stack(predictions, dim=0)
            predictive_mean = predictions.mean(dim=0)
            pred_y = torch.argmax(predictive_mean, dim=1)
            correct += (pred_y == y).sum().item()
            total_samples += batch_size
            std = predictions.var(dim=0).mean(dim=1)
            uncertainties.extend(std.cpu().numpy())
    accuracy = correct / total_samples
    uncertainties = np.array(uncertainties)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Mean Uncertainty: {np.mean(uncertainties):.4f}")
    return accuracy, uncertainties

def save_model(model, target_dir="model", model_name="vi_model.pth"):
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    model_save_path = target_dir_path / model_name
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

def load_model(model_path="model/vi_model.pth"):
    model = Build_MNISTClassifier(10)
    var_model = pyvarinf.Variationalize(model)
    var_model.load_state_dict(torch.load(model_path))
    var_model.to(device)
    return var_model

if __name__ == '__main__':
    model = load_model()
    for data in [test_loader, shift_loader, ood_loader]:
        evaluate_bnn_shift(model, data, num_samples=1000)

    # shift_acc(model, test_loader)
    # accuracy, uncertainties = evaluate_bnn(model, shift_loader, num_samples=10)






