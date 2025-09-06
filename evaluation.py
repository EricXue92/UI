from __future__ import print_function
import sys  # /home/xuelong/UI
import numpy as np
import torch
from pathlib import Path
# from data_setup import get_all_test_dataloaders

from train import parse_arguments
import pyvarinf
from build_model import Build_MNISTClassifier
import os
import pandas as pd

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/xuelong/UI')

import data_setup
from No_image import model_builder

from data import create_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"


ROTATE_DEGS, ROLL_PIXELS = 2, 4


args = parse_arguments()

test_loader, shift_loader, ood_loader = data_setup.get_all_test_dataloaders(batch_size=512, rotate_degs=ROTATE_DEGS, roll_pixels=ROLL_PIXELS) # 4


res = create_dataloaders()
input_dim, train_loader, val_loader, test_loader, shift_loader, ood_loader = (res["input_dim"], res["train"], res["val"],
                                                                              res["test"], res["shift"], res["ood"])

kwargs = {'num_workers': 4, 'pin_memory': True}
#
def shift_acc(model, data_loader):
    model.eval()
    total_samples = 0
    correct = 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y= X.to(device), y.to(device)
            batch_size = y.size(0)
            logits = model(X)
            if args.dataset == "MNIST":
                pred_y = torch.argmax(logits, dim=1)
            elif args.dataset == "Diabetes":
                pred_y = (torch.sigmoid(logits) > 0.5)
                print(f"pred_y shape : {pred_y.shape}")
            else:
                raise ValueError("Unknown dataset")
            correct += (pred_y == y).sum().item()
            total_samples += batch_size
    acc = correct / total_samples
    print(f"Each accuracy: {acc * 100:.2f}%")
    return acc

# with no ground truth labels for OOD data (y here is dummy variable)
def ood_acc(model, data_loader):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = torch.sigmoid(logits)
            all_preds.append(preds)
    all_preds = torch.cat(all_preds, dim=0)
    return all_preds


def evaluate_vi_model(models, data_loader):
    accuracies = []
    for model in models:
        acc = shift_acc(model, data_loader)
        accuracies.append(acc)
    mean_acc, var = np.mean(accuracies), np.var(accuracies)
    print(f"Mean Accuracy: {mean_acc * 100:.2f}%")
    print(f"Variance: {var:.4f}")
    return mean_acc, var

def save_model(model, target_dir="model", model_name="vi_model.pth"):
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    model_save_path = target_dir_path / model_name
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

def load_model(model_path="model/vi_model.pth"):
    if args.dataset == "MNIST":
        model = Build_MNISTClassifier(10)
    elif args.dataset == "Diabetes":
        model = model_builder.Build_DeepResNet(input_dim=input_dim)
    else:
        raise ValueError("Unknown dataset")

    var_model = pyvarinf.Variationalize(model)
    var_model.load_state_dict(torch.load(model_path))

    n_samples = 100
    models = [pyvarinf.Sample(var_model) for _ in range(n_samples)]
    for model in models:
        model.draw()
        model.to(device)
    return models

def save_results_to_csv(results, result_file_path):
    os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
    df = pd.DataFrame(results)
    df.index = ["Test", "Shift", "OOD"]
    if not os.path.isfile(result_file_path):
        df.to_csv(result_file_path, index=True, header=True)
    else:
        df.to_csv(result_file_path, mode='a', index=False, header=False)

if __name__ == '__main__':
    res = {"acc": [], "uncertainty": []}
    if args.dataset == "MNIST":
        models = load_model("model/vi_model_MNIST.pth")
        file_path = Path("results/vi_results_mnist.csv")
    else:
        models = load_model("model/vi_model_Diabetes.pth")
        file_path = Path("results/vi_results_diabetes.csv")

    for data in [test_loader, shift_loader, ood_loader ]:
        mean_acc, var = evaluate_vi_model(models, data)
        mean_acc, var = round(mean_acc, 4), round(var, 4)
        res["acc"].append(mean_acc)
        res["uncertainty"].append(var)

    save_results_to_csv(res, file_path)








