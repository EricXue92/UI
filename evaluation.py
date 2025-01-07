import torch
import data_setup, model_builder, utils
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

seed = utils.set_seed(42)


def load_model(model_path="models/normal_model.pth"):
    model = model_builder.MNISTClassifier(10).to(device)
    model.load_state_dict(torch.load(model_path))
    return model

def get_shift_acc(model, data_loader):
    model.eval()
    correct = 0
    for barch, (X, y) in enumerate(data_loader):
        X = X.to(device)
        with torch.no_grad():
            logits = model(X)
            _, predicted = torch.max(logits, 1)
            y_pred = predicted.item()
            correct += np.sum(y_pred == y).item()
    acc = correct / len(data_loader.dataset)
    print(f"Accuracy: {acc:.4f}")
    return acc

def get_mc_results(n_smaples=10, hidden_flag=False):
    model = load_model().to(device)
    test_loader, shift_loader, ood_loader = utils.get_all_test_dataloaders()
    if hidden_flag:
        acc, uncertainty, hiddens = utils.mc_dropout(model, test_loader, n_samples=n_smaples, return_hidden=hidden_flag)
        print(f"Test Accuracy: {acc:.4f} | Uncertainty shape: {uncertainty.shape} | Hiddens: {hiddens.shape}")
    else:
        acc, uncertainty = utils.mc_dropout(model, test_loader, n_samples=n_smaples, return_hidden=hidden_flag)
        print(f"Test Accuracy: {acc:.4f} | Uncertainty shape: {uncertainty.shape}")

def get_deep_ensemble():
    pass

def get_ood_hidden():
    pass

def main():
    model = load_model()
    shift_data = data_setup.get_shifted_mnist(rotate_degs=2, roll_pixels=2)
    ood_data = data_setup.get_ood_mnist()
    # get_shift_acc(model, shift_data)
    get_mc_results(n_smaples=5, hidden_flag=False)


if __name__ == "__main__":
    main()





