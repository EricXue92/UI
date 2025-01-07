import torch
import data_setup, model_builder
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path="models/normal_model.pth"):
    model = model_builder.MNISTClassifier(10).to(device)
    model.load_state_dict(torch.load(model_path))
    return model

def get_acc(model, data_loader):
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

def get_ood_hidden():
    pass

def main():
    model = load_model()
    shift_data = data_setup.get_shifted_mnist(rotate_degs=2, roll_pixels=2)
    ood_data = data_setup.get_ood_mnist()
    get_acc(model, shift_data)

if __name__ == "__main__":
    main()





