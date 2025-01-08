import torch
import data_setup, model_builder, utils, train
import numpy as np
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = utils.set_seed(42)

test_loader, shift_loader, ood_loader = data_setup.get_all_test_dataloaders()

def load_model(model_path="models/normal_model.pth"):
    model = model_builder.Build_MNISTClassifier(10)
    model.load_state_dict(torch.load(model_path))
    return model

def load_sngp_model(model_path="models/sngp_model.pth"):
    sngp_model = model_builder.Build_SNGP_MNISTClassifier(10)
    sngp_model.load_state_dict(torch.load(model_path))
    sngp_model.classifier.update_covariance_matrix()
    return sngp_model


model = load_model().to(device)
sngp_model = load_sngp_model().to(device)

# For NN and SNGP models
def shift_acc(model, data_loader):
    model.eval()
    correct = 0
    for barch, (X, y) in enumerate(data_loader):
        X = X.to(device)
        with torch.no_grad():
            logits = model(X)
            _, predicted = torch.max(logits, 1)
            y_pred = predicted
            correct += np.sum(y_pred.cpu().numpy() == y.cpu().numpy()).item()
    acc = correct / len(data_loader.dataset)
    print(f"Accuracy: {acc:.4f}")
    return acc

# Get all shift acc
def get_all_shift_acc():
    results = {}
    results["nn"] = shift_acc(model, shift_loader)
    results["sngp"] = shift_acc(sngp_model, shift_loader)
    results['dropout'] = utils.mc_dropout(model, shift_loader)['acc']
    results['deepensemble'] = train.get_deep_ensemble_results(dataset=shift_loader)["acc"]
    result_file_path = Path("results/all_shift_acc.csv")
    utils.save_results_to_csv(results, result_file_path)


def get_hidden(model, dataloader):
    hiddens = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            hidden = model(X, return_hidden=True)[1]
            hiddens.append(hidden)
    hiddens = torch.cat(hiddens, dim=0)
    print(f"Hidden shape: {hiddens.shape}")
    return hiddens


def get_all_hiddens(dataset=test_loader):
    results = {}
    results["nn"] = train.training_normal_model(num_classes=10, learning_rate=0.001, num_epochs=10, weight_decay=0.01)["hidden"]
    results['sngp'] = train.training_sngp_model(num_classes=10, coeff=3, learning_rate=0.001, num_epochs=10, weight_decay=0.01)["hidden"]
    results['dropout'] = utils.mc_dropout(model, test_loader, return_hidden=True)['hiddens']
    results['deepensemble'] = train.get_deep_ensemble_results(dataset=test_loader)["hiddens"]

    print(results)

def get_ood_hiddens():
    pass

def get_mc_results(num_models=5, return_hidden=True):
    model = load_model().to(device)
    test_loader, shift_loader, ood_loader = data_setup.get_all_test_dataloaders()
    results = utils.mc_dropout(model, test_loader, num_models, return_hidden)
    print(f"Test Accuracy: {results['acc']:.4f} | Uncertainty shape: {results['uncertainty'].shape} | Hiddens: {results['hiddens']}")
    result_file_path = Path("results/mcdropout.csv")
    utils.save_results_to_csv(results, result_file_path)

def main():
    get_hidden(model, test_loader)

if __name__ == "__main__":

    main()





