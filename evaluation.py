import torch
import data_setup, model_builder, utils, train
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from utils import mc_dropout

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = utils.set_seed(42)

train_data, test_data = data_setup.get_train_test_mnist()
train_loader = DataLoader(train_data, batch_size=512, shuffle=True, drop_last=True) # 938
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

# For saved NN and SNGP models to get the shift accuracy
def shift_acc(model, data_loader):
    model.eval()
    correct = 0
    for batch_index, (X, y) in enumerate(data_loader):
        X = X.to(device)
        with torch.no_grad():
            logits = model(X)
            _, y_pred = torch.max(logits, 1)
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

# For nn and sngp hidden
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
    results["nn"] = get_hidden(model, dataset)
    results["sngp"] = get_hidden(sngp_model, dataset)
    results['dropout'] = utils.mc_dropout(model, dataset, return_hidden=True)['hiddens']
    results['deepensemble'] = train.get_deep_ensemble_results(dataset=dataset, return_hidden=True)["hiddens"]
    print(results)
    result_file_path = Path("results/all_hiddens.csv")
    utils.save_results_to_csv(results, result_file_path)
    return results

# Only need to get the SNGP uncertainty, dropout and deep ensemble are already done
def get_sngp_uncertainty(model=sngp_model, dataset=test_loader):
    uncertainties = []
    kwargs = {"return_covariance": True}
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataset):
            X = X.to(device)
            _, cov = model(X, return_hidden=False, **kwargs )
            uncertainty = torch.diag(cov)
            uncertainties.append(uncertainty)
    uncertainties = torch.cat(uncertainties, dim=0)
    print(f"Uncertainties : {uncertainties}")
    print(f"Uncertainty shape: {uncertainties.shape}")
    return uncertainties

def get_all_uncertainty(model=model, dataset=test_loader):
    sngp_uncertainty = get_sngp_uncertainty(sngp_model, dataset)
    dropout_uncertainty = utils.mc_dropout(model, dataset)['uncertainty']
    ensemble_uncertainty = train.get_deep_ensemble_results(dataset=dataset)["uncertainty"]
    print(f"MC Dropout Uncertainty shape: {dropout_uncertainty .shape} | SNGP Uncertainty shape: {sngp_uncertainty.shape} | DeepEnsemble Uncertainty shape: {ensemble_uncertainty.shape}")
    return sngp_uncertainty, dropout_uncertainty, ensemble_uncertainty

def get_mc_results(num_models=5, return_hidden=True):
    model = load_model().to(device)
    test_loader, shift_loader, ood_loader = data_setup.get_all_test_dataloaders()
    results = utils.mc_dropout(model, test_loader, num_models, return_hidden)
    print(f"Test Accuracy: {results['acc']:.4f} | Uncertainty shape: {results['uncertainty'].shape} | Hiddens: {results['hiddens']}")
    result_file_path = Path("results/mcdropout.csv")
    utils.save_results_to_csv(results, result_file_path)

def get_all_distance(dataset=test_loader):
    sngp_hidden_tr = get_hidden(sngp_model, train_loader)
    dropout_hidden_tr = utils.mc_dropout(model, train_loader, return_hidden=True)['hiddens']
    ensemble_hidden_tr = train.get_deep_ensemble_results(dataset=train_loader, return_hidden=True)["hiddens"]

    all_hiddens = get_all_hiddens(dataset)
    nn_hiddens, sngp_hiddens, dropout_hiddens, ensemble_hiddens = all_hiddens.values()

    sngp_dist = utils.distance_helper(sngp_hidden_tr, sngp_hiddens)
    sngp_dist = sngp_dist.mean(dim=1)
    print(f"sngp_dist shape: {sngp_dist.shape}")
    dropout_dist = utils.distance_helper(dropout_hidden_tr, dropout_hiddens)
    dropout_dist = dropout_dist.mean(dim=1)
    print(f"dropout_dist shape: {dropout_dist.shape}")
    ensemble_dist = utils.distance_helper(ensemble_hidden_tr, ensemble_hiddens)
    ensemble_dist = ensemble_dist.mean(dim=1)
    print(f"deepensemble_dist shape: {ensemble_dist.shape}")

    # print(f"SNGP Distance: {sngp_dist:.4f} | Dropout Distance: {dropout_dist:.4f} | DeepEnsemble Distance: {deepensemble_dist:.4f}")
    return sngp_dist, dropout_dist, ensemble_dist

def cal_correlation(x1, x2):
    x1, x2 = x1.cpu().numpy(), x2.cpu().numpy()
    correlation_matrix = np.corrcoef(x1, x2)
    correlation_coefficient = correlation_matrix[0, 1]
    return correlation_coefficient

def concat_data(x1, x2, x3):
    return np.concatenate((x1, x2, x3), axis=0)

def main():
    data_loaders = [test_loader, shift_loader, ood_loader]

    sngp_combined_uncertainty, dropout_combined_uncertainty, ensemble_combined_uncertainty = [], [], []
    sngp_combined_dist, dropout_combined_dist, ensemble_combined_dist = [], [], []

    for dataloader in data_loaders:
        sngp_uncertainty, dropout_uncertainty, ensemble_uncertainty = get_all_uncertainty(dataset=dataloader)
        sngp_dist, dropout_dist, ensemble_dist =get_all_distance(dataset=dataloader)

        sngp_corr = cal_correlation(sngp_uncertainty, sngp_dist)
        dropout_corr = cal_correlation(dropout_uncertainty, dropout_dist)
        ensemble_corr = cal_correlation(ensemble_uncertainty, ensemble_dist)
        print(f"sngp_test_corr: {sngp_corr:.4f} | dropout_test_corr: {dropout_corr:.4f} | ensemble_test_corr: {ensemble_corr:.4f}")

        sngp_combined_uncertainty.append(sngp_uncertainty)
        dropout_combined_uncertainty.append(dropout_uncertainty)
        ensemble_combined_uncertainty.append(ensemble_uncertainty)
        sngp_combined_dist.append(sngp_dist)
        dropout_combined_dist.append(dropout_dist)
        ensemble_combined_dist.append(ensemble_dist)

    sngp_combined_uncertainty = torch.cat(sngp_combined_uncertainty, axis=0)
    dropout_combined_uncertainty = torch.cat(dropout_combined_uncertainty, axis=0)
    ensemble_combined_uncertainty = torch.cat(ensemble_combined_uncertainty, axis=0)
    sngp_combined_dist = torch.cat(sngp_combined_dist, axis=0)
    dropout_combined_dist = torch.cat(dropout_combined_dist, axis=0)
    ensemble_combined_dist = torch.cat(ensemble_combined_dist, axis=0)

    sngp_combined_corr = cal_correlation(sngp_combined_uncertainty, sngp_combined_dist)
    dropout_combined_corr = cal_correlation(dropout_combined_uncertainty, dropout_combined_dist)
    ensemble_combined_corr = cal_correlation(ensemble_combined_uncertainty, ensemble_combined_dist)

    print(f"sngp_combined_corr: {sngp_combined_corr:.4f} | dropout_combined_corr: {dropout_combined_corr:.4f} | ensemble_combined_corr: {ensemble_combined_corr:.4f}")




if __name__ == "__main__":

    main()





