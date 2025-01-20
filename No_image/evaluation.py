from pathlib import Path
import numpy as np
import torch
from collections import defaultdict
import data_setup, model_builder, utils, train
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import roc_curve

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = utils.set_seed(42)

res = data_setup.create_dataloaders()
input_dim, train_loader, val_loader, test_loader, shift_loader, ood_loader = (res["input_dim"], res["train"], res["val"],
                                                                              res["test"], res["shift"], res["ood"])
loss_fn = nn.BCEWithLogitsLoss()

def load_model(model_path="models/normal_model.pth"):
    model = model_builder.Build_DeepResNet(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    return model

def load_sngp_model(model_path="models/sngp_model.pth"):
    sngp_model = model_builder.Build_SNGP_DeepResNet(input_dim=input_dim)
    sngp_model.load_state_dict(torch.load(model_path))
    return sngp_model

model = load_model().to(device)
sngp_model = load_sngp_model().to(device)
sngp_model.classifier.update_covariance_matrix()

# For saved NN and SNGP models to get the shift accuracy
def shift_acc(model, data_loader):
    total_acc = 0
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            X, y= X.to(device), y.to(device)
            logits = model(X)
            preds = torch.sigmoid(logits) > 0.5
            total_acc += (preds.squeeze(1) == y).float().mean().item()
    avg_acc = round(total_acc / len(data_loader), 4)
    return avg_acc

# Get all shift acc
def get_all_shift_acc():
    results = {}
    results["nn"] = shift_acc(model, shift_loader)
    results["sngp"] = shift_acc(sngp_model, shift_loader)
    results['dropout'] = utils.mc_dropout(model, shift_loader)['acc']
    results['deepensemble'] = train.get_deep_ensemble_results(dataset=shift_loader)["acc"]
    result_file_path = Path("results/all_shift_acc.csv")
    utils.save_results_to_csv(results, result_file_path)


def get_hidden_representation(model, dataloader, model_type='nn'):
    hiddens = []
    # For SNGP, specific evaluation kwargs
    eval_kwargs = {
        'return_random_features': True,
        'return_covariance': False,
        'update_precision_matrix': False,
        'update_covariance_matrix': False
    }
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            if model_type == 'nn':
                logits, hidden = model(X, return_hidden=True)  # For regular NN
            elif model_type == 'sngp':
                logits, hidden = model(X, **eval_kwargs)  # For SNGP
            else:
                raise ValueError("Invalid model_type. Choose 'nn' or 'sngp'.")
            hiddens.append(hidden)
    hiddens = torch.cat(hiddens, dim=0)
    return hiddens

# Return the hidden states for all models on test_loader, shift_loader, ood_loader
def get_all_hiddens(dataset=test_loader):
    results = {}
    results["nn"] = get_hidden_representation(model, dataset, model_type='nn')
    results["sngp"] = get_hidden_representation(sngp_model, dataset, model_type='sngp')
    results['dropout'] = utils.mc_dropout(model, dataset, return_hidden=True)['hiddens']
    results['deepensemble'] = train.get_deep_ensemble_results(dataset=dataset, return_hidden=True)["hiddens"]
    return results

# Only need to get the SNGP uncertainty, dropout and deep ensemble are already done
def get_sngp_uncertainty(model=sngp_model, dataset=test_loader, return_correctness=False):
    uncertainties, correctness = [], []
    kwargs = {'return_random_features': False, 'return_covariance': True,
              'update_precision_matrix': False, 'update_covariance_matrix': False}
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataset):
            batch_correct = []
            X, y = X.to(device), y.to(device)
            logits, cov = model(X, return_hidden=False, **kwargs)
            uncertainty = torch.diag(cov)
            uncertainties.append(uncertainty)
            preds = torch.sigmoid(logits) > 0.5
            correct = (preds == y).float().squeeze(1)
            correctness.append(1 - correct)
    uncertainties = torch.cat(uncertainties, dim=0)
    correctness = torch.cat(correctness, dim=0)
    print(f"SNGP Uncertainty shape: {uncertainties.shape}")
    print(f"Correctness shape: {correctness.shape}")
    if not return_correctness:
        return uncertainties
    else:
        return uncertainties, correctness

def get_all_uncertainty(dataset):
    sngp_uncertainty = get_sngp_uncertainty(model=sngp_model, dataset=dataset)
    dropout_uncertainty = utils.mc_dropout(model, dataset)['uncertainty']
    ensemble_uncertainty = train.get_deep_ensemble_results(dataset=dataset)["uncertainty"]
    return sngp_uncertainty, dropout_uncertainty, ensemble_uncertainty

def get_mc_results(num_models=5, return_hidden=False):
    acc_result = {}
    results = utils.mc_dropout(model, test_loader, num_models, return_hidden)
    acc_result["mcdropout"] = round(results['acc'], 4)
    # print(f"Test Accuracy: {results['acc']:.4f} | Uncertainty shape: {results['uncertainty'].shape}")
    result_file_path = Path("results/mcdropout.csv")
    utils.save_results_to_csv(acc_result , result_file_path)

def get_all_distance(dataset=test_loader):
    # Get the train hiddens for all models
    sngp_hidden_tr = get_hidden_representation(sngp_model, train_loader, model_type='sngp')
    dropout_hidden_tr = utils.mc_dropout(model, train_loader, return_hidden=True)['hiddens']
    ensemble_hidden_tr = train.get_deep_ensemble_results(dataset=train_loader, return_hidden=True)["hiddens"]

    all_hiddens = get_all_hiddens(dataset)
    sngp_hiddens, dropout_hiddens, ensemble_hiddens =  all_hiddens['sngp'], all_hiddens['dropout'], all_hiddens['deepensemble']
    sngp_dist = utils.distance_helper(sngp_hidden_tr, sngp_hiddens, k=10) # 100
    sngp_dist = sngp_dist.mean(dim=1)
    print(f"sngp_dist shape: {sngp_dist.shape}")
    dropout_dist = utils.distance_helper(dropout_hidden_tr, dropout_hiddens)
    dropout_dist = dropout_dist.mean(dim=1)
    print(f"dropout_dist shape: {dropout_dist.shape}")
    ensemble_dist = utils.distance_helper(ensemble_hidden_tr, ensemble_hiddens)
    ensemble_dist = ensemble_dist.mean(dim=1)
    print(f"ensemble_dist shape: {ensemble_dist.shape}")
    return sngp_dist, dropout_dist, ensemble_dist

def cal_correlation(x1, x2):
    x1, x2 = x1.cpu().numpy(), x2.cpu().numpy()
    correlation_matrix = np.corrcoef(x1, x2)
    correlation_coefficient = correlation_matrix[0, 1]
    return correlation_coefficient

def get_all_corrs():
    res_corr, res_uncertainty = defaultdict(list), defaultdict(list)
    file_path_corr, file_path_uncertainty = Path("results/corr.csv"), Path("results/mean_uncertainty.csv")

    data_loaders = [test_loader, shift_loader, ood_loader]
    sngp_combined_uncertainty, dropout_combined_uncertainty, ensemble_combined_uncertainty = [], [], []
    sngp_combined_dist, dropout_combined_dist, ensemble_combined_dist = [], [], []

    for dataloader in data_loaders:
        sngp_uncertainty, dropout_uncertainty, ensemble_uncertainty = get_all_uncertainty(dataset=dataloader)
        res_uncertainty["sngp_uncertainty"].append(round(sngp_uncertainty.mean().item(), 4))
        res_uncertainty["dropout_uncertainty"].append(round(dropout_uncertainty.mean().item(),4))
        res_uncertainty["ensemble_uncertainty"].append(round(ensemble_uncertainty.mean().item(),4))

        sngp_dist, dropout_dist, ensemble_dist = get_all_distance(dataset=dataloader)
        sngp_corr = cal_correlation(sngp_uncertainty, sngp_dist)
        dropout_corr = cal_correlation(dropout_uncertainty, dropout_dist) ###
        ensemble_corr = cal_correlation(ensemble_uncertainty, ensemble_dist) ###
        print(f"sngp_corr: {sngp_corr:.4f} | dropout_corr: {dropout_corr:.4f} | ensemble_corr: {ensemble_corr:.4f}")

        res_corr["sngp_corr"].append(round(sngp_corr, 4))
        res_corr["dropout_corr"].append(round(dropout_corr, 4) )
        res_corr["ensemble_corr"].append(round(ensemble_corr, 4) )

        sngp_combined_uncertainty.append(sngp_uncertainty)
        dropout_combined_uncertainty.append(dropout_uncertainty)
        ensemble_combined_uncertainty.append(ensemble_uncertainty)

        sngp_combined_dist.append(sngp_dist)
        dropout_combined_dist.append(dropout_dist)
        ensemble_combined_dist.append(ensemble_dist)

    sngp_combined_uncertainty = torch.cat(sngp_combined_uncertainty, dim=0)
    dropout_combined_uncertainty = torch.cat(dropout_combined_uncertainty, dim=0)
    ensemble_combined_uncertainty = torch.cat(ensemble_combined_uncertainty, dim=0)

    sngp_combined_dist = torch.cat(sngp_combined_dist, dim=0)
    dropout_combined_dist = torch.cat(dropout_combined_dist, dim=0)
    ensemble_combined_dist = torch.cat(ensemble_combined_dist, dim=0)

    sngp_combined_corr = cal_correlation(sngp_combined_uncertainty, sngp_combined_dist)
    dropout_combined_corr = cal_correlation(dropout_combined_uncertainty, dropout_combined_dist)
    ensemble_combined_corr = cal_correlation(ensemble_combined_uncertainty, ensemble_combined_dist)

    res_corr["sngp_corr"].append( round(sngp_combined_corr, 4) )
    res_corr["dropout_corr"].append( round(dropout_combined_corr, 4) )
    res_corr["ensemble_corr"].append( round(ensemble_combined_corr, 4) )

    print(f"sngp_combined_corr: {sngp_combined_corr:.4f} | dropout_combined_corr: {dropout_combined_corr:.4f} "
          f"| ensemble_combined_corr: {ensemble_combined_corr:.4f}")
    utils.save_results_to_csv(res_corr, file_path_corr)
    utils.save_results_to_csv(res_uncertainty, file_path_uncertainty)

def uncertainty_thershold(model, data_loader):
    uncertainties, correctness = get_sngp_uncertainty(model, data_loader, True)
    uncertainties = uncertainties.cpu().numpy()
    correctness =  correctness.cpu().numpy()
    s_fpr, s_tpr, s_thresh  = roc_curve(correctness, uncertainties)
    max_j = max(zip(s_tpr, s_fpr), key=lambda x: x[0] - x[1])
    slide_uq = s_thresh[list(zip(s_tpr, s_fpr)).index(max_j)]
    print(f"Slide uncertainty: {slide_uq}, quantile of low uncertainty: {np.mean(uncertainties<=slide_uq)}")

def main():
    # get_mc_results()
    # get_all_shift_acc()
    get_all_corrs()
    # data_loaders = [test_loader, shift_loader, ood_loader]
    # for dataloader in data_loaders:
    #     get_sngp_uncertainty(model=sngp_model, dataset=dataloader)

    # uncertainty_thershold(sngp_model, test_loader)

if __name__ == "__main__":
    main()





