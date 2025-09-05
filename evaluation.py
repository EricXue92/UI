import os
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

from pathlib import Path
import numpy as np
import torch
from collections import defaultdict
import data_setup, model_builder, utils, train
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch.nn.functional as F
import random

from added_metrics import negative_log_likelihood, brier_score, expected_calibration_error
from utils import save_append_metric_results

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = utils.set_seed(42)
# seed = utils.set_seed(23)

train_data, test_data = data_setup.get_train_test_mnist()
train_loader = DataLoader(train_data, batch_size=512, shuffle=True, drop_last=True) # 938

# defined the shift data level
ROTATE_DEGS, ROLL_PIXELS = 2, 4
test_loader, shift_loader, ood_loader = data_setup.get_all_test_dataloaders(batch_size=1024, rotate_degs=ROTATE_DEGS, roll_pixels=ROLL_PIXELS) # 4


def load_model(model_path="checkpoints/normal_model_*.pth"):
    model = model_builder.Build_MNISTClassifier(10)
    model.load_state_dict(torch.load(model_path))
    return model

def load_sngp_model(model_path="models/sngp_model_*.pth"):
    sngp_model = model_builder.Build_SNGP_MNISTClassifier(10)
    print(f"Loading SNGP model from {model_path}")
    sngp_model.load_state_dict(torch.load(model_path))
    return sngp_model

def load_ensemble(model_path="models/ensemble_model_*.pth"):
    models = []
    for ckpt_path in sorted(Path().glob(model_path)):
        print(f"Loading model from {ckpt_path}")
        model = load_model(str(ckpt_path))
        model.to(device).eval()
        models.append(model)
    return models

# Load the pre-trained models
# model = load_model().to(device)
# ensemble_models = load_ensemble()
# sngp_model = load_sngp_model().to(device)
# sngp_model.classifier.update_covariance_matrix()
# sngp_ensemble_models = [load_sngp_model(f"models/sngp_model_{i}.pth").to(device) for i in range(5) ]

# Function to calculate the accuracy of a model on a given dataset
def shift_acc(model, data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y= X.to(device), y.to(device)
            logits = model(X)
            pred_y = torch.argmax(logits, dim=1)
            correct += (pred_y == y).sum().item()
    return correct / len(data_loader.dataset)

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

def get_all_hiddens(dataset=test_loader):
    results = {}
    results["nn"] = get_hidden_representation(model, dataset, model_type='nn')
    results["sngp"] = get_hidden_representation(sngp_model, dataset, model_type='sngp')
    results['dropout'] = utils.mc_dropout(model, dataset, return_hidden=True)['hiddens']
    results['deepensemble'] = train.get_deep_ensemble_results(dataset=dataset, return_hidden=True)["hiddens"]
    return results

# # Only need to get the SNGP uncertainty, dropout and deep ensemble are already done
# def get_sngp_uncertainty(model=sngp_model, dataset=test_loader):
#     uncertainties = []
#     kwargs = {'return_random_features': False, 'return_covariance': True,
#               'update_precision_matrix': False, 'update_covariance_matrix': False}
#     with torch.no_grad():
#         for batch, (X, y) in enumerate(dataset):
#             X = X.to(device)
#             _, cov = model(X, return_hidden=False, **kwargs)
#             uncertainty = torch.diag(cov)
#             uncertainties.append(uncertainty)
#     uncertainties = torch.cat(uncertainties, dim=0)
#     return uncertainties

# Only need to get the SNGP uncertainty, dropout and deep ensemble are already done
def get_sngp_uncertainty(model, dataset=test_loader, return_correctness=False):
    uncertainties, correctness = [], []
    kwargs = {'return_random_features': False, 'return_covariance': True,
              'update_precision_matrix': False, 'update_covariance_matrix': False}
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataset):
            X, y = X.to(device), y.to(device)
            logits, cov = model(X, return_hidden=False, **kwargs)
            uncertainty = torch.diag(cov)
            uncertainties.append(uncertainty)

            preds = torch.argmax(logits, dim=1)
            correct = (preds == y).float()
            correctness.append(1 - correct)

    if correctness and uncertainties:
        expected_size = correctness[0].shape[0]
        correctness = [c for c in correctness if c.shape[0] == expected_size]
        uncertainties = [u for u in uncertainties if u.shape[0] == expected_size]

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
    print(f"MC Dropout Uncertainty shape: {dropout_uncertainty .shape} | SNGP Uncertainty shape: {sngp_uncertainty.shape} | DeepEnsemble Uncertainty shape: {ensemble_uncertainty.shape}")
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
    sngp_dist = utils.distance_helper(sngp_hidden_tr, sngp_hiddens, k=50) # 100
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
        sngp_dist, dropout_dist, ensemble_dist =get_all_distance(dataset=dataloader)
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
    return slide_uq

def uncertainty_thershold(model, data_loader):
    uncertainties, correctness = get_sngp_uncertainty(model, data_loader, True)
    uncertainties = uncertainties.cpu().numpy()
    correctness =  correctness.cpu().numpy()
    s_fpr, s_tpr, s_thresh  = roc_curve(correctness, uncertainties)
    print(f"s_fpr: {len(s_fpr)}, s_tpr: {len(s_tpr)}, s_thresh: {len(s_thresh)}")
    J_index = s_tpr - s_fpr
    print(f"J_index:  {J_index}")
    max_j = max( zip(s_tpr, s_fpr), key=lambda x: x[0] - x[1] )
    slide_uq = s_thresh[list(zip(s_tpr, s_fpr)).index(max_j)]
    print(f"Slide uncertainty: {slide_uq}, quantile of low uncertainty: {np.mean(uncertainties<=slide_uq)}")
    return slide_uq, uncertainties, correctness, J_index

def plot_youden_j(uncertainties, correctness, J_index):
    if len(uncertainties) != len(correctness):
        raise ValueError("Uncertainties and correctness arrays must have the same length.")
    # Separate uncertainties based on correctness
    correct_uq = uncertainties[correctness == 1]
    incorrect_uq = uncertainties[correctness == 0]
    fig, ax = plt.subplots(figsize=(8, 6))
    kde_correct = gaussian_kde(correct_uq)
    x_correct = np.linspace(min(uncertainties), max(uncertainties), 200)  # Use overall min/max
    y_correct = kde_correct(x_correct)
    ax.plot(x_correct, y_correct, color='darkgreen', label='Correct')
    ax.fill_between(x_correct, y_correct, color='lightgreen', alpha=0.5)
    kde_incorrect = gaussian_kde(incorrect_uq)
    x_incorrect = np.linspace(min(uncertainties), max(uncertainties), 200)  # Use overall min/max
    y_incorrect = kde_incorrect(x_incorrect)
    ax.plot(x_incorrect, y_incorrect, color='saddlebrown', label='Incorrect')
    ax.fill_between(x_incorrect, y_incorrect, color='peru', alpha=0.5)
    kde_j = gaussian_kde(J_index)
    x_J = np.linspace(min(J_index), max(J_index))  # Use overall min/max
    y_J = kde_j(x_J)
    ax.plot(x_J, y_J, color='black', label='J Index')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0)  # Start y-axis at 0
    ax.set_xlim(min(uncertainties), max(uncertainties)) # consistent x limits
    plt.tight_layout()
    plt.show()


# For calculating metrics (nll, brier, and ece) for a single model (vanilla or sngp)
def evaluate_metrics(model, dataloader, device, n_bins, mode="vanilla", data_type="normal"):
    assert mode in {"vanilla", "mc_dropout", "sngp", "sngp_dropout"}
    if mode == "vanilla" or mode == "sngp":
        model.eval()
    elif mode == "mc_dropout" or mode == "sngp_dropout":
        model.train()
    else:
        print(f"Invalid mode: {mode}. Choose 'vanilla' or 'mc_dropout'.")
    logits_list, probs_list, labels_list = [], [], []

    with torch.no_grad():
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)

            if mode == "vanilla" or mode == "sngp":
                logits = model(X, return_hidden=False)
                probs = torch.softmax(logits, dim=1)
                logits_list.append(logits.cpu())
                probs_list.append(probs.cpu())

            elif mode == "mc_dropout":
                probs_accum = None
                for _ in range(5):
                    logits_mc = model(X, return_hidden=False)
                    if logits_mc.ndim == 1 or logits_mc.size(1) == 1:
                        probs_mc = torch.sigmoid(logits_mc).view(-1, 1)  # (N,1)
                    else:
                        probs_mc = F.softmax(logits_mc, dim=1)  # (N,K)
                    probs_accum = probs_mc if probs_accum is None else probs_accum + probs_mc
                probs_mean = probs_accum / 5  # predictive p
                probs_list.append(probs_mean.cpu())

                if probs_mean.ndim == 1 or probs_mean.size(1) == 1:  # binary
                    p_correct = probs_mean.view(-1).gather(0, y)
                else:
                    p_correct = probs_mean.gather(1, y.unsqueeze(1)).squeeze(1)
                logits_list.append(torch.log(p_correct + 1e-15).unsqueeze(1).cpu())

            elif mode == "sngp_dropout":
                probs_accum = None
                for _ in range(5):
                    logits_mc = model(X, return_hidden=False)
                    if logits_mc.ndim == 1 or logits_mc.size(1) == 1:
                        probs_mc = torch.sigmoid(logits_mc).view(-1, 1)
                    else:
                        probs_mc = F.softmax(logits_mc, dim=1)
                    probs_accum = probs_mc if probs_accum is None else probs_accum + probs_mc
                probs_mean = probs_accum / 5
                probs_list.append(probs_mean.cpu())
                if probs_mean.ndim == 1 or probs_mean.size(1) == 1:
                    p_correct = probs_mean.view(-1).gather(0, y)
                else:
                    p_correct = probs_mean.gather(1, y.unsqueeze(1)).squeeze(1)
                logits_list.append(torch.log(p_correct + 1e-15).unsqueeze(1).cpu())
            else:
                print(f"Invalid mode: {mode}")

            labels_list.append(y.cpu())

    # For NLL we still need logits or p(correct). The most numerically
    # stable way:  −log p̄_y  where p̄_y is the mean prob assigned to the true class.

    logits = torch.cat(logits_list, dim=0)
    probs = torch.cat(probs_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    # Negative log-likelihood
    if mode == "vanilla" or mode == "sngp":
        nll = negative_log_likelihood(logits, labels)
    elif mode == "mc_dropout" or mode == "sngp_dropout":
        # already have log p̄_y stored in logits (see above)
        nll = -logits.squeeze().mean().item()
    else:
        print(f"Invalid mode: {mode}")

    # Metrics
    brier = brier_score(probs, labels)
    # ece = adaptive_calibration_error(probs, labels, n_bins)
    ece = expected_calibration_error(probs, labels, n_bins)
    print(n_bins)
    print(f"{mode} | {data_type} | NLL: {nll:.4f} | Brier: {brier:.4f} | ECE: {ece:.4f}")
    results = {
        "mode": mode,
        "data_type": data_type,
        "nll": round(nll, 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4)
    }
    result_file_path = Path(f"results/{mode}_{data_type}_metrics.csv")
    save_append_metric_results(results, result_file_path)
    print(f"Results saved to {result_file_path}")

    return nll, brier, ece


# simple model to compute 10 times and report mean ± std (finished)
def evaluate_multiple_models(model_class, dataloader, device, n_models=10, n_bins=10, mode="vanilla", data_type="normal"):
    nlls, briers, eces = [], [], []
    for i in range(n_models):
        if mode == "vanilla" or mode == "mc_dropout" :
            ckpt_path = Path(f"checkpoints/normal_model_{i}.pth")
        elif mode == "sngp" or mode == "sngp_dropout":
            ckpt_path = Path(f"checkpoints/sngp_model_{i}.pth")
        else:
            raise ValueError("Invalid mode. Choose 'vanilla' or 'sngp'.")

        print(f"\n[INFO] Loading checkpoint: {ckpt_path}")
        # load model
        model = model_class().to(device)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        nll, brier, ece = evaluate_metrics(model, dataloader, device, n_bins=n_bins, mode=mode, data_type=data_type)
        nlls.append(nll)
        briers.append(brier)
        eces.append(ece)

    nlls = torch.tensor(nlls, dtype=torch.float32)
    briers = torch.tensor(briers, dtype=torch.float32)
    eces = torch.tensor(eces, dtype=torch.float32)

    nlls_mean, nlls_var = torch.mean(nlls).item(), nlls.var(unbiased=False).item()
    briers_mean, briers_var = torch.mean(briers).item(), torch.var(briers, unbiased=False).item()
    eces_mean, eces_var = torch.mean(eces).item(), torch.var(eces, unbiased=False).item()

    # convert to std
    nlls_std = nlls_var ** 0.5
    briers_std = briers_var ** 0.5
    eces_std = eces_var ** 0.5

    results = {
        "mode": "sngp",
        "data_type": data_type,
        "nll": f"{nlls_mean:.4f} ± {nlls_std:.4f}",
        "brier": f"{briers_mean:.4f} ± {briers_std:.4f}",
        "ece": f"{eces_mean:.4f} ± {eces_std:.4f}"
    }
    result_file_path = Path(f"results/{mode}_{data_type}_metrics.csv")
    print(f"Results saved to {result_file_path}")
    save_append_metric_results(results, result_file_path)
    return


def evaluate_bootstrapped_ensemble(model_class, dataloader, device, n_ensembles=5, mode="vanilla",
                                   ensemble_size=10, pool_size=20, n_bins=10, data_type="normal" ):
    nlls, briers, eces = [], [], []
    if mode == "vanilla" :
        model_paths = [Path(f"checkpoints/normal_model_{i}.pth") for i in range(pool_size)]
    elif mode == "sngp":
        model_paths = [Path(f"checkpoints/sngp_model_{i}.pth") for i in range(pool_size)]
    else:
        raise ValueError("Invalid mode. Choose 'vanilla' or 'sngp'.")

    for i in range(n_ensembles):
        print(f"\n[INFO] Evaluating bootstrapped ensemble {i + 1}/{n_ensembles}")
        chosen_paths = random.choices(model_paths, k=ensemble_size)
        print(f"Selected models: {[p.name for p in chosen_paths]}")

        ensemble_models = []
        for ckpt_path in chosen_paths:
            model = model_class().to(device)
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            ensemble_models.append(model)

        # evaluate this ensemble
        nll, brier, ece = evaluate_ensemble_helper(ensemble_models, dataloader, device, n_bins=n_bins)
        nlls.append(nll)
        briers.append(brier)
        eces.append(ece)

    nll_list = torch.tensor(nlls, dtype=torch.float32)
    brier_list = torch.tensor(briers, dtype=torch.float32)
    ece_list = torch.tensor(eces, dtype=torch.float32)

    # mean and std
    nll_mean, nll_std = nll_list.mean().item(), nll_list.std(unbiased=False).item()
    brier_mean, brier_std = brier_list.mean().item(), brier_list.std(unbiased=False).item()
    ece_mean, ece_std = ece_list.mean().item(), ece_list.std(unbiased=False).item()
    results = {
        "mode": "bootstrapped_ensemble",
        "data_type": data_type,
        "nll": f"{nll_mean:.4f} ± {nll_std:.4f}",
        "brier": f"{brier_mean:.4f} ± {brier_std:.4f}",
        "ece": f"{ece_mean:.4f} ± {ece_std:.4f}"
    }
    result_file_path = Path(f"results/bootstrapped_ensemble_{mode}_{data_type}_metrics.csv")
    print(f"Results saved to {result_file_path}")
    save_append_metric_results(results, result_file_path)
    return results

# def evaluate_ensemble_helper(ensemble_models, dataloader, device, n_bins=15):
#     logits_list, probs_list, labels_list = [], [], []
#     n = len(ensemble_models)
#     print(f"Evaluating {n} ensemble models.")
#
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             probs_accum = None
#
#             for model in ensemble_models:
#                 logits = model(X, return_hidden=False)
#                 if logits.ndim == 1 or logits.size(1) == 1: # binary
#                     probs = torch.sigmoid(logits).view(-1,1)
#                 else:
#                     probs = F.softmax(logits, dim=1)
#                 probs_accum = probs if probs_accum is None else probs_accum + probs
#
#             probs_mean = probs_accum / n
#             probs_list.append(probs_mean.cpu())
#             if probs_mean.ndim == 1 or probs_mean.size(1) == 1:  # binary
#                 p_correct = probs_mean.view(-1).gather(0, y)
#             else:
#                 p_correct = probs_mean.gather(1, y.unsqueeze(1)).squeeze(1)
#
#             logits_list.append(torch.log(p_correct + 1e-15).unsqueeze(1).cpu())
#             labels_list.append(y.cpu())
#
#     logits = torch.cat(logits_list, dim=0)
#     probs = torch.cat(probs_list, dim=0)
#     labels = torch.cat(labels_list, dim=0)
#
#     nll = -logits.squeeze().mean().item()
#     brier = brier_score(probs, labels)
#
#     ece = expected_calibration_error(probs, labels, n_bins)
#     # ece = adaptive_calibration_error(probs, labels, n_bins)
#     return round(nll, 4), round(brier, 4), round(ece, 4)

def evaluate_ensemble_helper(ensemble_models, dataloader, device, n_bins=15):
    all_probs, all_labels, nll_sum = [], [], 0.0
    n_samples = 0
    n = len(ensemble_models)
    per_model_probs = [[] for _ in range(n)]
    print(f"Evaluating {n} ensemble models.")
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            probs_accum = None
            for j, model in enumerate(ensemble_models):
                logits = model(X, return_hidden=False)
                if logits.ndim == 1 or logits.size(1) == 1: # binary
                    probs = torch.sigmoid(logits).view(-1,1)
                    probs_j = torch.stack([1.0 - probs, probs], dim=1)
                else:
                    probs = F.softmax(logits, dim=1)
                    probs_j = F.softmax(logits, dim=1)
                    # Save per-model probs (on CPU)
                per_model_probs[j].append(probs_j.detach().cpu())
                probs_accum = probs if probs_accum is None else probs_accum + probs
            probs_mean = probs_accum / n
            all_probs.append(probs_mean.cpu())
            all_labels.append(y.cpu())

            # calculate nll for the batch
            if probs_mean.size(1) == 1:
                p_true = torch.where(y==1, probs_mean.squeeze(), 1 - probs_mean.squeeze())
            else:
                p_true = probs_mean.gather(1, y.unsqueeze(1)).squeeze(1)

            nll_sum += -torch.log(p_true + 1e-15).sum().item()
            n_samples += y.size(0)

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    per_model_probs = [torch.cat(lst, dim=0) for lst in per_model_probs]

    nll = nll_sum / n_samples
    brier = brier_score(probs, labels)

    member_eces = [expected_calibration_error(pm, labels, n_bins) for pm in per_model_probs]
    ece_members_avg = sum(member_eces) / len(member_eces)
    ece_ens_mean = expected_calibration_error(probs, labels, n_bins)
    print(f"Ensemble ECE: {ece_ens_mean:.4f} | Members' avg ECE: {ece_members_avg:.4f}")
    return round(nll, 4), round(brier, 4), round(ece_members_avg, 4)





def main():

    # get_mc_results()
    # get_all_shift_acc()
    # get_all_corrs()
    # slide_uq, uncertainties, correctness, J_index  = uncertainty_thershold(sngp_model, test_loader)
    # plot_youden_j(uncertainties, correctness, J_index)
    # # sngp_uq = []

    # model, sngp_model, ensemble_models
    # data_loaders = [test_loader, shift_loxader]
    # "vanilla"


    model_class = lambda: model_builder.Build_MNISTClassifier(10)
    sngp_class = lambda: model_builder.Build_SNGP_MNISTClassifier(num_classes=10, coeff=1.0)


    # # # 10 single models
    evaluate_multiple_models(model_class, test_loader, device, n_models=5,
                             n_bins=15, mode="vanilla", data_type="normal")
    evaluate_multiple_models(model_class, shift_loader, device, n_models=5,
                             n_bins=15,mode="vanilla", data_type="shift")

    # 10 sngp single models
    evaluate_multiple_models(sngp_class, test_loader, device, n_models=5,
                             n_bins=15, mode="sngp", data_type="normal")
    evaluate_multiple_models(sngp_class, shift_loader, device, n_models=5,
                             n_bins=15,mode="sngp", data_type="shift")
    #
    # # # 10 mc dropout single models
    evaluate_multiple_models(model_class, test_loader, device, n_models=5,
                             n_bins=15, mode="mc_dropout", data_type="normal")
    evaluate_multiple_models(model_class, shift_loader, device, n_models=5,
                             n_bins=15, mode="mc_dropout", data_type="shift")

    # 10 sngp + dropout single models
    evaluate_multiple_models(sngp_class, test_loader, device, n_models=5,
                             n_bins=15,mode="sngp_dropout", data_type="normal")
    evaluate_multiple_models(sngp_class, shift_loader, device, n_models=5,
                             n_bins=15,mode="sngp_dropout", data_type="shift")


    # bootstrapped ensemble for vinillia
    evaluate_bootstrapped_ensemble(model_class, test_loader, device, n_ensembles=5,
                                   ensemble_size=10, pool_size=20, n_bins=15, data_type="normal")

    evaluate_bootstrapped_ensemble(model_class, shift_loader, device, n_ensembles=5,
                                   ensemble_size=10, pool_size=20, n_bins=15, data_type="shift")

    ## bootstrapped ensemble for sngp
    evaluate_bootstrapped_ensemble(sngp_class, test_loader, device, n_ensembles=5, mode="sngp",
                                   ensemble_size=10, pool_size=20, n_bins=15, data_type="normal")

    evaluate_bootstrapped_ensemble(sngp_class, shift_loader, device, n_ensembles=5, mode="sngp",
                                   ensemble_size=10, pool_size=20, n_bins=15, data_type="shift")



    # # for dataloader in data_loaders:
    # #     uq = get_sngp_uncertainty(model=sngp_model, dataset=dataloader)
    # #     sngp_uq.append(uq.cpu().numpy())
    # # print(np.mean(sngp_uq[0]))
    # # print(np.mean(sngp_uq[1]))
    # # print(np.mean(sngp_uq[2]))
    # # utils.plot_predictive_uncertainty(sngp_uq[0], sngp_uq[1], sngp_uq[2], Uq_threshold, 'UQ_Threshold.pdf')

if __name__ == "__main__":
    main()





