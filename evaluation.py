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

from added_metrics import negative_log_likelihood, brier_score, expected_calibration_error
from utils import save_append_results

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = utils.set_seed(42)

train_data, test_data = data_setup.get_train_test_mnist()
train_loader = DataLoader(train_data, batch_size=512, shuffle=True, drop_last=True) # 938

# defined the shift data level
ROTATE_DEGS, ROLL_PIXELS = 2, 4
test_loader, shift_loader, ood_loader = data_setup.get_all_test_dataloaders(batch_size=1024, rotate_degs=ROTATE_DEGS, roll_pixels=ROLL_PIXELS) # 4


def load_model(model_path="models/normal_model.pth"):
    model = model_builder.Build_MNISTClassifier(10)
    model.load_state_dict(torch.load(model_path))
    return model

def load_sngp_model(model_path="models/sngp_model.pth"):
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
ensemble_models = load_ensemble()
sngp_model = load_sngp_model().to(device)
sngp_model.classifier.update_covariance_matrix()
sngp_ensemble_models = [load_sngp_model(f"models/sngp_model_{i}.pth").to(device) for i in range(5)]


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
def get_sngp_uncertainty(model=sngp_model, dataset=test_loader, return_correctness=False):
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


def evaluate_metrics(model, dataloader, device, n_bins=15, mc_samples=5, mode="vanilla", data_type="normal"):
    assert mode in {"vanilla", "mc_dropout", "sngp"}
    if mode == "vanilla" or mode == "sngp":
        model.eval()
    elif mode == "mc_dropout":
        model.train()
    else:
        print(f"Invalid mode: {mode}. Choose 'vanilla' or 'mc_dropout'.")

    logits_list, probs_list, labels_list = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            if mode == "vanilla" or mode == "sngp":
                logits = model(X, return_hidden=False)
                logits_list.append(logits.cpu())
                # convert logits → probabilities
                if logits.ndim == 1 or logits.size(1) == 1:
                    probs = torch.sigmoid(logits).view(-1)
                else:
                    probs = torch.softmax(logits, dim=1)
                probs_list.append(probs.cpu())

            elif mode == "mc_dropout":
                # accumulate probabilities (not logits) to avoid overflow
                probs_accum = None
                for _ in range(mc_samples):
                    logits_mc = model(X, return_hidden=False)
                    if logits_mc.ndim == 1 or logits_mc.size(1) == 1:
                        probs_mc = torch.sigmoid(logits_mc).view(-1, 1)  # (N,1)
                    else:
                        probs_mc = F.softmax(logits_mc, dim=1)  # (N,K)
                    probs_accum = probs_mc if probs_accum is None else probs_accum + probs_mc
                probs_mean = probs_accum / mc_samples  # predictive p
                probs_list.append(probs_mean.cpu())
                # For NLL we still need logits or p(correct).  The most numerically
                # stable way:  −log p̄_y  where p̄_y is the mean prob assigned to the
                # true class.
                if probs_mean.ndim == 1 or probs_mean.size(1) == 1:  # binary
                    p_correct = probs_mean.view(-1).gather(0, y)
                else:
                    p_correct = probs_mean.gather(1, y.unsqueeze(1)).squeeze(1)
                logits_list.append(torch.log(p_correct + 1e-15).unsqueeze(1).cpu())  # fake logits = log p̄

            else:
                print(f"Invalid mode: {mode}")
            labels_list.append(y.cpu())

    logits = torch.cat(logits_list, dim=0)
    probs = torch.cat(probs_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    # Negative log-likelihood
    if mode == "vanilla" or mode == "sngp":
        nll = negative_log_likelihood(logits, labels)
    elif mode == "mc_dropout":
        # already have log p̄_y stored in logits (see above)
        nll = -logits.squeeze().mean().item()
    else:
        print(f"Invalid mode: {mode}")

    brier = brier_score(probs, labels)
    ece = expected_calibration_error(probs, labels, n_bins)

    print(f"{mode} | {data_type} | NLL: {nll:.4f} | Brier: {brier:.4f} | ECE: {ece:.4f}")

    results ={
        "mode": mode,
        "data_type": data_type,
        "nll": round(nll, 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4)
    }
    result_file_path = Path(f"results/ensemble_{mode}_{data_type}_metrics.csv")
    save_append_results(results, result_file_path)
    print(f"Results saved to {result_file_path}")
    # utils.save_results_to_csv(results, result_file_path)
    return nll, brier, ece

def evaluate_ensemble(ensemble_models, dataloader, device, n_bins=15, type="sngp", data_type="normal"):
    logits_list, probs_list, labels_list = [], [], []
    n = len(ensemble_models)
    print(f"Evaluating {n} ensemble models.")

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            probs_accum = None
            for model in ensemble_models:
                logits = model(X, return_hidden=False)
                if logits.ndim == 1 or logits.size(1) == 1:
                    probs = torch.sigmoid(logits).view(-1,1)
                else:
                    probs = F.softmax(logits, dim=1)
                probs_accum = probs if probs_accum is None else probs_accum + probs
            probs_mean = probs_accum / n
            probs_list.append(probs_mean.cpu())

            if probs_mean.ndim == 1 or probs_mean.size(1) == 1:  # binary
                p_correct = probs_mean.view(-1).gather(0, y)
            else:
                p_correct = probs_mean.gather(1, y.unsqueeze(1)).squeeze(1)

            logits_list.append(torch.log(p_correct + 1e-15).unsqueeze(1).cpu())
            labels_list.append(y.cpu())

    logits = torch.cat(logits_list, dim=0)
    probs = torch.cat(probs_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    nll = -logits.squeeze().mean().item()
    brier = brier_score(probs, labels)
    ece = expected_calibration_error(probs, labels, n_bins)
    print(f"Ensemble | NLL: {nll:.4f} | Brier: {brier:.4f} | ECE: {ece:.4f}")
    results = {
        "mode": "ensemble",
        "data_type": data_type,
        "nll": round(nll, 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4)
    }
    filepath= f"results/ensemble_{type}_{data_type}_metrics.csv"
    result_file_path = Path(filepath)
    utils.save_results_to_csv(results, result_file_path)
    return round(nll, 4), round(brier, 4), round(ece, 4)



# def evaluate_ensemble(ensemble_models, dataloader, device, n_bins=15, data_type="normal"):
#     n = len(ensemble_models)
#     probs_list, labels_list = [], []
#     for X,y in dataloader:
#         X, y = X.to(device), y.to(device)
#         # predictive probabilities
#         p_sum = None
#         for model in ensemble_models:
#             logits = model(X, return_hidden=False)
#             p_i = torch.sigmoid(logits).view(-1, 1) if logits.size(-1) == 1 \
#                 else F.softmax(logits, dim=1)
#             p_sum = p_i if p_sum is None else p_sum + p_i
#         p_bar = p_sum / n  # predictive p̄
#
#         probs_list.append(p_bar)
#         labels_list.append(y)
#
#     probs = torch.cat(probs_list, dim=0)
#     labels = torch.cat(labels_list, dim=0)
#
#     if probs.ndim == 1 or probs.size(1) == 1:  # binary
#         p_correct = torch.where(labels == 1, probs, 1. - probs)
#     else:  # multi-class
#         p_correct = probs[torch.arange(labels.numel(), device=device), labels]
#
#     nll = (-torch.log(p_correct.clamp_min(1e-15))).mean().item()  # scalar ↩︎ CPU
#     brier = brier_score(probs, labels)
#     ece = expected_calibration_error(probs, labels, n_bins)
#     results = {
#         "mode": "ensemble",
#         "data_type": data_type,
#         "nll": round(nll, 4),
#         "brier": round(brier, 4),
#         "ece": round(ece, 4)
#     }
#     result_file_path = Path("results/ensemble_metrics.csv")
#     utils.save_results_to_csv(results, result_file_path)
#     print(f"Ensemble | NLL: {nll:.4f} | Brier: {brier:.4f} | ECE: {ece:.4f}")
#     return round(nll, 4), round(brier, 4), round(ece, 4)


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

    # sngp ensemble
    # evaluate_ensemble(sngp_ensemble_models, test_loader, device, n_bins=15, type="sngp", data_type="normal")
    # evaluate_ensemble(sngp_ensemble_models, shift_loader, device, n_bins=15, type="sngp", data_type="shift")

    # sngp single model
    # evaluate_metrics(sngp_model, test_loader , device, n_bins=15, mc_samples=5, mode="sngp",data_type="normal")
    # evaluate_metrics(sngp_model, shift_loader, device, n_bins=15, mc_samples=5, mode="sngp", data_type="shift")

    # ensemble models
    evaluate_ensemble(ensemble_models, test_loader, device, n_bins=15, type="vanilla", data_type="normal")
    evaluate_ensemble(ensemble_models, shift_loader, device, n_bins=15, type="shift", data_type="normal")

    # # for dataloader in data_loaders:
    # #     uq = get_sngp_uncertainty(model=sngp_model, dataset=dataloader)
    # #     sngp_uq.append(uq.cpu().numpy())
    # # print(np.mean(sngp_uq[0]))
    # # print(np.mean(sngp_uq[1]))
    # # print(np.mean(sngp_uq[2]))
    # # utils.plot_predictive_uncertainty(sngp_uq[0], sngp_uq[1], sngp_uq[2], Uq_threshold, 'UQ_Threshold.pdf')

if __name__ == "__main__":
    main()





