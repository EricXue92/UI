import os
gpu_choice = "1"   # or "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_choice

from pathlib import Path
import torch
import data_setup, model_builder, utils
import torch.nn.functional as F
import random
import pandas as pd
import time
import os
from added_metrics import negative_log_likelihood, brier_score, expected_calibration_error
from utils import save_append_metric_results, batch_ttests

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = utils.set_seed(42)

# defined the shift data level
ROTATE_DEGS, ROLL_PIXELS = 2, 4
test_loader, shift_loader, ood_loader = data_setup.get_all_test_dataloaders(batch_size=1024, rotate_degs=ROTATE_DEGS, roll_pixels=ROLL_PIXELS) # 4


# For calculating metrics (nll, brier, and ece) for a single model (vanilla or sngp)
def evaluate_metrics(model, dataloader, device, n_bins, mode="vanilla", data_type="normal"):
    assert mode in {"vanilla", "mc_dropout", "sngp", "sngp_dropout"}

    if mode in {"vanilla", "sngp"}:
        model.eval()
    elif mode in {"mc_dropout","sngp_dropout"}:
        model.train()
    else:
        print(f"Invalid mode: {mode}. Choose 'vanilla' or 'mc_dropout'.")
    logits_list, probs_list, labels_list = [], [], []
    with torch.no_grad():
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)
            if mode in {"vanilla","sngp"}:
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
    if mode in {"vanilla","sngp"}:
        nll = negative_log_likelihood(logits, labels)
    elif mode in {"mc_dropout", "sngp_dropout"}:
        # already have log p̄_y stored in logits (see above)
        nll = -logits.squeeze().mean().item()
    else:
        print(f"Invalid mode: {mode}")
    # Metrics
    brier = brier_score(probs, labels)
    ece = expected_calibration_error(probs, labels, n_bins)
    print(f"{mode} | {data_type} | NLL: {nll:.4f} | Brier: {brier:.4f} | ECE: {ece:.4f}")
    return float(nll) , float(brier) , float(ece)


# simple model to compute 10 times and report mean ± std (finished)
def evaluate_multiple_models(model_class, dataloader, device, n_models=10, n_bins=10,
                             mode="vanilla", data_type="normal"):
    nlls, briers, eces = [], [], []
    for i in range(n_models):
        if mode in {"vanilla", "mc_dropout"}:
            ckpt_path = Path(f"checkpoints/normal_model_{i}.pth")
        elif mode in {"sngp", "sngp_dropout"}:
            ckpt_path = Path(f"checkpoints/sngp_model_{i}.pth")
        else:
            raise ValueError("Invalid mode. Choose 'vanilla' or 'sngp'.")
        print(f"\n[INFO] Loading checkpoint: {ckpt_path}")

        # load model
        model = model_class().to(device)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        nll, brier, ece = evaluate_metrics(model, dataloader, device,
                                           n_bins=n_bins, mode=mode, data_type=data_type)
        nlls.append(nll)
        briers.append(brier)
        eces.append(ece)

    nlls = torch.round(torch.tensor(nlls, dtype=torch.float32), decimals=4)
    briers = torch.round(torch.tensor(briers, dtype=torch.float32), decimals=4)
    eces = torch.round(torch.tensor(eces, dtype=torch.float32), decimals=4)

    nlls_mean, nlls_std = torch.mean(nlls).item(), nlls.std(unbiased=False).item()
    briers_mean, briers_std = torch.mean(briers).item(), torch.std(briers, unbiased=False).item()
    eces_mean, eces_std = torch.mean(eces).item(), torch.std(eces, unbiased=False).item()

    df = pd.DataFrame({
        "model_id": list(range(n_models)),
        "nll": [f"{x:.4f}" for x in nlls.tolist()],
        "brier": [f"{x:.4f}" for x in briers.tolist()],
        "ece": [f"{x:.4f}" for x in eces.tolist()]
    })

    df.loc["mean±std"] = [
        "mean±std",
        f"{nlls_mean:.4f} ± {nlls_std:.4f}",
        f"{briers_mean:.4f} ± {briers_std:.4f}",
        f"{eces_mean:.4f} ± {eces_std:.4f}"
    ]
    result_file_path = Path(f"results/calibration_evaluation/{mode}_{data_type}_metrics.csv")
    result_file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(result_file_path, index=False)
    print(f"[INFO] Created new results file at {result_file_path}")

    return


def evaluate_bootstrapped_ensemble(model_class, dataloader, device,
                                   n_ensembles=5, mode="vanilla",
                                   ensemble_size=10, pool_size=20,
                                   n_bins=10, data_type="normal"):
    if mode == "vanilla":
        model_paths = [Path(f"checkpoints/normal_model_{i}.pth") for i in range(pool_size)]
    elif mode == "sngp":
        model_paths = [Path(f"checkpoints/sngp_model_{i}.pth") for i in range(pool_size)]
    else:
        raise ValueError("Invalid mode. Choose 'vanilla' or 'sngp'.")

    nlls, briers, eces = [], [], []
    records = []

    for i in range(n_ensembles):
        print(f"\n[INFO] Evaluating bootstrapped ensemble {i+1}/{n_ensembles}")
        chosen_paths = random.choices(model_paths, k=ensemble_size)
        print(f"Selected models: {[p.name for p in chosen_paths]}")

        # load models
        ensemble_models = []
        for ckpt_path in chosen_paths:
            model = model_class().to(device)
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            ensemble_models.append(model)

        # evaluate this ensemble
        nll, brier, ece = evaluate_ensemble_helper(ensemble_models, dataloader, device, n_bins)
        nlls.append(nll); briers.append(brier); eces.append(ece)

        records.append({
            "ensemble_id": i,
            "mode": "bootstrapped_ensemble",
            "data_type": data_type,
            "nll": nll,
            "brier": brier,
            "ece": ece
        })

    # summary
    nll_mean, nll_std = torch.tensor(nlls).mean().item(), torch.tensor(nlls).std(unbiased=False).item()
    brier_mean, brier_std = torch.tensor(briers).mean().item(), torch.tensor(briers).std(unbiased=False).item()
    ece_mean, ece_std = torch.tensor(eces).mean().item(), torch.tensor(eces).std(unbiased=False).item()

    summary = {
        "ensemble_id": "mean±std",
        "mode": "bootstrapped_ensemble",
        "data_type": data_type,
        "nll": f"{nll_mean:.4f} ± {nll_std:.4f}",
        "brier": f"{brier_mean:.4f} ± {brier_std:.4f}",
        "ece": f"{ece_mean:.4f} ± {ece_std:.4f}"
    }

    # Save all results once
    result_file_path = Path(f"results/calibration_evaluation/bootstrapped_ensemble_{mode}_{data_type}_metrics.csv")
    result_file_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records + [summary])
    df.to_csv(result_file_path, index=False)
    print(f"[INFO] Results saved to {result_file_path}")
    return summary


def evaluate_ensemble_helper(ensemble_models, dataloader, device, n_bins=15):
    all_probs, all_labels = [], []
    nll_sum, n_samples = 0.0, 0
    n = len(ensemble_models)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            probs_accum = 0.0

            for model in ensemble_models:
                logits = model(X, return_hidden=False)
                if logits.ndim == 1 or logits.size(1) == 1:  # binary case
                    probs = torch.sigmoid(logits).view(-1, 1)
                    probs = torch.cat([1 - probs, probs], dim=1)
                else:
                    probs = F.softmax(logits, dim=1)
                probs_accum += probs

            probs_mean = probs_accum / n
            all_probs.append(probs_mean.cpu())
            all_labels.append(y.cpu())
            # batch NLL
            p_true = probs_mean.gather(1, y.unsqueeze(1)).squeeze(1)
            nll_sum += -torch.log(p_true + 1e-15).sum().item()
            n_samples += y.size(0)
    # Concatenate results
    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    # Metrics
    nll = nll_sum / n_samples
    brier = brier_score(probs, labels)
    ece = expected_calibration_error(probs, labels, n_bins)
    return round(nll, 4), round(brier, 4), round(ece, 4)



def main():

    model_class = lambda: model_builder.Build_MNISTClassifier(10)
    sngp_class = lambda: model_builder.Build_SNGP_MNISTClassifier(num_classes=10, coeff=1.0)


    # # # 20 single models
    evaluate_multiple_models(model_class, test_loader, device, n_models=20,
                             n_bins=15, mode="vanilla", data_type="normal")
    evaluate_multiple_models(model_class, shift_loader, device, n_models=20,
                             n_bins=15,mode="vanilla", data_type="shift")

    # # 20 sngp single models
    evaluate_multiple_models(sngp_class, test_loader, device, n_models=20,
                             n_bins=15, mode="sngp", data_type="normal")
    evaluate_multiple_models(sngp_class, shift_loader, device, n_models=20,
                             n_bins=15,mode="sngp", data_type="shift")
    #
    # # # 20 mc dropout single models
    evaluate_multiple_models(model_class, test_loader, device, n_models=20,
                             n_bins=15, mode="mc_dropout", data_type="normal")
    evaluate_multiple_models(model_class, shift_loader, device, n_models=20,
                             n_bins=15, mode="mc_dropout", data_type="shift")

    # 20 sngp + dropout single models
    evaluate_multiple_models(sngp_class, test_loader, device, n_models=20,
                             n_bins=15,mode="sngp_dropout", data_type="normal")
    evaluate_multiple_models(sngp_class, shift_loader, device, n_models=20,
                             n_bins=15,mode="sngp_dropout", data_type="shift")


    # 5 bootstrapped ensemble for vinillia
    evaluate_bootstrapped_ensemble(model_class, test_loader, device, n_ensembles=5, mode="vanilla",
                                   ensemble_size=10, pool_size=20, n_bins=15, data_type="normal")
    evaluate_bootstrapped_ensemble(model_class, shift_loader, device, n_ensembles=5, mode="vanilla",
                                   ensemble_size=10, pool_size=20, n_bins=15, data_type="shift")

    # ## 5 bootstrapped ensemble for sngp
    evaluate_bootstrapped_ensemble(sngp_class, test_loader, device, n_ensembles=5, mode="sngp",
                                   ensemble_size=10, pool_size=20, n_bins=15, data_type="normal")
    evaluate_bootstrapped_ensemble(sngp_class, shift_loader, device, n_ensembles=5, mode="sngp",
                                   ensemble_size=10, pool_size=20, n_bins=15, data_type="shift")


    pairs = [
        ("results/calibration_evaluation/vanilla_normal_metrics.csv", "results/calibration_evaluation/sngp_normal_metrics.csv", "normal"),
        ("results/calibration_evaluation/vanilla_shift_metrics.csv",  "results/calibration_evaluation/sngp_shift_metrics.csv",  "shift"),

        ("results/calibration_evaluation/mc_dropout_normal_metrics.csv", "results/calibration_evaluation/sngp_normal_metrics.csv", "normal"),
        ("results/calibration_evaluation/mc_dropout_shift_metrics.csv",  "results/calibration_evaluation/sngp_shift_metrics.csv",  "shift"),

        ("results/calibration_evaluation/bootstrapped_ensemble_vanilla_normal_metrics.csv", "results/calibration_evaluation/sngp_normal_metrics.csv", "normal"),
        ("results/calibration_evaluation/bootstrapped_ensemble_vanilla_shift_metrics.csv", "results/calibration_evaluation/sngp_shift_metrics.csv", "shift"),

        ("results/calibration_evaluation/mc_dropout_normal_metrics.csv", "results/calibration_evaluation/sngp_dropout_normal_metrics.csv", "normal"),
        ("results/calibration_evaluation/mc_dropout_shift_metrics.csv", "results/calibration_evaluation/sngp_dropout_shift_metrics.csv", "shift"),

        ("results/calibration_evaluation/bootstrapped_ensemble_vanilla_normal_metrics.csv","results/calibration_evaluation/bootstrapped_ensemble_sngp_normal_metrics.csv", "normal"),
        ("results/calibration_evaluation/bootstrapped_ensemble_vanilla_shift_metrics.csv", "results/calibration_evaluation/bootstrapped_ensemble_sngp_shift_metrics.csv", "shift"),
    ]

    time.sleep(2)
    batch_ttests(pairs)
    #
    df = pd.read_csv("results/ttest_results.csv")

    # keep only relevant columns
    cols = ["label", "baseline", "proposed", "nll_pval", "brier_pval", "ece_pval"]
    table = df[cols].copy()

    # format to 4 decimals and bold if <0.05
    def fmt_p(p):
        if pd.isna(p): return "—"
        p = float(p)
        return f"**{p:.4f}**" if p < 0.05 else f"{p:.4f}"

    for m in ["nll_pval", "brier_pval", "ece_pval"]:
        table[m] = table[m].apply(fmt_p)

    print(table.to_markdown(index=False))

    saved_path = "results/calibration_evaluation/ttest_results_formatted.csv"
    table.to_csv(saved_path, index=False)
    print(f"[INFO] Formatted table saved to {saved_path}")


if __name__ == "__main__":
    main()





