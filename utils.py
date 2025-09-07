import torch
from pathlib import Path
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def save_model(model, target_dir, model_name):
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    model_save_path = target_dir_path / model_name
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

def save_results_to_csv(results, result_file_path):
    os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
    results = {key: (value.cpu().numpy() if isinstance(value, torch.Tensor) else value if isinstance(value, (list, pd.Series)) else [value])
               for key, value in results.items() }
    df = pd.DataFrame(results)
    if not os.path.isfile(result_file_path):
        if len(df) == 3:
            df.index = ["Test", "Shift", "OOD"]
            df.to_csv(result_file_path, index=True, header=True)
        elif len(df) == 4:
            df.index = ["Test", "Shift", "OOD", "All over"]
            df.to_csv(result_file_path, index=True, header=True)
        else:
            df.to_csv(result_file_path, index=False, header=True)
    else:
        df.to_csv(result_file_path, mode='a', index=False, header=False)

def distance_helper(x1, x2, k=200):
    dist = torch.zeros((x2.shape[0], k), device=x1.device)
    # Define the chunk size (affects computational efficiency)
    chunk_size = 50
    for i in range(0, x2.shape[0], chunk_size):
        s, t = i, min(i + chunk_size, x2.shape[0])
        dist[s:t, :] = compute_distance(x1, x2[s:t, :], k=k)
    return dist

# For each testing point, we choose 10 nearest points from training data
def compute_distance(train_hidden, test_hidden, k):
    distances = torch.norm(test_hidden.unsqueeze(1) - train_hidden.unsqueeze(0), dim=-1)
    topk_distances, _ = torch.topk(-distances, k=k, dim=1)
    return -topk_distances

def mc_dropout(model, dataloader, n_samples=5, return_hidden=False):
    predictions, hiddens, all_labels = [], [], []
    model.train()  # Ensure dropout is active
    with torch.no_grad():
        for _ in range(n_samples):
            batch_preds, batch_hiddens = [], []
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                all_labels.append(y)
                logits, hidden = model(X, return_hidden=True) if return_hidden else (model(X), None)
                batch_preds.append(logits)
                if return_hidden:
                    batch_hiddens.append(hidden)
            predictions.append(torch.cat(batch_preds, dim=0))
            if return_hidden:
                hiddens.append(torch.cat(batch_hiddens, dim=0))

        predictions = torch.stack(predictions, dim=0)
        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0).mean(dim=1)
        pred_y = mean_prediction.argmax(dim=1)
        y_true = torch.cat(all_labels, dim=0)[:len(pred_y)]
        acc = (pred_y == y_true).float().mean().item()

        results = {"acc": round(acc, 4), "uncertainty": uncertainty}
        if return_hidden:
            results["hiddens"] = torch.stack(hiddens, dim=0).mean(dim=0)
        return results

def plot_distance_variance(test_mean_sngp, test_var_sngp, shift_mean_sngp, shift_var_sngp, OOD_mean_sngp, OOD_var_sngp,
                           test_mean_mc, test_var_mc, shift_mean_mc, shift_var_mc, OOD_mean_mc, OOD_var_mc,
                           test_mean_deep, test_var_deep, shift_mean_deep, shift_var_deep, OOD_mean_deep, OOD_var_deep,
                           filename='Dist-Var-comb.pdf'):
    methods = [
        ("(a) Proposed method", test_mean_sngp, test_var_sngp, shift_mean_sngp, shift_var_sngp, OOD_mean_sngp,
         OOD_var_sngp),
        ("(b) MC dropout", test_mean_mc, test_var_mc, shift_mean_mc, shift_var_mc, OOD_mean_mc, OOD_var_mc),
        ("(c) Deep ensemble", test_mean_deep, test_var_deep, shift_mean_deep, shift_var_deep, OOD_mean_deep,
         OOD_var_deep),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout='constrained')

    for ax, (title, test_mean, test_var, shift_mean, shift_var, OOD_mean, OOD_var) in zip(axes, methods):
        ax.scatter(test_mean, test_var, alpha=0.5, rasterized=True, label='Normal')
        ax.scatter(shift_mean, shift_var, s=10, alpha=0.2, rasterized=True, label='Shift')
        ax.scatter(OOD_mean, OOD_var, alpha=0.5, rasterized=True, label='OOD')
        ax.legend(fontsize=14)
        ax.set_xlabel('Distance in the hidden space', fontsize=16)
        ax.set_ylabel('Variance', fontsize=16)
        ax.set_title(title, fontweight="bold")

    plt.savefig(filename)
    plt.show()

def plot_hidden_distance_histograms(test_mean_inp, shift_mean_inp, OOD_mean_inp,
                                    test_mean_sngp, shift_mean_sngp, OOD_mean_sngp,
                                    test_mean_mc, shift_mean_mc, OOD_mean_mc,
                                    test_mean_deep, shift_mean_deep, OOD_mean_deep,
                                    filename='Hidden-Distance-comb.pdf'):
    sns.set(style="whitegrid", font_scale=1.5)

    methods = [
        ("(a) Histogram of distance", test_mean_inp, shift_mean_inp, OOD_mean_inp, 'Distance in the input space'),
        ("(b) Proposed method", test_mean_sngp, shift_mean_sngp, OOD_mean_sngp, 'Distance in the hidden space'),
        ("(c) MC dropout", test_mean_mc, shift_mean_mc, OOD_mean_mc, 'Distance in the hidden space'),
        ("(d) Deep ensemble", test_mean_deep, shift_mean_deep, OOD_mean_deep, 'Distance in the hidden space'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), layout='constrained')

    for ax, (title, test_mean, shift_mean, OOD_mean, xlabel) in zip(axes, methods):
        ax.hist(test_mean, bins=20, alpha=0.5, label='Normal')
        ax.hist(shift_mean, bins=20, alpha=0.5, label='Shift')
        ax.hist(OOD_mean, bins=20, alpha=0.5, label='OOD')
        ax.set_yscale('log', base=10)
        ax.legend(fontsize=14)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel('Frequency', fontsize=16)
        ax.set_title(title, fontweight="bold")

    plt.savefig(filename)
    plt.show()

def plot_variance_histograms(test_var_sngp, shift_var_sngp, OOD_var_sngp,
                             test_var_mc, shift_var_mc, OOD_var_mc,
                             test_var_deep, shift_var_deep, OOD_var_deep,
                             filename='Var-hist-comb.pdf'):
    methods = [
        ("(a) Proposed method", test_var_sngp, shift_var_sngp, OOD_var_sngp),
        ("(b) MC dropout", test_var_mc, shift_var_mc, OOD_var_mc),
        ("(c) Deep ensemble", test_var_deep, shift_var_deep, OOD_var_deep),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout='constrained')

    for ax, (title, test_var, shift_var, OOD_var) in zip(axes, methods):
        ax.hist(test_var, bins=20, alpha=0.5, label='Normal')
        ax.hist(shift_var, bins=20, alpha=0.5, label='Shift')
        ax.hist(OOD_var, bins=20, alpha=0.5, label='OOD')
        ax.set_yscale('log', base=10)
        ax.legend(fontsize=12)
        ax.set_xlabel('Variance', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.set_title(title, fontweight="bold")

    plt.savefig(filename)
    plt.show()


def plot_predictive_uncertainty(test_var_sngp, shift_var_sngp, OOD_var_sngp, Uq_threshold,
                                save_path='UQ_Threshold.pdf'):
    sns.set(style="whitegrid", font_scale=1.5)
    # Create the figure and axis
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6), layout='constrained')
    # Plot histograms for each dataset
    ax1.hist(test_var_sngp, bins=20, alpha=0.5, label='Normal')
    ax1.hist(shift_var_sngp, bins=20, alpha=0.5, label='Shift')
    ax1.hist(OOD_var_sngp, bins=20, alpha=0.5, label='OOD')
    # Add vertical lines for specific statistics
    plt.axvline(x=test_var_sngp.max(), lw=2, color='black', ls='--', label =r'$U_{q}$')
    plt.axvline(x=Uq_threshold, lw=2, color='red', ls='--', label = r'$U_{YJ}$')
    # Print values for debugging (optional)
    print(f"Max Normal: {test_var_sngp.max()}")
    # print(f"Min Shift: {shift_var_sngp.min()}")
    # print(f"Min OOD: {OOD_var_sngp.min()}")
    # Customize plot
    ax1.legend(fontsize=20)
    ax1.set_xlabel('Predictive Uncertainty', fontsize=16)
    ax1.set_ylabel('Frequency', fontsize=16)
    # Save the plot
    plt.savefig(save_path)
    plt.show()
    plt.close()

def save_append_metric_results(results, result_file_path):
    os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
    results = {key: (value.cpu().numpy() if isinstance(value, torch.Tensor) else value if isinstance(value, (list, pd.Series)) else [value])
               for key, value in results.items() }
    df = pd.DataFrame(results)

    if not os.path.isfile(result_file_path):
        df.to_csv(result_file_path, index=False, header=True)
    else:
        df.to_csv(result_file_path, mode='a', index=False, header=False)


def ttest_from_csv(baseline_csv, proposed_csv, metrics=("nll","brier","ece") ):
    base = pd.read_csv(baseline_csv).iloc[:-1]
    prop = pd.read_csv(proposed_csv).iloc[:-1]
    res = {}
    for col in metrics:
        xb = pd.to_numeric(base[col], errors="coerce").dropna()
        xp = pd.to_numeric(prop[col], errors="coerce").dropna()

        # Need at least 2 samples per group for a t-test
        if len(xb) < 2 or len(xp) < 2:
            res[f"{col}_pval"] = np.nan
            continue

        t_stat, p_val = stats.ttest_ind(xp, xb, alternative="less", equal_var=False)
        res[f"{col}_pval"] = float(p_val)
    return res



def batch_ttests(pairs, out_csv="results/ttest_results.csv", alpha=0.05):
    """
    pairs: list of (baseline_csv, proposed_csv, label)
    Writes/append a summary CSV with p-values and boolean significance flags.
    """
    rows = []
    for baseline_csv, proposed_csv, label in pairs:
        bpath, ppath = Path(baseline_csv), Path(proposed_csv)

        # Skip missing files gracefully (and tell you which)
        if not bpath.exists() or not ppath.exists():
            rows.append({
                "label": label,
                "baseline": baseline_csv,
                "proposed": proposed_csv,
                "nll_pval": np.nan,
                "brier_pval": np.nan,
                "ece_pval": np.nan,
                "nll_sig": False,
                "brier_sig": False,
                "ece_sig": False,
                "any_sig": False,
                "note": f"Missing: "
                        f"{'baseline ' if not bpath.exists() else ''}"
                        f"{'proposed' if not ppath.exists() else ''}".strip()
            })
            continue

        res = ttest_from_csv(baseline_csv, proposed_csv)
        # boolean flags at alpha
        nll_p = res.get("nll_pval", np.nan)
        brier_p = res.get("brier_pval", np.nan)
        ece_p = res.get("ece_pval", np.nan)

        nll_sig = (nll_p == nll_p) and (nll_p < alpha)
        brier_sig = (brier_p == brier_p) and (brier_p < alpha)
        ece_sig = (ece_p == ece_p) and (ece_p < alpha)

        rows.append({
            "label": label,
            "baseline": baseline_csv,
            "proposed": proposed_csv,
            "nll_pval": nll_p,
            "brier_pval": brier_p,
            "ece_pval": ece_p,
            "nll_sig": nll_sig,
            "brier_sig": brier_sig,
            "ece_sig": ece_sig,
            "any_sig": bool(nll_sig or brier_sig or ece_sig),
            "note": ""
        })

    df = pd.DataFrame(rows)

    # Ensure output directory exists
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Append if file exists, else create with header; keep 4 decimals
    if out_path.exists():
        df.to_csv(out_path, mode="a", header=False, index=False, float_format="%.4f")
    else:
        df.to_csv(out_path, mode="w", header=True, index=False, float_format="%.4f")

    print(f"[SAVED] Results written to {out_path.resolve()}")
    return df




