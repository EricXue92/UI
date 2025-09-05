import torch
import torch.nn.functional as F

def negative_log_likelihood(logits, labels):
    loss = F.cross_entropy(logits, labels, reduction='mean')
    return loss.item()


# def negative_log_likelihood(probs, labels):
#
#     if probs.size(1) == 1:
#         probs_binary = torch.cat([1 - probs, probs], dim=1)
#         log_probs = torch.log(probs_binary + 1e-15)
#         nll = F.poisson_nll_loss(log_probs, labels, reduction='mean')
#     else:
#         log_probs = torch.log(probs + 1e-15)
#         nll = F.nll_loss(log_probs, labels, reduction='mean')
#     return nll.item()


def brier_score(probs, labels):
    if probs.ndim == 1:                             # (N,)
        return torch.mean((probs - labels.float())**2).item()
    if probs.size(1) == 1:                          # (N,1)
        return torch.mean((probs.squeeze(1) - labels.float())**2).item()
    # multi-class
    one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
    return torch.mean(torch.sum((probs - one_hot)**2, dim=1)).item()


def expected_calibration_error(probs, labels, n_bins=15) -> float:
    if probs.ndim == 1:
        conf = torch.max(probs, 1-probs)
        pred = (probs>=0.5).long()
    elif probs.size(1)==1:
        probs_flat = probs.view(-1)
        conf = torch.max(probs_flat, 1-probs_flat)
        pred = (probs_flat>=0.5).long()
    else:
        conf, pred = probs.max(dim=1)

    acc   = pred.eq(labels).float()
    ece   = torch.tensor(0., device=probs.device)

    edges = torch.linspace(0., 1., n_bins+1, device=probs.device)
    # Calculate ECE across all bins
    for i in range(n_bins):
        # First bin includes 0.0, others exclude lower boundary
        if i ==0:
            mask = (conf >= edges[i]) & (conf <= edges[i+1])
        else:
            mask = (conf > edges[i]) & (conf <= edges[i+1])
        if mask.any():
            bin_weight = mask.float().mean()
            bin_acc = acc[mask].mean()
            bin_conf = conf[mask].mean()
            ece += bin_weight * torch.abs(bin_acc - bin_conf)
    return ece.item()


