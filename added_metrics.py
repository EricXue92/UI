import torch
import torch.nn.functional as F

def negative_log_likelihood(logits: torch.Tensor,
                            labels: torch.Tensor) -> float:
    """
    NLL (mean).  Handles binary and multi-class logits.
    """
    # Binary if shape is (N,) or (N,1)
    if logits.ndim == 1 or logits.size(1) == 1:
        loss = F.binary_cross_entropy_with_logits(
            logits.view(-1), labels.float(), reduction='mean')
    else:                                           # multi-class
        loss = F.cross_entropy(logits, labels, reduction='mean')
    return loss.item()

def brier_score(probs: torch.Tensor,
                labels: torch.Tensor) -> float:
    """
    Brier score (mean squared error between predicted prob and truth).
    """
    if probs.ndim == 1:                             # (N,)
        return torch.mean((probs - labels.float())**2).item()
    if probs.size(1) == 1:                          # (N,1)
        return torch.mean((probs.squeeze(1) - labels.float())**2).item()
    # multi-class
    one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
    return torch.mean(torch.sum((probs - one_hot)**2, dim=1)).item()

def expected_calibration_error(probs: torch.Tensor,
                               labels: torch.Tensor,
                               n_bins: int = 15) -> float:
    """
    Equal-width ECE (Küller & Dosti, 2015 style).
    """
    if probs.ndim == 1 or probs.size(1) == 1:       # binary
        conf = probs.view(-1)
        pred = (conf >= 0.5).long()
    else:                                           # multi-class
        conf, pred = probs.max(dim=1)

    acc   = pred.eq(labels).float()
    ece   = torch.zeros([], device=probs.device)
    edges = torch.linspace(0., 1., n_bins+1, device=probs.device)

    for i in range(n_bins):
        mask = (conf > edges[i]) & (conf <= edges[i+1])
        if mask.any():
            ece += mask.float().mean() * torch.abs(acc[mask].mean() - conf[mask].mean())
    return ece.item()

