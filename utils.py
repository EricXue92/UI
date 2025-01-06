import torch
from pathlib import Path

def save_model(model, target_dir, model_name):
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)

  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(), f=model_save_path)

def mc_dropout(model, x, n_samples=5, return_hidden=False):
  predictions, hiddens = [], []
  model.train()  # Ensure dropout is active
  with torch.no_grad():
    for _ in range(n_samples):
      if return_hidden:
        logits, hidden = model(x, return_hidden=True)
        hiddens.append(hidden)
      else:
        logits = model(x)
      predictions.append(logits)
  predictions = torch.stack(predictions, dim=0)
  mean_prediction = predictions.mean(dim=0)
  mean_std = predictions.std(dim=0)
  if return_hidden:
    hiddens = torch.stack(hiddens, dim=0)
    hiddens = hiddens.mean(dim=0)
    return mean_prediction, mean_std, hiddens
  else:
    return mean_prediction, mean_std


def deep_ensemble(models, x, return_hidden=False):
  predictions, hiddens = [], []
  with torch.no_grad():
    for model in models:
      model.eval()  # Set each model to evaluation mode
      if return_hidden:
        logits, hidden = model(x, return_hidden=True)
        hiddens.append(hidden.unsqueeze(0))
      else:
        logits = model(x)
      predictions.append(logits.unsqueeze(0))

  predictions = torch.cat(predictions, dim=0)
  mean_prediction = predictions.mean(dim=0)
  mean_std = predictions.std(dim=0)

  if return_hidden:
    hiddens = torch.cat(hiddens, dim=0)
    hiddens = hiddens.mean(dim=0)
    return mean_prediction, mean_std, hiddens
  else:
    return mean_prediction, mean_std





