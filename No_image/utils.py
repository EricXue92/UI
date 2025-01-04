import random
import os
import torch
import numpy as np

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
    target_dir_path = os.path.join(target_dir)
    os.makedirs(target_dir_path, exist_ok=True)
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pth' or '.pt'"
    model_filepath = os.path.join(target_dir_path, model_name)
    torch.save(model.state_dict(), model_filepath)
    print(f"[INFO] Saved model to: {model_filepath}")
