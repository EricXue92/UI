import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from tableshift import get_dataset
dataset_name = "diabetes_readmission"
dset = get_dataset(dataset_name)
print(dset)

def create_dataloaders(train_filepath, shift_filepath, ood_filepath):

    batch_size, seed, frac = 128, 12, 0.1
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load and sample the training data
    X_train = pd.read_csv(train_filepath)
    y_train = pd.read_csv(train_filepath)

    # Train-validation split
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)

    # Load and sample the shifted test data
    X_shift = pd.read_csv('Diabetes-Data-Shift/X_ood_test.csv').sample(frac=frac, random_state=seed)
    y_shift = pd.read_csv('Diabetes-Data-Shift/y_ood_test.csv').sample(frac=frac, random_state=seed)

    # Add Gaussian noise to the shifted test data
    shift_noises = np.random.normal(loc=0.0, scale=0.3, size=X_shift.shape)
    X_shift = shift_noises + X_shift.to_numpy()

    # Load out-of-distribution (OOD) data
    ood = pd.read_csv("heart_attack.csv")

    # Add Gaussian noise to the OOD data
    ood_noises = np.random.normal(loc=0.0, scale=0.7, size=ood.shape)
    OOD = ood_noises + ood.to_numpy()

    # Scale all datasets using MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler on the training data and transform all datasets
    X_train = scaler.fit_transform(X_train.values)
    X_val = scaler.transform(X_val.values)
    X_test = scaler.transform(X_test.values)
    X_shift = scaler.transform(X_shift)
    OOD = MinMaxScaler().fit_transform(OOD)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    X_shift_tensor = torch.tensor(X_shift, dtype=torch.float32)
    y_shift_tensor = torch.tensor(y_shift.values, dtype=torch.long)
    OOD_tensor = torch.tensor(OOD, dtype=torch.float32)

    # Create PyTorch datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    shift_dataset = TensorDataset(X_shift_tensor, y_shift_tensor)
    ood_dataset = TensorDataset(OOD_tensor, torch.zeros(OOD_tensor.shape[0], dtype=torch.long))  # Dummy labels for OOD

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    shift_loader = DataLoader(shift_dataset, batch_size=batch_size, shuffle=False)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False)

    # Return DataLoaders in a dictionary
    return {
        "input_dim": X_train.shape[1],
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "shift": shift_loader,
        "ood": ood_loader,
    }
