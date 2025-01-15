import os

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
pd.set_option('display.max_columns', None)

def create_dataloaders():
    batch_size, Frac = 512, 0.1
    seed = 23
    # Load and sample the training data
    X_train = pd.read_csv('Diabetes-Data-Shift/X_train.csv').sample(frac=Frac, random_state=seed)
    y_train = pd.read_csv('Diabetes-Data-Shift/y_train.csv').sample(frac=Frac, random_state=seed)

    # Train-validation split
    # train: (87230, 142)
    # val: (9693, 142)
    # train: (872306, 142)  val: (96923, 142)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)
    print(f"train: {X_train.shape}  val: {X_val.shape}")

    # Load and sample the shifted test data
    X_test = pd.read_csv('Diabetes-Data-Shift/X_id_test.csv').sample(frac=Frac, random_state=seed)
    y_test = pd.read_csv('Diabetes-Data-Shift/y_id_test.csv').sample(frac=Frac, random_state=seed)

    X_shift = pd.read_csv('Diabetes-Data-Shift/X_ood_test.csv').sample(frac=Frac, random_state=seed)
    y_shift = pd.read_csv('Diabetes-Data-Shift/y_ood_test.csv').sample(frac=Frac, random_state=seed)


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
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    X_shift_tensor = torch.tensor(X_shift, dtype=torch.float32)
    y_shift_tensor = torch.tensor(y_shift.values, dtype=torch.float32)

    OOD_tensor = torch.tensor(OOD, dtype=torch.float32)

    # Create PyTorch datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    shift_dataset = TensorDataset(X_shift_tensor, y_shift_tensor)
    ood_dataset = TensorDataset(OOD_tensor, torch.zeros(OOD_tensor.shape[0], dtype=torch.float32))  # Dummy labels for OOD

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
        "ood": ood_loader }
