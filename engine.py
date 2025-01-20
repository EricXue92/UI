import torch
import time
from tqdm.auto import tqdm
import torch.nn as nn

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss, total_acc = 0, 0
    kwargs = {}

    if not isinstance(model.classifier, nn.Linear):
        model.classifier.reset_covariance_matrix()
        kwargs = {'return_random_features': False, 'return_covariance': False,
                  'update_precision_matrix': True, 'update_covariance_matrix': False}

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X, **kwargs)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += (y_pred.argmax(dim=1) == y).float().mean().item()
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    print(f"Train Loss: {avg_loss:.4f} | Train Accuracy: {avg_acc * 100:.2f}%")
    return avg_loss, avg_acc

def test_step(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, total_acc = 0, 0
    eval_kwargs = {}

    if not isinstance(model.classifier, nn.Linear):
        model.classifier.update_covariance_matrix()
        eval_kwargs = {'return_random_features': False, 'return_covariance': False,
                       'update_precision_matrix': False, 'update_covariance_matrix': False}

    with torch.no_grad():
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X, return_hidden=False, **eval_kwargs)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            total_acc += (logits.argmax(dim=1) == y).float().mean().item()

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    print(f"Test Loss: {avg_loss:.4f} | Test Accuracy: {avg_acc * 100:.2f}%\n")
    return avg_loss, avg_acc

def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device):
    torch.cuda.synchronize()
    start_time = time.time()
    results = {"train_loss": [], "train_acc": [], "test_loss": [],  "test_acc": [], "time":[]}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    results["time"].append(elapsed_time)
    print(f"Training completed in {elapsed_time:.2f} seconds.")
    return {key: round(value[-1], 4) for key, value in results.items()}

def train_model(model, train_loader, loss_fn, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            loss = loss_fn(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model








