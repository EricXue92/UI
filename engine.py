import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import time
import torch.nn as nn

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0
    if not isinstance(model.classifier, nn.Linear):
        model.classifier.reset_covariance_matrix()
        kwargs = {'return_random_features': False, 'return_covariance': False,
                  'update_precision_matrix': True, 'update_covariance_matrix': False}
    else:
        kwargs = {}
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        y_pred = model(X, **kwargs)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc * 100:.2f}%")
    return train_loss, train_acc

def test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss, test_acc = 0, 0
    hiddens = []
    if not isinstance(model.classifier, nn.Linear):
        model.classifier.update_covariance_matrix()
        eval_kwargs = {'return_random_features': False, 'return_covariance': False,
                       'update_precision_matrix': False, 'update_covariance_matrix': False}
    else:
        eval_kwargs = {}
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits, hidden = model(X, return_hidden=True, **eval_kwargs)
            hiddens.append(hidden)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
    hiddens = torch.cat(hiddens, dim=0)
    print(f"Hidden shape: {hiddens.shape}")
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    print(f"Test Loss: {test_loss:.4f} | Test accuracy: {test_acc * 100:.2f}%\n")
    return test_loss, test_acc, hiddens

def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device):
    start_time = time.time()
    results = {"train_loss": [], "train_acc": [], "test_loss": [],  "test_acc": [], "time":[]}
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device)
        test_loss, test_acc, hidden = test_step(model=model,  dataloader=test_dataloader, loss_fn=loss_fn,  device=device)
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    end_time = time.time()
    train_time = end_time - start_time
    results["time"].append(train_time)
    print(f"Training time: {train_time:.2f} seconds")
    result = {key: round(value[-1], 4) for key, value in results.items()}
    # get the last hidden state during the testing
    # result["hidden"] = hidden.cpu().numpy()
    print(f"last element: {result}")
    return result

# For ensemble training
def train_model(model, train_loader, loss_fn, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()







