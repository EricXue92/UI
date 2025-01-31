import torch
import time
from tqdm.auto import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        preds = torch.sigmoid(y_pred) > 0.5
        total_acc += (preds.squeeze(1) == y).float().mean().item()
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    return avg_loss, avg_acc

def test_step(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, total_acc = 0, 0
    hiddens = []
    eval_kwargs = {}
    if not isinstance(model.classifier, nn.Linear):
        model.classifier.update_covariance_matrix()
        eval_kwargs = {'return_random_features': False, 'return_covariance': False,
                       'update_precision_matrix': False, 'update_covariance_matrix': False}
    with torch.no_grad():
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)
            logits, hidden = model(X, return_hidden=True, **eval_kwargs)
            hiddens.append(hidden)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            preds = torch.sigmoid(logits) > 0.5
            total_acc += (preds.squeeze(1) == y).float().mean().item()
    hiddens = torch.cat(hiddens, dim=0)
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    return avg_loss, avg_acc, hiddens

def train(model, train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn, epochs, device):

    results = {"train_loss": [], "train_acc": [], "val_loss": [],  "val_acc": [],
               "time":[], "test_loss": [], "test_acc": [] }
    # best_val_loss, epochs_no_improve = float('inf'), 0
    # best_model_state, patience = None, 200

    torch.cuda.synchronize()
    start_time = time.time()

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn,
                                           optimizer=optimizer, device=device)
        val_loss, val_acc, _ = test_step(model, val_dataloader, loss_fn, device)
        print(f"Epoch: {epoch + 1} | " f"train_loss: {train_loss:.4f} | "  f"train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f} | " f"val_acc: {val_acc:.4f} \n")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     epochs_no_improve = 0
        #     best_model_state = model.state_dict()
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve == patience:
        #         print("Early stopping triggered")
        #         if best_model_state is not None:
        #             model.load_state_dict(best_model_state)  # Restore best model
        #         break

    torch.cuda.synchronize()
    end_time = time.time()

    elapsed_time = round(end_time - start_time, 2)
    results["time"].append(elapsed_time)

    utils.plot_loss_curves(results)
    test_loss, test_acc, _ = test_step(model, test_dataloader, loss_fn, device)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%\n")
    print(f"Training completed in {elapsed_time:.2f} seconds.")
    return {key: round(value[-1], 4) for key, value in results.items()}


def train_model(model, train_loader, loss_fn, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            loss = loss_fn(model(X), y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model







