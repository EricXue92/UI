import torch
from tqdm.auto import tqdm

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        y_pred_logits = model(X)
        loss = loss_fn(y_pred_logits, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)
        train_acc += (y_pred == y).sum().item() / len(y_pred_logits)
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model, train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn, epochs, device):
    results = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[], "test_loss":[], "test_acc":[]}
    best_val_loss, epochs_no_improve = float('inf'), 0
    best_model_state, patience = None, 10
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device)
        val_loss, val_acc = test_step(model=model, dataloader=val_dataloader, loss_fn=loss_fn, device=device)
        print( f"Epoch: {epoch + 1} | " f"train_loss: {train_loss:.4f} | "  f"train_acc: {train_acc:.4f} | " 
               f"test_loss: {val_loss:.4f} | " f"test_acc: {val_acc:.4f}" )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping triggered")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)  # Restore best model
                break
    test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
    return results


