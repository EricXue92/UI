import os, argparse
import torch
import data_setup, engine, model_builder, utils, evaluation
from torch.utils.data import DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"
# os.environ["CUDA_VISIBLE_DEVICES"]= "1"
# export CUDA_VISIBLE_DEVICES=1
utils.set_seed(42)

NUM_EPOCHS = 10
NUM_MODELS = 5
BATCH_SIZE = 512
LEARNING_RATE = 0.0001

train_data, test_data = data_setup.get_train_test_mnist()
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) # 938
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True) # 157
loss_fn = torch.nn.CrossEntropyLoss()

def training_sngp_model(num_classes, coeff, learning_rate, num_epochs):
    model = model_builder.Build_SNGP_MNISTClassifier(num_classes, coeff).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=learning_rate)
    results = engine.train(model=model, train_dataloader=train_loader, test_dataloader=test_loader, loss_fn=loss_fn,
                 optimizer=optimizer, epochs=num_epochs, device=device)
    utils.save_model(model=model, target_dir="models",  model_name="sngp_model.pth")

def training_normal_model(num_classes,learning_rate, num_epochs):
    model = model_builder.MNISTClassifier(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=learning_rate)
    results = engine.train(model=model, train_dataloader=train_loader, test_dataloader=test_loader, loss_fn=loss_fn,
                 optimizer=optimizer, epochs=num_epochs, device=device)
    utils.save_model(model=model, target_dir="models",  model_name="normal_model.pth")

def train_ensemble(models, train_loader, device):
    for i, model in enumerate(models):
        model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        engine.train_model(model, train_loader, loss_fn, optimizer, epochs=NUM_EPOCHS, device=device)
        print(f"Model {i} trained successfully")

def evaluate_ensemble(models, dataloader, device, return_hidden=False):
    predictions, hiddens, all_labels = [], [], []
    with torch.no_grad():
        for model in models:
            model.eval()
            model_preds, model_hiddens = [], []
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                all_labels.append(y)
                if return_hidden:
                    logits, hidden = model(X, return_hidden=True)
                    model_hiddens.append(hidden)
                else:
                    logits = model(X)
                model_preds.append(logits)
            predictions.append(torch.cat(model_preds, dim=0).unsqueeze(0))
            if return_hidden:
                hiddens.append(torch.cat(model_hiddens, dim=0).unsqueeze(0))
    predictions = torch.cat(predictions, dim=0)
    mean_prediction = predictions.mean(dim=0)
    mean_std = predictions.std(dim=0)
    uncertainty = mean_std.mean(dim=1)
    pred_y = mean_prediction.argmax(dim=1)
    y_true = torch.cat(all_labels, dim=0)
    y_true = y_true[:len(pred_y)]
    acc = (pred_y == y_true).float().mean().item()
    if return_hidden:
        hiddens = torch.cat(hiddens, dim=0).mean(dim=0)
        return acc, uncertainty, hiddens
    else:
        return acc, uncertainty

def get_deep_ensemble_results(num_classes=10, return_hidden=False):
    models = [model_builder.Build_MNISTClassifier(num_classes=num_classes) for _ in range(NUM_MODELS)]
    train_ensemble(models, train_loader, device)
    results = evaluate_ensemble(models, test_loader, device, return_hidden=return_hidden)
    if return_hidden:
        acc, uncertainty, hiddens = results
        print(f"Accuracy: {acc:.4f} | Uncertainty: {uncertainty.mean().item():.4f} | Hiddens Shape: {hiddens.shape}")
    else:
        acc, uncertainty = results
        print(f"Accuracy: {acc:.4f} | Uncertainty: {uncertainty.mean().item():.4f}")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes.")
    parser.add_argument("--num_models", type=int, default=5, help="Number of models in the ensemble.")

    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")

    parser.add_argument("--nn", action="store_false", help="Use normal NN or not")
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble or not")
    parser.add_argument("--sngp", action="store_true", help="Use SNGP or not")

    parser.add_argument("--coeff", type=float, default=3., help="Spectral normalization coefficient") # 3
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer.")

    args = parser.parse_args()
    if sum([args.sngp, args.nn, args.ensemble]) != 1:
        parser.error("Exactly one of --nn, --sngp or --ensemble must be set.")
    return args

def main(args):
    if args.sngp:
        training_sngp_model(args.num_classes, args.coeff, args.learning_rate, args.num_epoch)
    elif args.normal:
        training_normal_model(args.num_classes, args.learning_rate, args.num_epochs)
    elif args.deep_ensemble:
        get_deep_ensemble_results()

    # get_deep_ensemble_results()
    # training_sngp_model()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)


