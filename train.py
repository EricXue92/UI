import os, argparse
import time
import pandas as pd
import torch
import data_setup, engine, model_builder, utils, evaluation
from torch.utils.data import DataLoader
from collections import defaultdict
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
# os.environ["CUDA_VISIBLE_DEVICES"]= "1"
# export CUDA_VISIBLE_DEVICES=1
utils.set_seed(42)

BATCH_SIZE = 512
train_data, test_data = data_setup.get_train_test_mnist()
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) # 938
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True) # 157

shift_data = data_setup.get_shifted_mnist(rotate_degs=2, roll_pixels=2)
shift_dataloader = DataLoader(shift_data , batch_size=1024, shuffle=False, drop_last=True)

loss_fn = torch.nn.CrossEntropyLoss()

def training_sngp_model(num_classes, coeff, learning_rate, num_epochs, weight_decay,):
    model = model_builder.Build_SNGP_MNISTClassifier(num_classes, coeff).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
    results = engine.train(model=model, train_dataloader=train_loader, test_dataloader=test_loader, loss_fn=loss_fn,
                 optimizer=optimizer, epochs=num_epochs, device=device)
    utils.save_model(model=model, target_dir="models",  model_name="sngp_model.pth")
    return results

def training_normal_model(num_classes, learning_rate, num_epochs, weight_decay):
    model = model_builder.MNISTClassifier(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
    results = engine.train(model=model, train_dataloader=train_loader, test_dataloader=test_loader, loss_fn=loss_fn,
                 optimizer=optimizer, epochs=num_epochs, device=device)
    utils.save_model(model=model, target_dir="models",  model_name="normal_model.pth")
    return results

def train_ensemble(models, train_loader, learning_rate, num_epochs, device):
    for i, model in enumerate(models):
        model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        engine.train_model(model, train_loader, loss_fn, optimizer, epochs=num_epochs, device=device)
        print(f"Model {i} trained successfully")

def evaluate_ensemble(models, dataloader, device, return_hidden):
    results = {}
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
    results["acc"] = acc
    results["uncertainty"] = uncertainty
    if return_hidden:
        hiddens = torch.cat(hiddens, dim=0).mean(dim=0)
        results["hiddens"] = hiddens
    else:
        results["hiddens"] = None
    return results

def get_deep_ensemble_results(num_classes=10, dataset=test_loader, num_models=5, learning_rate=0.001, num_epochs=10, return_hidden=True):
    models = [model_builder.Build_MNISTClassifier(num_classes) for _ in range(num_models)]
    start_time = time.time()
    train_ensemble(models, train_loader, learning_rate, num_epochs, device)
    end_time = time.time()
    train_time = end_time - start_time
    results = evaluate_ensemble(models, dataset, device, return_hidden)
    results["time"] = train_time
    print(results)
    return results

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes.")
    parser.add_argument("--num_models", type=int, default=5, help="Number of models in the ensemble.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--nn", action="store_true", help="Use normal NN or not")
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble or not")
    parser.add_argument("--sngp", action="store_false", help="Use SNGP or not")
    parser.add_argument("--return_hidden", action="store_false", help="Return hidden or not")
    parser.add_argument("--coeff", type=float, default=0.95, help="Spectral normalization coefficient") # 3
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer.")
    args = parser.parse_args()
    if sum([args.sngp, args.nn, args.ensemble]) != 1:
        parser.error("Exactly one of --nn, --sngp or --ensemble must be set.")
    return args

def main(args):
    result_dict = defaultdict(list)
    if args.nn:
        results = training_normal_model(args.num_classes, args.learning_rate, args.num_epochs, args.weight_decay)
        result_file_path = Path("results/nn.csv")
    elif args.sngp:
        results = training_sngp_model(args.num_classes, args.coeff, args.learning_rate, args.num_epochs, args.weight_decay)
        result_file_path = Path("results/sngp.csv")
    elif args.ensemble:
        result_file_path = Path("results/deepensemble.csv")
        if args.return_hidden:
            results = get_deep_ensemble_results(args.num_classes, test_loader, args.num_models, args.learning_rate, args.num_epochs, args.return_hidden)
        else:
            results = get_deep_ensemble_results(args.num_classes, test_loader, args.num_models, args.learning_rate, args.num_epochs, args.return_hidden)

    utils.save_results_to_csv(results, result_file_path)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)