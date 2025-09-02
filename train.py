import argparse
import time
import torch
import data_setup, engine, model_builder, utils
from torch.utils.data import DataLoader
from pathlib import Path
import os

from added_metrics import negative_log_likelihood, brier_score, expected_calibration_error

device = "cuda" if torch.cuda.is_available() else "cpu"
# utils.set_seed(42)

BATCH_SIZE = 512
LR = 0.003
EPOCHS = 10
WEIGHT_DECAY = 1e-4
NUM_MODELS = 5
ROTATE_DEGS = 2
ROLL_PIXELS = 4
COEFF = 1
NUM_CLASSES = 10

train_data, test_data = data_setup.get_train_test_mnist()

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) # 938
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True) # 157

# Shifted MNIST dataset for distribution evaluation
shift_data = data_setup.get_shifted_mnist(rotate_degs=ROTATE_DEGS, roll_pixels=ROLL_PIXELS)
shift_dataloader = DataLoader(shift_data , batch_size=1024, shuffle=False, drop_last=True)

loss_fn = torch.nn.CrossEntropyLoss()

def training_normal_model():
    model = model_builder.MNISTClassifier(NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # results :
    # {'train_loss': 0.0012, 'train_acc': 0.9991, 'test_loss': 0.0015,
    # 'test_acc': 0.9987, 'time': 12.34}
    results = engine.train(model, train_loader, test_loader, optimizer, loss_fn, EPOCHS, device)
    utils.save_model(model, "models",  "normal_model.pth")
    return results

def training_sngp_model():
    # ensemble of 5 SNGP models
    for i in range(5):
        utils.set_seed(i)
        model = model_builder.Build_SNGP_MNISTClassifier(NUM_CLASSES, COEFF).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        results = engine.train(model, train_loader, test_loader, optimizer, loss_fn,  EPOCHS, device)
        utils.save_model(model, "models",  f"sngp_model_{i}.pth", overwrite=True)
    return results

def train_ensemble(models, train_loader, learning_rate, num_epochs, device, weight_decay=WEIGHT_DECAY):
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    for i, model in enumerate(models):
        model.to(device)
        model_file = f"models/ensemble_model_{i}.pth"
        if os.path.exists(model_file):
            print(f"Model {i + 1} already exists. Loading the saved model.")
            model.load_state_dict(torch.load(model_file))
        else:
            print(f"Training ensemble model {i + 1} ")
            optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
            model = engine.train_model(model, train_loader, loss_fn, optimizer, epochs=num_epochs, device=device)
            print(f"Ensemble model {i + 1} trained.")
            torch.save(model.state_dict(), model_file)
            print(f"Model {i + 1} saved to {model_file}.")
        models[i] = model
    return models

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

    results = {"acc": round(acc, 4), "uncertainty": uncertainty}

    if return_hidden:
        results["hiddens"] = torch.cat(hiddens, dim=0).mean(dim=0)
    return results

def get_deep_ensemble_results(num_classes=NUM_CLASSES, dataset=test_loader, num_models=NUM_MODELS,
                              learning_rate=LR, num_epochs=EPOCHS, return_hidden=False):
    models = [model_builder.Build_MNISTClassifier(num_classes) for _ in range(num_models)]
    torch.cuda.synchronize()
    start_time = time.time()
    models = train_ensemble(models, train_loader, learning_rate, num_epochs, device)
    results = evaluate_ensemble(models, dataset, device, return_hidden)
    torch.cuda.synchronize()
    end_time = time.time()
    results["time"] = round(end_time - start_time, 4)
    return results

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nn", action="store_true", help="Use normal NN or not")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes.")
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble or not")
    parser.add_argument("--sngp", action="store_false", help="Use SNGP or not")
    parser.add_argument("--return_hidden", action="store_true", help="Return hidden or not")
    args = parser.parse_args()
    if sum([args.sngp, args.nn, args.ensemble]) != 1:
        parser.error("Exactly one of --nn, --sngp or --ensemble must be set.")
    return args

def main():
    args = parse_arguments()
    if args.nn:
        results = training_normal_model()
        output_file = "results/nn.csv"

    elif args.sngp:
        results = training_sngp_model()
        output_file = "results/sngp.csv"

    elif args.ensemble:
        res = get_deep_ensemble_results(NUM_CLASSES, test_loader, NUM_MODELS, LR, EPOCHS, args.return_hidden)
        results = {"acc": res["acc"], "time": res["time"]}
        output_file = "results/deepensemble.csv"
    else:
        raise ValueError("Invalid argument combination.")

    utils.save_results_to_csv(results, Path(output_file))
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()