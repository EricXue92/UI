import argparse
import os.path
import time
import torch
import data_setup, engine, model_builder, utils
import torch.nn as nn
from pathlib import Path
import pyvarinf

device = "cuda" if torch.cuda.is_available() else "cpu"
utils.set_seed(23)
BATCH_SIZE = 512
LR = 1e-4
EPOCHS = 2000
WEIGHT_DECAY = 1e-4
NUM_MODELS = 5

res = data_setup.create_dataloaders()
input_dim, train_loader, val_loader, test_loader, shift_loader, ood_loader = (res["input_dim"], res["train"], res["val"],
                                                                              res["test"], res["shift"], res["ood"])
loss_fn = nn.BCEWithLogitsLoss()


def train_vi_model():
    model = model_builder.Build_DeepResNet(input_dim=input_dim)
    model = pyvarinf.Variationalize(model)
    model.set_prior('gaussian', **{'n_mc_samples':1})
    model = model.to(device)
    # print(hasattr(model, '_variationalize_module'))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) #  weight_decay=WEIGHT_DECAY
    results = engine.train(model, train_loader, val_loader, test_loader, optimizer, loss_fn, EPOCHS, device)
    utils.save_model(model, "models",  "vi_model.pth")
    return results

def training_normal_model():
    model = model_builder.Build_DeepResNet(input_dim=input_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) #  weight_decay=WEIGHT_DECAY
    results = engine.train(model, train_loader, val_loader, test_loader, optimizer, loss_fn, EPOCHS, device)
    utils.save_model(model, "models",  "normal_model.pth")
    return results

def training_sngp_model():
    model = model_builder.Build_SNGP_DeepResNet(input_dim=input_dim).to(device)

    print(dir(model))
    exit()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) #   weight_decay=WEIGHT_DECAY
    results = engine.train(model, train_loader, val_loader, test_loader, optimizer, loss_fn,  EPOCHS, device)
    utils.save_model(model, "models",  "sngp_model.pth")
    return results

def train_ensemble(models, train_loader, learning_rate, num_epochs, device, weight_decay=WEIGHT_DECAY):
    torch.cuda.synchronize()
    start_time = time.time()
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
    torch.cuda.synchronize()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Ensemble training completed in {total_time:.2f} seconds.")
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

def get_deep_ensemble_results(dataset=test_loader, num_models=NUM_MODELS,
                              learning_rate=LR, num_epochs=EPOCHS, return_hidden=False):

    models = [model_builder.Build_DeepResNet(input_dim=input_dim) for _ in range(num_models)]
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
    parser.add_argument("--vi", action="store_false", help="Use variational model or not")
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble or not")
    parser.add_argument("--sngp", action="store_true", help="Use SNGP or not")
    parser.add_argument("--return_hidden", action="store_true", help="Return hidden or not")
    args = parser.parse_args()
    if sum([args.sngp, args.nn, args.ensemble, args.vi]) != 1:
        parser.error("Exactly one of --nn, --sngp, --ensemble or --vi must be set.")
    return args

def main():
    args = parse_arguments()
    if args.nn:
        results = training_normal_model()
        output_file = "results/nn.csv"
    elif args.sngp:
        results = training_sngp_model()
        output_file = "results/sngp.csv"
    elif args.vi:
        results = train_vi_model()
        output_file = "results/vi.csv"
    elif args.ensemble:
        res = get_deep_ensemble_results(test_loader, NUM_MODELS, LR, EPOCHS, args.return_hidden)
        results = {"acc": res["acc"] }
        output_file = "results/deepensemble.csv"
    else:
        raise ValueError("Invalid argument combination.")
    utils.save_results_to_csv(results, Path(output_file))
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()