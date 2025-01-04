
import data_setup, engine, utils, DeepResNet
import torch
import torch.nn as nn
import torch.optim as optim

NUM_EPOCHS = 200
BATCH_SIZE = 512
HIDDEN_UNITS = 128
LEARNING_RATE = 1e-4
NUM_LAYERS = 3
NUM_CLASSES=1
DROPOUT_RATE = 0.1

# # Setup directories
# train_dir = "data/pizza_steak_sushi/train"
# test_dir = "data/pizza_steak_sushi/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
utils.set_seed(42)

dataloaders_dict = data_setup.create_dataloaders()
input_dim, train_dataloader, val_dataloader =dataloaders_dict["input_dim"], dataloaders_dict["train"], dataloaders_dict["val"]
test_dataloader, shift_dataloader, ood_dataloader = dataloaders_dict["test"], dataloaders_dict["shift"], dataloaders_dict["ood"]


model = DeepResNet.DeepResNet(
    input_dim=input_dim,
    num_layers=NUM_LAYERS,
    num_hidden=HIDDEN_UNITS,
    activation="relu",
    num_classes=NUM_CLASSES,
    dropout_rate=DROPOUT_RATE).to(device)

# Set loss and optimizer
loss_fn = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             val_dataloader=val_dataloader,
             test_dataloader=test_dataloader,
             optimizer=optimizer,
             loss_fn=loss_fn,
             epochs=NUM_EPOCHS,
             device=device)

utils.save_model(model=model, target_dir="models", model_name="DeepResNet.pth")







