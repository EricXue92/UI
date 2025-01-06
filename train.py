import os
import torch
import data_setup, engine, model_builder, utils
from torch.utils.data import DataLoader
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


os.environ["CUDA_VISIBLE_DEVICES"]= "1"
# export CUDA_VISIBLE_DEVICES=1
# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 512
LEARNING_RATE = 0.0001

train_data, test_data = data_setup.get_train_test_mnist()
shift_data = data_setup.get_shifted_mnist(rotate_degs=2, roll_pixels=10)

ood_data = data_setup.get_ood_mnist()

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True) # 938
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False) # 157

shift_loader = DataLoader(shift_data, batch_size=1024, shuffle=True)
OOD_loader = DataLoader(ood_data, batch_size=1204, shuffle=False)

#
# model = model_builder.Net().to(device)
#
# # model = model_builder.build_model(num_classes=10).to(device)
#
loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#
# results = engine.train(model=model, train_dataloader=train_loader, test_dataloader=test_loader, loss_fn=loss_fn,
#              optimizer=optimizer, epochs=NUM_EPOCHS, device=device)
# print("reach here")
#
# utils.save_model(model=model, target_dir="models",  model_name="normal_model.pth")












