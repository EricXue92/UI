from __future__ import print_function
import argparse
import pyvarinf
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from torchvision import datasets, transforms
from torch.autograd import Variable
from utils import set_seed


input_dim, train_loader, val_loader, test_loader, shift_loader, ood_loader = (res["input_dim"], res["train"], res["val"],
                                                                              res["test"], res["shift"], res["ood"])
loss_fn = nn.BCEWithLogitsLoss()


device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

ROTATE_DEGS, ROLL_PIXELS = 2, 4



# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='WD',
                    help='weight_decay (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--prior', type=str, default='gaussian', metavar='P',
                    help='prior used (default: gaussian)',
                    choices=['gaussian', 'mixtgauss', 'conjugate', 'conjugate_known_mean'])

args = parser.parse_args()

# setting up prior parameters
prior_parameters = {}
if args.prior != 'gaussian':
    prior_parameters['n_mc_samples'] = 1
if args.prior == 'mixtgauss':
    prior_parameters['sigma_1'] = 0.02
    prior_parameters['sigma_2'] = 0.2
    prior_parameters['pi'] = 0.5
if args.prior == 'conjugate':
    prior_parameters['mu_0'] = 0.
    prior_parameters['kappa_0'] = 3.
    prior_parameters['alpha_0'] = .5
    prior_parameters['beta_0'] = .5
if args.prior == 'conjugate_known_mean':
    prior_parameters['alpha_0'] = .5
    prior_parameters['beta_0'] = .5
    prior_parameters['mean'] = 0.
torch.manual_seed(args.seed)

kwargs = {'num_workers': 4, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

model = Build_MNISTClassifier(10)
var_model = pyvarinf.Variationalize(model)
var_model.set_prior(args.prior, **prior_parameters)
var_model = var_model.to(device)

# optimizer = optim.Adam(var_model.parameters(), lr=args.lr)

optimizer = torch.optim.AdamW(var_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train(epoch):
    var_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = var_model(data)
        loss_error = F.nll_loss(output, target)
        loss_prior = var_model.prior_loss() / 59904  #60000 drop_out=True
        loss = loss_error + loss_prior

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]'
                f'\tLoss: {loss.item():.6f}\tLoss error: {loss_error.item():.6f}\tLoss weights: {loss_prior.item():.6f}')

def test(epoch):
    var_model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            output = var_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total_samples += data.size(0)
    test_loss /= total_samples
    accuracy = 100. * correct / total_samples
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total_samples} ({accuracy:.2f}%)\n')

def save_model(model, target_dir="model", model_name="vi_model.pth"):
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    model_save_path = target_dir_path / model_name
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

def main():
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
    save_model(var_model)

if __name__ == '__main__':
    main()

