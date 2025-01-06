import torch.nn as nn
import torch.nn.functional as F
import torch

class DeepResNet(nn.Module):
    def __init__(self, input_dim, num_layers=3, num_hidden=128,
                 activation="relu", num_classes=10, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self.dropout_rate = dropout_rate

        self.input_layer = nn.Linear(input_dim, num_hidden)
        # Input layer (non-trainable)
        # self.input_layer.weight.requires_grad = False
        # self.input_layer.bias.requires_grad = False
        self.residual_layers = nn.ModuleList([self.make_dense_layer() for _ in range(num_layers)])
        if self.num_classes is not None:
            self.classifier = self.make_output_layer(num_classes)
        self.activation_function = self.get_activation_function(activation)

    def forward(self, x,  return_hidden=False, **kwargs):
        hidden = self.input_layer(x)
        hidden = self.activation_function(hidden)
        for i in range(self.num_layers):
            resid = self.activation_function(self.residual_layers[i](hidden))
            resid = F.dropout(resid, p=self.dropout_rate, training=self.training)
            hidden = hidden + resid  # Residual connection
        logits = self.classifier(hidden, **kwargs)
        if return_hidden:
            return logits, hidden
        else:
            return logits

    def make_dense_layer(self):
        return nn.Linear(self.num_hidden, self.num_hidden)
    def make_output_layer(self, num_outputs):
        return nn.Linear(self.num_hidden, num_outputs)
    def get_activation_function(self, activation):
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU()
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}")
        return activations[activation]


def mc_dropout(model, x,  n_samples=5, return_hidden=False, **kwargs):
    predictions, hidden_states = [], []
    model.train()  # Ensure dropout is active
    with torch.no_grad():
        for _ in range(n_samples):
            if return_hidden:
                output, hidden = model(x, return_hidden=True, **kwargs)
                predictions.append(output)
                hidden_states.append(hidden)
            else:
                output = model(x, **kwargs)
                predictions.append(output)
    predictions = torch.stack(predictions, dim=0)  # Shape: (n_samples, batch_size, num_classes)
    mean_prediction = predictions.mean(dim=0)  # Mean across MC samples
    std_prediction = predictions.std(dim=0)  # Standard deviation across MC samples
    if return_hidden:
        hidden_states = torch.stack(hidden_states, dim=0)  # Shape: (n_samples, batch_size, hidden_dim)
        return mean_prediction, std_prediction, hidden_states
    else:
        return mean_prediction, std_prediction

def deep_ensemble_inference(models, x, return_hidden=False, **kwargs):
    predictions, hiddens = [], []
    with torch.no_grad():
        for model in models:
            model.eval()  # Set each model to evaluation mode
            if return_hidden:
                output, hidden = model(x, return_hidden=True, **kwargs)
                hiddens.append(hidden.unsqueeze(0))
            else:
                output = model(x, return_hidden=return_hidden, **kwargs)
            predictions.append(output.unsqueeze(0))

    predictions = torch.cat(predictions, dim=0)  # Shape: (n_models, batch_size, num_classes)
    mean_prediction = predictions.mean(dim=0)  # Mean across ensemble models
    std_prediction = predictions.std(dim=0)  # Standard deviation across ensemble models

    if return_hidden:
        hiddens = torch.cat(hiddens, dim=0)  # Shape: (n_models, batch_size, hidden_dim)
        hiddens = hiddens.mean(dim=0)  # Average hidden states across ensemble models
        return mean_prediction, std_prediction, hiddens
    else:
        return mean_prediction, std_prediction


# # Example usage:
# models = [ DeepResNet(input_dim=128, num_layers=3, num_hidden=128, activation="relu", num_classes=3, dropout_rate=0.1)
#            for _ in range(5)]
# input_data = torch.randn(5, 128)  # Example input
# mean_pred, uncertainty = deep_ensemble_inference(models, input_data, kwargs={})
#
# print("Mean predictions shape:", mean_pred)
# print("Mean uncertainty:", uncertainty)
# Example usage:

models = [DeepResNet(input_dim=128, num_layers=3, num_hidden=128, activation="relu", num_classes=3, dropout_rate=0.1)
          for _ in range(5)]

input_data = torch.randn(5, 128)  # Example input
kwargs={}
mean_pred, uncertainty = deep_ensemble_inference(models, input_data, **kwargs)
print("Mean predictions shape:", mean_pred.shape)
print("Uncertainty shape:", uncertainty.shape)


# Create an instance of the DeepResNet model
input_dim = 128
num_classes = 3
model = DeepResNet(input_dim=input_dim, num_layers=3, num_hidden=128, activation="relu", num_classes=num_classes, dropout_rate=0.1)

# Generate some synthetic input data
batch_size = 5
input_data = torch.randn(batch_size, input_dim)  # Example input

# Define any additional arguments required by the classifier (if any)
kwargs = {}

# Perform MC Dropout inference
n_samples = 5
mean_pred, uncertainty = mc_dropout(model, input_data,  n_samples=n_samples, **kwargs)
# Print the results
print("Mean Predictions:\n", mean_pred)
print("Uncertainty (Standard Deviation):\n", uncertainty)