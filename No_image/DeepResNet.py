
import torch.nn as nn
import torch.nn.functional as F
import torch

class DeepResNet(nn.Module):
    def __init__(self, input_dim, num_layers=3, num_hidden=128,
                 activation="relu", num_classes=10, dropout_rate=0.1, **classifier_kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self.dropout_rate = dropout_rate
        self.classifier_kwargs = classifier_kwargs
        self.input_layer = nn.Linear(input_dim, num_hidden)
        # Input layer (non-trainable)
        # self.input_layer.weight.requires_grad = False
        # self.input_layer.bias.requires_grad = False

        self.residual_layers = nn.ModuleList([self.make_dense_layer() for _ in range(num_layers)])
        if self.num_classes is not None:
            self.classifier = self.make_output_layer(num_classes)
        self.activation_function = self.get_activation_function(activation)

    def forward(self, x, kwargs, return_hidden=False):
        hidden = self.input_layer(x)
        hidden = self.activation_function(hidden)
        for i in range(self.num_layers):
            resid = self.activation_function(self.residual_layers[i](hidden))
            resid = F.dropout(resid, p=self.dropout_rate, training=self.training)
            hidden = hidden + resid  #  # Residual connection

        out_put = self.classifier(hidden, **kwargs)
        if return_hidden:
            return out_put, hidden
        else:
            return out_put
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

# 1. Define input dimension and batch size
input_dimension = 10  # Example input dimension
batch_size = 32      # Example batch size
# 2. Generate random input data using torch
input_data = torch.randn(batch_size, input_dimension)
print("Input data shape:", input_data.shape)
# 3. Instantiate the model
model = DeepResNet(input_dim=input_dimension)
print("Model architecture:")
print(model)
# 4. Pass the input data through the model
output = model(input_data, kwargs={} )
print("Output shape:", output.shape)
print("Sample output values (first 5):\n", output[:2])
# 5. Test with different configurations (optional)
# Example with multiple classes
model_multi_class = DeepResNet(input_dim=input_dimension, num_classes=5)
output_multi_class = model_multi_class(input_data,  kwargs={} )
print("\nOutput shape (multi-class):", output_multi_class.shape)
print("Sample output values (multi-class, first 5):\n", output_multi_class[:5])
# Example with return_hidden=True
output_with_hidden, hidden_state = model(input_data, kwargs={}, return_hidden=True)
print("\nOutput shape (with hidden):", output_with_hidden.shape)
print("Hidden state shape:", hidden_state.shape)

# Example with a different activation function
model_tanh = DeepResNet(input_dim=input_dimension, activation="tanh")
output_tanh = model_tanh(input_data,kwargs={})
print("\nOutput shape (tanh activation):", output_tanh.shape)

print("\nData generation and model forward pass successful!")