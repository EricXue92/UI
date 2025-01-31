import torch.nn as nn
import torch.nn.functional as F
import torch

from sngp_wrapper.covert_utils import replace_layer_with_gaussian, convert_to_sn_my

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepResNet(nn.Module):
    def __init__(self, input_dim, num_layers=3, num_hidden=128,
                 activation="relu", num_classes=1, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self.dropout_rate = dropout_rate
        self.input_layer = nn.Linear(input_dim, num_hidden)
        ### Input layer (non-trainable)
        self.input_layer.weight.requires_grad = False
        self.input_layer.bias.requires_grad = False
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

def Build_DeepResNet(input_dim):
    model = DeepResNet(input_dim=input_dim)
    model = model.to(device)
    return model

def Build_SNGP_DeepResNet(input_dim):
    model = Build_DeepResNet(input_dim=input_dim)
    GP_KWARGS = {
        'num_inducing': 512,
        'gp_scale': 10.0, # 10
        'gp_bias': 0.,
        'gp_kernel_type': 'gaussian', # linear
        'gp_input_normalization': False,  #####
        'gp_cov_discount_factor': -1,
        'gp_cov_ridge_penalty': 1.,
        'gp_output_bias_trainable': False,
        'gp_scale_random_features': False,
        'gp_use_custom_random_features': True,
        'gp_random_feature_type': 'orf',
        'gp_output_imagenet_initializer': True,
        'num_classes': 1}

    spec_norm_replace_list = ["Linear", "Conv2D"]
    coeff = 3.
    model = convert_to_sn_my(model, spec_norm_replace_list, coeff)
    replace_layer_with_gaussian(container=model, signature="classifier", **GP_KWARGS)
    return model