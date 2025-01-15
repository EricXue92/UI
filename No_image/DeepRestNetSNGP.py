import torch.nn as nn
import torch.nn.functional as F
from layers import spectral_norm_fc
import torch
import numpy as np
from DeepResNet import DeepResNet
from sngp_wrapper.covert_utils import convert_to_sn_my, replace_layer_with_gaussian

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GP_KWARGS = {
    'num_inducing': 512,
    'gp_scale': 1.0,
    'gp_bias': 0.,
    'gp_kernel_type': 'gaussian', # 'linear'
    'gp_input_normalization': True,
    'gp_cov_discount_factor': -1,
    'gp_cov_ridge_penalty': 1.,
    'gp_output_bias_trainable': False,
    'gp_scale_random_features': False,
    'gp_use_custom_random_features': True,
    'gp_random_feature_type': 'orf',
    'gp_output_imagenet_initializer': True,
    'num_classes': 10,
}

class SpectralNormResNet(DeepResNet):
    def __init__(self, input_dim, num_layers=3, num_hidden=128,
                 activation="relu", num_outputs=1, dropout_rate=0.1,
                 coeff=0.95, n_power_iterations=1):
        super().__init__(input_dim, num_layers, num_hidden, activation, num_outputs, dropout_rate)
        self.coeff = coeff
        self.n_power_iterations = n_power_iterations
        self._apply_spectral_norm()

    def _apply_spectral_norm(self):
        self.input_layer = spectral_norm_fc(self.input_layer, coeff=self.coeff, n_power_iterations=self.n_power_iterations)
        self.residual_layers = nn.ModuleList([
            spectral_norm_fc(layer, coeff=self.coeff, n_power_iterations=self.n_power_iterations)
            for layer in self.residual_layers
        ])
        self.classifier = spectral_norm_fc(self.classifier, coeff=self.coeff, n_power_iterations=self.n_power_iterations)

def get_spectral_norm_resnet(input_dim, num_layers=3, num_hidden=128,
                             activation="relu", num_outputs=1, dropout_rate=0.1,
                             coeff=0.95, n_power_iterations=1):
    return SpectralNormResNet(input_dim, num_layers, num_hidden, activation, num_outputs, dropout_rate, coeff, n_power_iterations)


def build_sngp_model(input_dim):
    sngp_model = get_spectral_norm_resnet(input_dim=input_dim, num_layers=3, num_hidden=128, activation="relu",
                                     num_outputs=10, dropout_rate=0.1, coeff=0.95, n_power_iterations=1)
    replace_layer_with_gaussian(container=sngp_model, signature="classifier", **GP_KWARGS)
    return sngp_model.to(device)


# 1. Define input dimension and batch size
input_dimension = 10  # Example input dimension
batch_size = 32      # Example batch size
# 2. Generate random input data using torch
input_data = torch.randn(batch_size, input_dimension).to(device)

sngp_model = build_sngp_model(input_dimension)

print("Model architecture:")
print(sngp_model)


sngp_model.classifier.update_covariance_matrix()

# 4. Pass the input data through the model
output, hidden= sngp_model(input_data, kwargs={"update_precision_matrix": False, "return_covariance": True}, return_hidden=True)
y_pred, cov = output
uncertainty = torch.diagonal(cov, 0)

print("Predictions shape:", y_pred.shape)
print("predictions:", y_pred)
print("Covariance shape:", cov.shape)
print("Uncertainty:", uncertainty)






