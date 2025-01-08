import torch
import torch.nn as nn
import torchvision
from sngp_wrapper.covert_utils import replace_layer_with_gaussian, convert_to_sn_my
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# class ConvNextGP(nn.Module):
#     def __init__(self, num_classes: int):
#         super(ConvNextGP, self).__init__()
#         feature_extractor = torchvision.models.convnext_tiny(weights="ConvNeXt_Tiny_Weights.DEFAULT")
#         # remove the classifier layer
#         feature_extractor.classifier = nn.Identity()
#         self.feature_extractor = feature_extractor
#         self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
#         self.classifier = nn.Linear(768, num_classes)
#
#     def forward(self, x, return_hidden=False, **kwargs):
#         features = self.flatten(self.feature_extractor(x))
#         logits = self.classifier(features, **kwargs)
#         if return_hidden:
#             return logits, features
#         else:
#             return logits

# def build_sngp_model(num_classes):
#     model = ConvNextGP(num_classes=num_classes)
#     GP_KWARGS = {
#         'num_inducing': 1024,
#         'gp_scale': 1.0,
#         'gp_bias': 0.,
#         'gp_kernel_type': 'gaussian', # linear
#         'gp_input_normalization': True,
#         'gp_cov_discount_factor': -1,
#         'gp_cov_ridge_penalty': 1.,
#         'gp_output_bias_trainable': False,
#         'gp_scale_random_features': False,
#         'gp_use_custom_random_features': True,
#         'gp_random_feature_type': 'orf',
#         'gp_output_imagenet_initializer': True,
#         'num_classes': num_classes,
#     }
#     spec_norm_replace_list, coeff = ["Linear", "Conv2D"], 3.
#     model = convert_to_sn_my(model, spec_norm_replace_list, coeff)
#     replace_layer_with_gaussian(container=model, signature="classifier", **GP_KWARGS)
#     return model

class MNISTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(128, num_classes)
        self.dropout_rate = 0.1

    def forward(self, x, return_hidden=False, **kwargs):
        x = self.pool(self.relu(self.conv1(x)))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.pool(self.relu(self.conv2(x)))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        logits = self.classifier(x, **kwargs)
        if return_hidden:
            return logits, x
        else:
            return logits

def Build_MNISTClassifier(num_classes):
    model = MNISTClassifier(num_classes=num_classes)
    model = model.to(device)
    return model

def Build_SNGP_MNISTClassifier(num_classes=10, coeff=3.):
    model = Build_MNISTClassifier(num_classes=num_classes)
    GP_KWARGS = {
        'num_inducing': 1024,
        'gp_scale': 1.0,
        'gp_bias': 0.,
        'gp_kernel_type': 'gaussian', # linear
        'gp_input_normalization': True,
        'gp_cov_discount_factor': -1,
        'gp_cov_ridge_penalty': 1.,
        'gp_output_bias_trainable': False,
        'gp_scale_random_features': False,
        'gp_use_custom_random_features': True,
        'gp_random_feature_type': 'orf',
        'gp_output_imagenet_initializer': True,
        'num_classes': num_classes,
    }
    spec_norm_replace_list = ["Linear", "Conv2D"]
    model = convert_to_sn_my(model, spec_norm_replace_list, coeff)
    replace_layer_with_gaussian(container=model, signature="classifier", **GP_KWARGS)
    return model

# def main():
#     num_classes = 10
#     kwargs = {"return_random_features": False, "return_covariance":False,
#               "update_precision_matrix": False, "update_covariance_matrix": False }
#
#     model = Build_MNISTClassifier(num_classes).to(device)
#     sngp_model = Build_SNGP_MNISTClassifier(num_classes).to(device)
#     ind_data = torch.randn(10, 1, 28, 28).to(device)
#
#     logits, features = model(ind_data, return_hidden=True)
#     print("logits shape", logits.shape, "features shape", features.shape)
#
#
#     for _ in range(10):
#         sngp_model(ind_data, **{"update_precision_matrix": True})  # we remember the in-domain data
#
#     sngp_model.classifier.update_covariance_matrix()
#
#     ind_output =sngp_model(ind_data, **{"update_precision_matrix": False, "return_covariance": True})
#
#     ind_prob, ind_cov = ind_output
#
#     ind_uncertainty = torch.diagonal(ind_cov, 0)
#
#     print("ind_uncertainty", ind_uncertainty, "ind mean", torch.mean(ind_uncertainty))
#
#
# if __name__ == "__main__":
#     main()