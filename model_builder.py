import torch
import torch.nn as nn
import torchvision
from sngp_wrapper.covert_utils import replace_layer_with_gaussian, convert_to_sn_my
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNextGP(nn.Module):
    def __init__(self, num_classes: int):
        super(ConvNextGP, self).__init__()
        feature_extractor = torchvision.models.convnext_tiny(weights="ConvNeXt_Tiny_Weights.DEFAULT")
        feature_extractor.classifier = nn.Identity()
        self.feature_extractor = feature_extractor
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.classifier = nn.Linear(768, num_classes)
    def forward(self, x, **kwargs):
        features = self.flatten(self.feature_extractor(x))
        output = self.classifier(features, **kwargs)
        return output

def build_model(num_classes):
    model = ConvNextGP(num_classes=num_classes)
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
    spec_norm_replace_list, coeff = ["Linear", "Conv2D"], 3.
    model = convert_to_sn_my(model, spec_norm_replace_list, coeff)
    replace_layer_with_gaussian(container=model, signature="classifier", **GP_KWARGS)
    return model

def main():
    num_classes = 10
    kwargs = {"return_random_features": False, "return_covariance":False,
              "update_precision_matrix": False, "update_covariance_matrix": False }

    model = build_model(num_classes).to(device)
    ind_data = torch.randn(10, 3, 224, 224).to(device)
    ood_data = (torch.randn(10, 3, 224, 224) + 10).to(device)

    for _ in range(10):
        model(ind_data, **{"update_precision_matrix": True})  # we remember the in-domain data

    model.classifier.update_covariance_matrix()

    ind_output = model(ind_data, **{"update_precision_matrix": False, "return_covariance": True, })
    ood_output = model(ood_data, **{"update_precision_matrix": False, "return_covariance": True, })

    ind_prob, ind_cov = ind_output
    ood_prob, ood_cov = ood_output

    ind_uncertainty = torch.diagonal(ind_cov, 0)
    ood_uncertainty = torch.diagonal(ood_cov, 0)

    print("ind_uncertainty", ind_uncertainty, "ind mean", torch.mean(ind_uncertainty))
    print("ood_uncertainty", ood_uncertainty, "ood mean", torch.mean(ood_uncertainty))


if __name__ == "__main__":
    main()