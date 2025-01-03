
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from sngp_wrapper.covert_utils import replace_layer_with_gaussian, convert_to_sn_my

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

class ConvNextGP(nn.Module):
    def __init__(self, num_classes: int):
        super(ConvNextGP, self).__init__()
        self.model_ft = torchvision.models.convnext_tiny(weights="ConvNeXt_Tiny_Weights.DEFAULT").to(device)
        # Freeze the feature extraction layers
        # for param in model.features.parameters():
        #     param.requires_grad = False
        self.num_ftrs = self.model_ft.classifier[2].in_features
        self.model_ft.classifier = nn.Sequential(
            nn.Linear(self.num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        ).to(device)

    def forward(self, x, **kwargs):
        # Forward pass through the feature extractor
        x = self.model_ft.features(x)
        # Global average pooling
        x = self.model_ft.avgpool(x)
        # Flatten the tensor
        x = torch.flatten(x, 1)
        # Pass through the custom classifier
        x = self.model_ft.classifier(x)
        return x


def build_model(args, num_classes):
    if args.sngp:
        model = ConvNextGP(num_classes=num_classes)
        GP_KWARGS = {
            'num_inducing': 256,
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
        model = convert_to_sn_my(model, args.spec_norm_replace_list, args.coeff)
        replace_layer_with_gaussian(container=model, signature="classifier", **GP_KWARGS)
    elif args.snn:
        model = ConvNextGP(num_classes=num_classes)
    else:
        raise ValueError("Invalid model type")
    model = model.to(device)
    return model



