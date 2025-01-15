import torch
import torch.nn as nn
import torchvision
from sngp_wrapper.covert_utils import replace_layer_with_gaussian, convert_to_sn_my
import torch.nn.functional as F

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
    return MNISTClassifier(num_classes=num_classes)

def Build_SNGP_MNISTClassifier(num_classes=10, coeff=0.95):
    model = Build_MNISTClassifier(num_classes=num_classes)
    GP_KWARGS = {
        'num_inducing': 1024,
        'gp_scale': 1.0,
        'gp_bias': 0.,
        'gp_kernel_type': 'gaussian', #  linear
        'gp_input_normalization': False,  #####
        'gp_cov_discount_factor': -1,
        'gp_cov_ridge_penalty': 1.,
        'gp_output_bias_trainable': False,
        'gp_scale_random_features': False,
        'gp_use_custom_random_features': True,
        'gp_random_feature_type': 'orf',
        'gp_output_imagenet_initializer': True,
        'num_classes': num_classes }

    spec_norm_replace_list = ["Linear", "Conv2D"]
    model = convert_to_sn_my(model, spec_norm_replace_list, coeff)
    replace_layer_with_gaussian(container=model, signature="classifier", **GP_KWARGS)
    return model
