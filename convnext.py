import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNextGP(nn.Module):
    def __init__(self, num_classes: int):
        super(ConvNextGP, self).__init__()
        self.model_ft = torchvision.models.convnext_tiny(weights="ConvNeXt_Tiny_Weights.DEFAULT").to(device) # 768
        #self.feature_extractor = torchvision.models.convnext_base(weights="ConvNeXt_Base_Weights.DEFAULT").to(device) # 1024
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        self.num_ftrs = self.model_ft.fc.in_features
        # Replace the classifier with nn.Identity to keep the features unchanged
        # self.feature_extractor.classifier = nn.Identity()
        # self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.num_classes = num_classes

        if self.num_classes is not None:
            self.classifier = nn.Linear(768, num_classes) # please determine 768 by the classifier/head of the model
        else:
            self.classifier = None

    def forward(self, x, **kwargs):
        features = self.feature_extractor(x)
        features = self.flatten(features)
        if self.classifier is None:
            return features
        logits = self.classifier(features, **kwargs)
        if isinstance(logits, tuple):
            logits, uncertainty = logits
            prob = F.log_softmax(logits, dim=1)
            return prob, uncertainty
        else:
            return F.log_softmax(logits, dim=1)
