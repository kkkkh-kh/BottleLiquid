import torch.nn as nn
from torchvision import models


def _load_resnet18_imagenet():
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        return models.resnet18(weights=weights)
    except AttributeError:
        return models.resnet18(pretrained=True)
    except TypeError:
        return models.resnet18(pretrained=True)


def build_resnet18_classifier(num_classes=2, freeze_backbone=True):
    model = _load_resnet18_imagenet()

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    return model


def build_resnet18_binary(freeze_backbone=True):
    return build_resnet18_classifier(num_classes=2, freeze_backbone=freeze_backbone)
