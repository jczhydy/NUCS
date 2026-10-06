import torch
import torch.nn as nn
import torchvision.models as models
import timm
from transformers import ViTForImageClassification

def create_model(model_name, num_classes, imgpretrain=True, device='cuda'):

    model = None

    # ResNet34
    if "resnet34" in model_name:
        if imgpretrain:
            model = models.resnet34(pretrained=True)
            print("Using ImageNet pretrained weights for ResNet34")
        else:
            model = models.resnet34(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    # ResNet18
    elif "resnet18" in model_name:
        if imgpretrain:
            model = models.resnet18(pretrained=True)
            print("Using ImageNet pretrained weights for ResNet18")
        else:
            model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    # EfficientNet-B0
    elif "efficientnet-b0" in model_name:
        if imgpretrain:
            model = EfficientNet.from_pretrained('efficientnet-b0')
            print("Using ImageNet pretrained weights for EfficientNet-B0")
        else:
            model = EfficientNet.from_name('efficientnet-b0')
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, num_classes)

    # ViT (Vision Transformer)
    elif "vit_l" in model_name:
        if imgpretrain:
            print("Using ImageNet pretrained weights for ViT")
            model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224-in21k')
            model.classifier =  torch.nn.Linear(model.classifier.in_features, num_classes)
        #model = ViTl(pretrained=imgpretrain, num_classes=num_classes)
        
    elif "vit_b" in model_name:
        if imgpretrain:
            print("Using ImageNet pretrained weights for ViT")
            model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
            model.classifier =  torch.nn.Linear(model.classifier.in_features, num_classes)
        #model = ViTl(pretrained=imgpretrain, num_classes=num_classes)
        
    elif "vit_h_14_224" in model_name:
        if imgpretrain:
            print("Using ImageNet pretrained weights for ViT")
       # model = ViTh(pretrained=imgpretrain, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    print(f"Predict class number: {num_classes}")
    model = model.to(device)
    return model


class ViTl(nn.Module):
    def __init__(self, pretrained=True, num_classes=10):
        super(ViTl, self).__init__()
        self.model = timm.create_model("vit_large_patch16_224", pretrained=pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
class ViTh(nn.Module):
    def __init__(self, pretrained=True, num_classes=10):
        super(ViTh, self).__init__()
        self.model = timm.create_model("vit_huge_patch14_224", pretrained=pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x