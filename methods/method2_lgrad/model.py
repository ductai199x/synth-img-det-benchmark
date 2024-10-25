import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, CenterCrop

class LGradModel(nn.Module):
    def __init__(self, discriminator: nn.Module, classifier: nn.Module):
        super().__init__()
        self.discriminator = discriminator.eval()
        self.classifier = classifier

        self.clf_transforms = Compose([
            CenterCrop(256),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_grad(self, x):
        x = x.clone().detach().requires_grad_(True)
        pre = self.discriminator(x)
        self.discriminator.zero_grad()
        grad = torch.autograd.grad(pre.sum(), x, create_graph=True, retain_graph=True, allow_unused=False)[0]
        grad = grad.detach()
        grad = grad.sub(grad.min()).div(grad.max() - grad.min())
        grad = (grad * 255).int().float().div(255)
        return grad

    def forward(self, x):
        with torch.set_grad_enabled(True):
            grad = self.get_grad(x)
        return self.classifier(self.clf_transforms(grad))

    def forward_intermediate(self, x):
        with torch.set_grad_enabled(True):
            grad = self.get_grad(x)
        return self.classifier.forward_intermediate(self.clf_transforms(grad))
