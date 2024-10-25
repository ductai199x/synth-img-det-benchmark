import torch
import torchattacks
import timm
from timm.data import resolve_model_data_config, create_transform
from torchattacks.attack import Attack
from torch.utils.data import Dataset, IterableDataset
from torchvision.transforms import Compose

from typing import Union


__all__ = ["AdversarialAttacksImageDataset"]


class ModelWithTransformsWrapper(torch.nn.Module):
    def __init__(self, model, transforms):
        super().__init__()
        self.model = model
        self.transforms = transforms

    def forward(self, x):
        return self.model(self.transforms(x))


class AdversarialAttacksImageDataset(IterableDataset):
    def __init__(
        self,
        dataset: Union[Dataset, IterableDataset],
        model_name: str,
        attack_name: str,
        attack_config: dict = dict(),
        device: str = "cpu",
        post_atk_transforms=None,
    ):
        self.dataset = dataset
        self.device = device
        model = timm.create_model(model_name, pretrained=True).eval().to(device)
        transforms = create_transform(**resolve_model_data_config(model), is_training=False)
        self.model = ModelWithTransformsWrapper(model, transforms)
        self.attack = self.get_attack(attack_name, attack_config)
        self.attack.device = device
        self.post_atk_transforms = self.get_transforms(post_atk_transforms)

    def __len__(self):
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset)
        else:
            print("Warning: Dataset does not have __len__ method. Returning None.")
            return None

    def __iter__(self):
        for data in self.dataset:
            image, label = data
            adv_image = self.attack(image.unsqueeze(0), torch.tensor(label).unsqueeze(0))
            adv_image = self.post_atk_transforms(adv_image.squeeze(0))
            yield adv_image, None

    def get_transforms(self, other_transforms):
        transforms = [] + [] if other_transforms is None else [other_transforms]
        return Compose(transforms)

    def get_attack(self, attack_name, attack_config) -> Attack:
        if attack_name.lower() == "fgsm":
            attack = torchattacks.FGSM(self.model, **attack_config)
        elif attack_name.lower() in ["ifgsm", "bim"]:
            attack = torchattacks.BIM(self.model, **attack_config)
        elif attack_name.lower() == "mifgsm":
            attack = torchattacks.MIFGSM(self.model, **attack_config)
        elif attack_name.lower() == "pgd":
            attack = torchattacks.PGD(self.model, **attack_config)
        elif attack_name.lower() == "cw":
            attack = torchattacks.CW(self.model, **attack_config)
        elif attack_name.lower() == "deepfool":
            attack = torchattacks.DeepFool(self.model, **attack_config)
        else:
            raise NotImplementedError(f"Attack {attack_name} not implemented")
        return attack
