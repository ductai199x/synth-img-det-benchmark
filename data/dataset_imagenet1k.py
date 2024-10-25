import json
import torch
from urllib.request import Request, urlopen
from webdataset import WebDataset, decode, map, split_by_node, ignore_and_continue
from torchvision.transforms import Compose, RandomCrop
from torchvision.transforms.functional import pil_to_tensor
from huggingface_hub import get_token
from braceexpand import braceexpand
from io import BytesIO
from PIL import Image
from typing import List, Union


__all__ = ["ImageNet1kWebdataset"]
# "https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main"
# "file:///media/nas2/Datasets/imagenet-1k-wds"


class ImageNet1kWebdataset(WebDataset):
    def __init__(
        self,
        split: str,
        max_num_examples: int = None,
        decode_type: str = "torchrgb",
        transforms=None,
        shardshuffle=True,
        debug=False,
        **kwargs,
    ):
        assert split in ["train", "val", "test"], f"Split {split} must be one of ['train', 'val', 'test']"
        if split in ["val", "test"]:
            split = "validation"
        self.debug = debug

        hf_token = get_token()
        self.base_url = "file:///media/nas2/Datasets/imagenet-1k-wds"

        info = json.loads(
            urlopen(
                Request(f"{self.base_url}/_info.json", headers={"Authorization": f"Bearer {hf_token}"})
            ).read()
        )

        self.num_examples = int(info["splits"][split]["num_samples"])
        file_names = info["splits"][split]["filenames"]
        file_counts = info["splits"][split]["shard_lengths"]
        if max_num_examples is not None:
            name_vs_count = list(zip(file_names, file_counts))
            total_count = 0
            file_names_subset = []
            for name, count in name_vs_count:
                total_count += count
                if total_count <= max_num_examples:
                    file_names_subset.append(name)
                else:
                    break
            file_names = file_names_subset
            self.num_examples = int(max_num_examples)

        file_urls = [f"{self.base_url}/{fn}" for fn in file_names]
        curl_cmds = [
            f"pipe:curl -s -L {url} -H 'Authorization:Bearer {hf_token}' || true" for url in file_urls
        ]
        super().__init__(
            curl_cmds, shardshuffle=shardshuffle, nodesplitter=split_by_node, handler=ignore_and_continue
        )

        self.transforms = self.get_transforms(transforms)

        # Append pipes to the pipeline
        self.append(decode(decode_type))
        self.append(map(self.get_data))

    def get_transforms(self, other_transforms):
        transforms = [] + [] if other_transforms is None else [other_transforms]
        return Compose(transforms)

    def get_data(self, example):
        image_key = next(k for k in example.keys() if k.endswith("jpg"))
        label_key = next(k for k in example.keys() if k.endswith("cls"))
        image = example[image_key]
        image = self.transforms(image)
        label = example[label_key]
        return image, label

    def __len__(self):
        return self.num_examples
