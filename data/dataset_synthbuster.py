import json
import torch
from torchvision.transforms import Compose
from urllib.request import Request, urlopen
from webdataset import WebDataset, decode, map, split_by_node, ignore_and_continue
from braceexpand import braceexpand
from huggingface_hub import get_token
from typing import Union, List


__all__ = ["SynthbusterWebdataset"]


BASE_URL = "file:///media/nas2/Datasets/synthbuster/wds"


class SynthbusterWebdataset(WebDataset):
    def __init__(
        self,
        base_url: str = BASE_URL,
        decode_type: str = "torchrgb",
        calc_class_weights=True,
        ignore_classes: Union[None, List[int]] = None,
        transforms=None,
        shardshuffle=True,
        debug=False,
        **kwargs,
    ):
        split = "test"
        self.debug = debug
        ignore_classes_int = ignore_classes or []

        hf_token = get_token()

        summary = json.loads(
            urlopen(
                Request(f"{base_url}/summary.json", headers={"Authorization": f"Bearer {hf_token}"})
            ).read()
        )
        class_mapping = json.loads(
            urlopen(
                Request(f"{base_url}/class_mapping.json", headers={"Authorization": f"Bearer {hf_token}"})
            ).read()
        )

        self.label_count = {int(k): v for k, v in summary["label_count"][split].items()}
        int_to_str_label_mapping = {int(k): v for k, v in class_mapping["int_to_str"].items()}
        str_to_int_label_mapping = {k: int(v) for k, v in class_mapping["str_to_int"].items()}
        ignore_classes_str = [
            int_to_str_label_mapping[c] for c in ignore_classes_int if c in int_to_str_label_mapping
        ]
        print(f"Ignoring classes: {ignore_classes_str}")

        if ignore_classes_int:
            old_ids = sorted(list(int_to_str_label_mapping.keys()))
            for c in ignore_classes_int:
                old_ids.remove(c)
            new_ids = list(range(len(old_ids)))
            self.label_mapping = {old_id: new_id for old_id, new_id in zip(old_ids, new_ids)}
            self.label_mapping_str = {
                new_id: int_to_str_label_mapping[old_id] for old_id, new_id in zip(old_ids, new_ids)
            }
            print(f"Label remapping: {self.label_mapping}\n{self.label_mapping_str}")
        else:
            self.label_mapping = {k: k for k in int_to_str_label_mapping.keys()}
            self.label_mapping_str = int_to_str_label_mapping

        self.shard_urls = []
        self.num_examples = 0
        for string in summary[split]:
            arch = string.split("-")[0]
            arch_int = str_to_int_label_mapping[arch]
            if arch_int in ignore_classes_int:
                continue
            self.num_examples += self.label_count[arch_int]
            urls = [f"{base_url}/{shard_name}" for shard_name in braceexpand(string)]
            urls = [f"pipe:curl -s -L {url} -H 'Authorization:Bearer {hf_token}' || true" for url in urls]
            self.shard_urls += urls

        if calc_class_weights:
            _total = self.num_examples
            _num_classes = len(self.label_count) - len(ignore_classes_int)
            _average = _total / _num_classes
            self.class_weights = torch.zeros(len(self.label_mapping))
            for label, count in self.label_count.items():
                if label not in ignore_classes_int:
                    self.class_weights[self.label_mapping[label]] = _average / count
        else:
            self.class_weights = torch.ones(len(self.label_mapping))

        for i in range(len(self.class_weights)):
            assert (
                self.class_weights[i] > 0
            ), f"Class {self.label_mapping_str[i]} has weight {self.class_weights[i]}"
            # print(f"Class {self.label_mapping_str[i]}: {self.class_weights[i]}")

        super().__init__(
            self.shard_urls,
            shardshuffle=shardshuffle,
            nodesplitter=split_by_node,
            handler=ignore_and_continue,
        )

        self.transforms = self.get_transforms(transforms)

        # Append pipes to the pipeline
        self.append(decode(decode_type))
        self.append(map(self.get_data))

    def get_transforms(self, other_transforms):
        transforms = [] if other_transforms is None else [other_transforms]
        return Compose(transforms)

    def get_data(self, example):
        image_key = next(k for k in example.keys() if k.endswith("png"))
        label_key = next(k for k in example.keys() if k.endswith("cls"))
        image = example[image_key]
        image = self.transforms(image)
        label = example[label_key]
        new_label = self.label_mapping.get(label, -1)
        if self.debug:
            print(f"Label {label} is remapped to {new_label}. Extra info: {example['png.json']}")
        return image, new_label

    def __len__(self):
        return self.num_examples
