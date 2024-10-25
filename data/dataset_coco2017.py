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


__all__ = ["Coco2017Webdataset"]
# "https://huggingface.co/datasets/ductai199x/coco2017-caption-and-inst-seg/resolve/main"
# "file:///media/nas2/Datasets/coco2017/wds"


class Coco2017Webdataset(WebDataset):
    def __init__(
        self,
        split: str,
        decode_type: str = "torchrgb",
        transforms=None,
        shardshuffle=True,
        debug=False,
        output_label_type=None,
        **kwargs,
    ):
        assert split in ["train", "val", "test"], f"Split {split} must be one of ['train', 'val', 'test']"
        assert output_label_type in [
            None,
            "captions",
            "instances",
        ], f"Output label type {output_label_type} must be one of [None, 'captions', 'instances']"
        if not isinstance(output_label_type, list):
            output_label_type = [output_label_type]
        self.output_label_type = [t for t in output_label_type if t is not None]
        self.debug = debug

        hf_token = get_token()
        self.base_url = "file:///media/nas2/Datasets/coco2017/wds"

        summary = json.loads(
            urlopen(
                Request(f"{self.base_url}/summary.json", headers={"Authorization": f"Bearer {hf_token}"})
            ).read()
        )

        self.num_examples = int(summary["count"][split])
        url = f"{self.base_url}/{summary[split]}"
        curl_cmd = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {hf_token}' || true"
        super().__init__(
            curl_cmd, shardshuffle=shardshuffle, nodesplitter=split_by_node, handler=ignore_and_continue
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
        image = example[image_key]
        image = self.transforms(image)
        if len(self.output_label_type) == 0:
            return image, 0

        data = {"image": image}
        if "captions" in self.output_label_type:
            caption_key = "captions.json"
            caption = example[caption_key]
            data["caption"] = caption
        if "instances" in self.output_label_type:
            instances_key = "inst_segs.json"
            instances = example[instances_key]
            data["instances"] = instances
        return data

    def __len__(self):
        return self.num_examples
