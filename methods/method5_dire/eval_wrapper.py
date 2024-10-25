import webdataset as wds
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from PIL import Image
from functools import partial
from ..base_eval_wrapper import *
from ..common_transforms_utils import Compose, Resize, Normalize, CenterCrop, to_pil_image, pil_to_tensor, center_crop_no_pad


def key_gen():
    k = 0
    while True:
        yield str(k)
        k += 1


def dire_center_crop_fn(x):
        assert x.shape[0] == 1
        device = x.device
        x = x.squeeze(0)
        pil_x = to_pil_image(x)
        while min(*pil_x.size) >= 2 * 256:
            pil_x = pil_x.resize(tuple(x // 2 for x in pil_x.size), resample=Image.Resampling.BOX)
        scale = 256 / min(*pil_x.size)
        pil_x = pil_x.resize(tuple(int(x * scale) for x in pil_x.size), resample=Image.Resampling.BICUBIC)

        left, upper, right, lower = (pil_x.size[0] - 256) // 2, (pil_x.size[1] - 256) // 2, (pil_x.size[0] + 256) // 2, (pil_x.size[1] + 256) // 2
        pil_x = pil_x.crop((left, upper, right, lower))
        x = pil_to_tensor(pil_x).float().div(255).unsqueeze(0).to(device)
        return x


class DIREEvalWrapper(BaseEvalWrapper):
    def metrics(self):
        self.accuracy = BinaryAccuracy(threshold=0.0)
        self.auroc = BinaryAUROC()

    def transforms_fn(self, x):
        if self.transforms is None:
            self.transforms = Compose(
                [
                    Resize(256, interpolation=Image.Resampling.BICUBIC),
                    CenterCrop(256),
                ]
            )
        return self.transforms(x)

    def save_logits_fn(self, logits, targets):
        if self.save_logits_target is None:
            self.save_logits_target = wds.ShardWriter(f"{self.output_dir}/logits-%06d.tar", maxcount=1e4)
            self.logits_key_generator = key_gen()

        for logit, target in zip(logits, targets):
            self.save_logits_target.write(
                {
                    "__key__": next(self.logits_key_generator),
                    "logit.pth": logit.cpu(),
                    "target.cls": target.squeeze().item(),
                }
            )

    def save_embeds_fn(self, embeds, targets):
        if self.save_embeds_target is None:
            self.save_embeds_target = wds.ShardWriter(f"{self.output_dir}/embeds-%06d.tar", maxcount=1e4)
            self.embeds_key_generator = key_gen()

        for embed, target in zip(embeds, targets):
            self.save_embeds_target.write(
                {
                    "__key__": next(self.embeds_key_generator),
                    "embed.pth": embed.cpu(),
                    "target.cls": target.squeeze().item(),
                }
            )

    def save_results_fn(self):
        if self.is_saving:
            with open(f"{self.output_dir}/auroc.txt", "w") as f:
                f.write(f"{str(self.auroc.compute().item())}\n")
