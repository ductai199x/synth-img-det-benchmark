import webdataset as wds
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from ..base_eval_wrapper import *
from ..common_transforms_utils import Compose, Normalize, Resize


def key_gen():
    k = 0
    while True:
        yield str(k)
        k += 1


class LGradEvalWrapper(BaseEvalWrapper):
    def metrics(self):
        self.accuracy = BinaryAccuracy(threshold=0.0)
        self.auroc = BinaryAUROC()

    def transforms_fn(self, x):
        if self.transforms is None:
            self.transforms = Compose(
                [
                    Resize((self.model.discriminator.img_resolution, self.model.discriminator.img_resolution)),
                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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
