import torch
import torch.nn as nn
from .srec import SrecModel, Bits


available_levels = [0, 1, 2]
available_features = [
    f"{type_}{level}" for level in available_levels for type_ in ["nll", "entropy", "d", "dlt"]
]
feature_to_idx = {
    f"{type_}{level}": (type_i, level)
    for level in available_levels
    for type_i, type_ in enumerate(["nll", "entropy", "d", "dlt"])
}


class ZEDModel(nn.Module):
    def __init__(self, srec: SrecModel, selected_feature="dlt0"):
        super().__init__()
        assert selected_feature in available_features, f"selected_feature must be one of {available_features}"

        self.idx_of_feature = feature_to_idx[selected_feature]
        print(self.idx_of_feature)

        self.srec = srec
        self.srec_loss_fn = self.srec.loss_fn

        # preallocate tensor for the vectorized entropy calculation
        max_h, max_w = 1024, 1024
        self.entropy_ys = nn.Parameter(
            torch.arange(0, 256) # pixel values ranging from 0 to 255
            .view(256, 1, 1, 1) # 256x1x1x1
            .expand(256, 3, max_h, max_w) # make a mask of 3xHxW for each pixel value
            .reshape(8, 32, 3, max_h, max_w), # split the pixel values into 8 groups of 32 to reduce memory usage during loss calculation
            requires_grad=False,
        )

    def get_features(self, bits: Bits):
        bits_keys = list(bits.get_keys())
        keys_with_lm_probs = {}
        for i, k in enumerate(bits_keys):
            mode, type_ = k.split("/")
            assert mode == "eval", f"only eval mode is supported, got {mode}"
            if "codes" in type_ or "rounding" in type_:
                continue
            level, pixel_group = type_.split("_")
            level, pixel_group = int(level), int(pixel_group)
            if level not in keys_with_lm_probs:
                keys_with_lm_probs[level] = []
            keys_with_lm_probs[level].append(
                {
                    "pixel_group": pixel_group,
                    "y": bits.probs[i][0].detach(),
                    "log_probs": bits.probs[i][1].probs.detach(),
                }
            )

        features = {}
        for level, pixel_groups in keys_with_lm_probs.items():
            nll_level = []
            entropy_level = []
            for pixel_group in pixel_groups:
                y_group, log_probs_group = pixel_group["y"], pixel_group["log_probs"]
                nll_group = self.srec_loss_fn(y_group, log_probs_group).detach()  # 1x3xHxW
                nll_group = nll_group.sum(dim=1)  # 1xHxW

                y_pixel = self.entropy_ys[:, :, :, : y_group.size(2), : y_group.size(3)]
                nll_pixel = torch.cat(
                    [
                        self.srec_loss_fn(y_pixel_group, log_probs_group.repeat(32, 1, 1, 1)).detach()
                        for y_pixel_group in y_pixel
                    ]
                )  # 256x3xHxW
                ll_pixel = torch.exp(-nll_pixel)
                entropy_group = ll_pixel * nll_pixel  # 256x3xHxW

                entropy_group = entropy_group.sum(dim=0)  # 1x3xHxW
                entropy_group = entropy_group.sum(dim=0)  # 1xHxW

                nll_level.append(nll_group.view(1, -1))
                entropy_level.append(entropy_group.view(1, -1))
            nll_level = torch.cat(nll_level, dim=1).mean(dim=1)
            entropy_level = torch.cat(entropy_level, dim=1).mean(dim=1)
            D_level = nll_level - entropy_level
            features[level] = {
                "nll": nll_level,
                "entropy": entropy_level,
                "D": D_level,
            }

        NLLs, Hs, Ds = [], [], []
        for level, feature in list(features.items())[::-1]:
            NLLs.append(feature["nll"][0])
            Hs.append(feature["entropy"][0])
            Ds.append(feature["D"][0])
        Dlts = [Ds[0] - Ds[1], Ds[1] - Ds[2], Ds[0] - Ds[2]]
        feature_tensor = torch.tensor([NLLs, Hs, Ds, Dlts]).to(NLLs[0].device)
        return feature_tensor

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        type_i, level = self.idx_of_feature
        features = []
        for x_i in x:
            x_i = x_i.unsqueeze(0)
            bits_i = self.srec(x)
            all_features_i = self.get_features(bits_i)
            feature_i = all_features_i[type_i, level].unsqueeze(0)
            features.append(feature_i)
        features = torch.stack(features)
        return -features

    def forward_intermediate(self, x: torch.Tensor):
        B, C, H, W = x.shape
        type_i, level = self.idx_of_feature
        features = []
        all_features = []
        for x_i in x:
            x_i = x_i.unsqueeze(0)
            bits_i = self.srec(x)
            all_features_i = self.get_features(bits_i)
            feature_i = all_features_i[type_i, level].unsqueeze(0)
            features.append(feature_i)
            all_features.append(all_features_i.flatten())
        features = torch.stack(features)
        all_features = torch.stack(all_features)
        return -features, all_features
