from .lpips import LPIPS, normalize_tensor, spatial_average, upsample


class CustomLPIPS(LPIPS):
    """Custom version of LPIPS which returns layer-wise output without upsampling."""

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (
            (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == "0.1" else (in0, in1)
        )
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = (
                normalize_tensor(outs0[kk]),
                normalize_tensor(outs1[kk]),
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if self.lpips:
            if self.spatial:
                res_no_up = [self.lins[kk](diffs[kk]) for kk in range(self.L)]
                res = [upsample(res_no_up[kk], out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
                res_no_up = res
        else:
            if self.spatial:
                res_no_up = [diffs[kk].sum(dim=1, keepdim=True) for kk in range(self.L)]
                res = [upsample(res_no_up[kk], out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [
                    spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)
                ]
                res_no_up = res

        val = 0
        for layer in range(self.L):
            val += res[layer]

        if retPerLayer:
            return (val, res_no_up)
        else:
            return val
