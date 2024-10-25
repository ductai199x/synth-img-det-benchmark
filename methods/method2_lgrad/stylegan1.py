"""
-------------------------------------------------
   File Name:    GAN.py
   Author:       Zhonghao Huang
   Date:         2019/10/17
   Description:  Modified from:
                 https://github.com/akanimax/pro_gan_pytorch
                 https://github.com/lernapparat/lernapparat
                 https://github.com/NVlabs/stylegan
-------------------------------------------------
"""

import copy
import datetime
import os
import random
import time
import timeit
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import interpolate
from torch.nn.modules.sparse import Embedding

# from data import get_data_loader


def update_average(model_tgt, model_src, beta):
    """
    update the model_target using exponential moving averages
    :param model_tgt: target model
    :param model_src: source model
    :param beta: value of decay beta
    :return: None (updates the target model)
    """

    # utility function for toggling the gradient requirements of the models
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    # turn off gradient calculation
    toggle_grad(model_tgt, False)
    toggle_grad(model_src, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_tgt
        p_tgt.copy_(beta * p_tgt + (1.0 - beta) * p_src)

    # turn back on the gradient calculation
    toggle_grad(model_tgt, True)
    toggle_grad(model_src, True)


# ----------------------------------------LOSSES----------------------------------------#
class GANLoss:
    """Base class for all losses

    @args:
    dis: Discriminator used for calculating the loss
         Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")


class ConditionalGANLoss:
    """Base class for all conditional losses"""

    def __init__(self, dis):
        self.criterion = BCEWithLogitsLoss()
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        # small assertion:
        assert real_samps.device == fake_samps.device, "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately :
        r_preds = self.dis(real_samps, height, alpha, labels_in=labels)
        f_preds = self.dis(fake_samps, height, alpha, labels_in=labels)

        # calculate the real loss:
        real_loss = self.criterion(torch.squeeze(r_preds), torch.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(torch.squeeze(f_preds), torch.zeros(fake_samps.shape[0]).to(device))

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps, labels, height, alpha):
        preds = self.dis(fake_samps, height, alpha, labels_in=labels)
        return self.criterion(torch.squeeze(preds), torch.ones(fake_samps.shape[0]).to(fake_samps.device))


# =============================================================
# Normal versions of the Losses:
# =============================================================


class StandardGAN(GANLoss):

    def __init__(self, dis):

        super().__init__(dis)

        # define the criterion and activation used for object
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # small assertion:
        assert real_samps.device == fake_samps.device, "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately :
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # calculate the real loss:
        real_loss = self.criterion(torch.squeeze(r_preds), torch.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(torch.squeeze(f_preds), torch.zeros(fake_samps.shape[0]).to(device))

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps, height, alpha):
        preds, _, _ = self.dis(fake_samps, height, alpha)
        return self.criterion(torch.squeeze(preds), torch.ones(fake_samps.shape[0]).to(fake_samps.device))


class HingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        loss = torch.mean(nn.ReLU()(1 - r_preds)) + torch.mean(nn.ReLU()(1 + f_preds))

        return loss

    def gen_loss(self, _, fake_samps, height, alpha):
        return -torch.mean(self.dis(fake_samps, height, alpha))


class RelativisticAverageHingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        loss = torch.mean(nn.ReLU()(1 - r_f_diff)) + torch.mean(nn.ReLU()(1 + f_r_diff))

        return loss

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        return torch.mean(nn.ReLU()(1 + r_f_diff)) + torch.mean(nn.ReLU()(1 - f_r_diff))


class LogisticGAN(GANLoss):
    def __init__(self, dis):
        super().__init__(dis)

    # gradient penalty
    def R1Penalty(self, real_img, height, alpha):

        # TODO: use_loss_scaling, for fp16
        apply_loss_scaling = lambda x: x * torch.exp(
            x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device)
        )
        undo_loss_scaling = lambda x: x * torch.exp(
            -x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device)
        )

        real_img = torch.autograd.Variable(real_img, requires_grad=True)
        real_logit = self.dis(real_img, height, alpha)
        # real_logit = apply_loss_scaling(torch.sum(real_logit))
        real_grads = torch.autograd.grad(
            outputs=real_logit,
            inputs=real_img,
            grad_outputs=torch.ones(real_logit.size()).to(real_img.device),
            create_graph=True,
            retain_graph=True,
        )[0].view(real_img.size(0), -1)
        # real_grads = undo_loss_scaling(real_grads)
        r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
        return r1_penalty

    def dis_loss(self, real_samps, fake_samps, height, alpha, r1_gamma=10.0):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        loss = torch.mean(nn.Softplus()(f_preds)) + torch.mean(nn.Softplus()(-r_preds))

        if r1_gamma != 0.0:
            r1_penalty = self.R1Penalty(real_samps.detach(), height, alpha) * (r1_gamma * 0.5)
            loss += r1_penalty

        return loss

    def gen_loss(self, _, fake_samps, height, alpha):
        f_preds = self.dis(fake_samps, height, alpha)

        return torch.mean(nn.Softplus()(-f_preds))


# ----------------------------------------LOSSES----------------------------------------#


# -------------------------------------CUSTOM LAYERS------------------------------------#


class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class Upscale2d(nn.Module):
    @staticmethod
    def upscale2d(x, factor=2, gain=1):
        assert x.dim() == 4
        if gain != 1:
            x = x * gain
        if factor != 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, factor, -1, factor)
            x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])
        return x

    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        return self.upscale2d(x, factor=self.factor, gain=self.gain)


class Downscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor
        self.gain = gain
        if factor == 2:
            f = [np.sqrt(gain) / factor] * factor
            self.blur = BlurLayer(kernel=f, normalize=False, stride=factor)
        else:
            self.blur = None

    def forward(self, x):
        assert x.dim() == 4
        # 2x2, float32 => downscale using _blur2d().
        if self.blur is not None and x.dtype == torch.float32:
            return self.blur(x)

        # Apply gain.
        if self.gain != 1:
            x = x * self.gain

        # No-op => early exit.
        if self.factor == 1:
            return x

        # Large factor => downscale using tf.nn.avg_pool().
        # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
        return F.avg_pool2d(x, self.factor)


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_size, output_size, gain=2**0.5, use_wscale=False, lrmul=1, bias=True):
        super().__init__()
        he_std = gain * input_size ** (-0.5)  # He init
        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul, bias)


class EqualizedConv2d(nn.Module):
    """Conv layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride=1,
        gain=2**0.5,
        use_wscale=False,
        lrmul=1,
        bias=True,
        intermediate=None,
        upscale=False,
        downscale=False,
    ):
        super().__init__()
        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None
        if downscale:
            self.downscale = Downscale2d()
        else:
            self.downscale = None
        he_std = gain * (input_channels * kernel_size**2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_mul = lrmul
        else:
            self.bias = None
        self.intermediate = intermediate

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul

        have_convolution = False
        if self.upscale is not None and min(x.shape[2:]) * 2 >= 128:
            # this is the fused upscale + conv from StyleGAN, sadly this seems incompatible with the non-fused way
            # this really needs to be cleaned up and go into the conv...
            w = self.weight * self.w_mul
            w = w.permute(1, 0, 2, 3)
            # probably applying a conv on w would be more efficient. also this quadruples the weight (average)?!
            w = F.pad(w, [1, 1, 1, 1])
            w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            x = F.conv_transpose2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
        elif self.upscale is not None:
            x = self.upscale(x)

        downscale = self.downscale
        intermediate = self.intermediate
        if downscale is not None and min(x.shape[2:]) >= 128:
            w = self.weight * self.w_mul
            w = F.pad(w, [1, 1, 1, 1])
            # in contrast to upscale, this is a mean...
            w = (
                w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            ) * 0.25  # avg_pool?
            x = F.conv2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
            downscale = None
        elif downscale is not None:
            assert intermediate is None
            intermediate = downscale

        if not have_convolution and intermediate is None:
            return F.conv2d(x, self.weight * self.w_mul, bias, padding=self.kernel_size // 2)
        elif not have_convolution:
            x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size // 2)

        if intermediate is not None:
            x = intermediate(x)

        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        return x


class NoiseLayer(nn.Module):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight"""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None

    def forward(self, x, noise=None):
        if noise is None and self.noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        elif noise is None:
            # here is a little trick: if you get all the noise layers and set each
            # modules .noise attribute, you can have pre-defined noise.
            # Very useful for analysis
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x


class StyleMod(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(StyleMod, self).__init__()
        self.lin = EqualizedLinear(latent_size, channels * 2, gain=1.0, use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.lin(latent)  # style => [batch_size, n_channels*2]

        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.0) + style[:, 1]
        return x


class LayerEpilogue(nn.Module):
    """Things to do at the end of each layer."""

    def __init__(
        self,
        channels,
        dlatent_size,
        use_wscale,
        use_noise,
        use_pixel_norm,
        use_instance_norm,
        use_styles,
        activation_layer,
    ):
        super().__init__()

        layers = []
        if use_noise:
            layers.append(("noise", NoiseLayer(channels)))
        layers.append(("activation", activation_layer))
        if use_pixel_norm:
            layers.append(("pixel_norm", PixelNormLayer()))
        if use_instance_norm:
            layers.append(("instance_norm", nn.InstanceNorm2d(channels)))

        self.top_epi = nn.Sequential(OrderedDict(layers))

        if use_styles:
            self.style_mod = StyleMod(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, dlatents_in_slice=None):
        x = self.top_epi(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        else:
            assert dlatents_in_slice is None
        return x


class BlurLayer(nn.Module):
    def __init__(self, kernel=None, normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        if kernel is None:
            kernel = [1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer("kernel", kernel)
        self.stride = stride

    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x, kernel, stride=self.stride, padding=int((self.kernel.size(2) - 1) / 2), groups=x.size(1)
        )
        return x


class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class StddevLayer(nn.Module):
    def __init__(self, group_size=4, num_new_features=1):
        super().__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def forward(self, x):
        b, c, h, w = x.shape
        group_size = min(self.group_size, b)
        y = x.reshape([group_size, -1, self.num_new_features, c // self.num_new_features, h, w])
        y = y - y.mean(0, keepdim=True)
        y = (y**2).mean(0, keepdim=True)
        y = (y + 1e-8) ** 0.5
        y = y.mean([3, 4, 5], keepdim=True).squeeze(3)  # don't keep the meaned-out channels
        y = y.expand(group_size, -1, -1, h, w).clone().reshape(b, self.num_new_features, h, w)
        z = torch.cat([x, y], dim=1)
        return z


class Truncation(nn.Module):
    def __init__(self, avg_latent, max_layer=8, threshold=0.7, beta=0.995):
        super().__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        self.beta = beta
        self.register_buffer("avg_latent", avg_latent)

    def update(self, last_avg):
        self.avg_latent.copy_(self.beta * self.avg_latent + (1.0 - self.beta) * last_avg)

    def forward(self, x):
        assert x.dim() == 3
        interp = torch.lerp(self.avg_latent, x, self.threshold)
        do_trunc = (torch.arange(x.size(1)) < self.max_layer).view(1, -1, 1).to(x.device)
        return torch.where(do_trunc, interp, x)


# -------------------------------------CUSTOM LAYERS------------------------------------#

# -------------------------------------CUSTOM BLOCKS------------------------------------#


class InputBlock(nn.Module):
    """
    The first block (4x4 "pixels") doesn't have an input.
    The result of the first convolution is just replaced by a (trained) constant.
    We call it the InputBlock, the others GSynthesisBlock.
    (It might be nicer to do this the other way round,
    i.e. have the LayerEpilogue be the Layer and call the conv from that.)
    """

    def __init__(
        self,
        nf,
        dlatent_size,
        const_input_layer,
        gain,
        use_wscale,
        use_noise,
        use_pixel_norm,
        use_instance_norm,
        use_styles,
        activation_layer,
    ):
        super().__init__()
        self.const_input_layer = const_input_layer
        self.nf = nf

        if self.const_input_layer:
            # called 'const' in tf
            self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
            self.bias = nn.Parameter(torch.ones(nf))
        else:
            self.dense = EqualizedLinear(dlatent_size, nf * 16, gain=gain / 4, use_wscale=use_wscale)
            # tweak gain to match the official implementation of Progressing GAN

        self.epi1 = LayerEpilogue(
            nf,
            dlatent_size,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer,
        )
        self.conv = EqualizedConv2d(nf, nf, 3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(
            nf,
            dlatent_size,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer,
        )

    def forward(self, dlatents_in_range):
        batch_size = dlatents_in_range.size(0)

        if self.const_input_layer:
            x = self.const.expand(batch_size, -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_in_range[:, 0]).view(batch_size, self.nf, 4, 4)

        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv(x)
        x = self.epi2(x, dlatents_in_range[:, 1])

        return x


class GSynthesisBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        blur_filter,
        dlatent_size,
        gain,
        use_wscale,
        use_noise,
        use_pixel_norm,
        use_instance_norm,
        use_styles,
        activation_layer,
    ):
        # 2**res x 2**res
        # res = 3..resolution_log2
        super().__init__()

        if blur_filter:
            blur = BlurLayer(blur_filter)
        else:
            blur = None

        self.conv0_up = EqualizedConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            gain=gain,
            use_wscale=use_wscale,
            intermediate=blur,
            upscale=True,
        )
        self.epi1 = LayerEpilogue(
            out_channels,
            dlatent_size,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer,
        )
        self.conv1 = EqualizedConv2d(
            out_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale
        )
        self.epi2 = LayerEpilogue(
            out_channels,
            dlatent_size,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer,
        )

    def forward(self, x, dlatents_in_range):
        x = self.conv0_up(x)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv1(x)
        x = self.epi2(x, dlatents_in_range[:, 1])
        return x


class DiscriminatorTop(nn.Sequential):
    def __init__(
        self,
        mbstd_group_size,
        mbstd_num_features,
        in_channels,
        intermediate_channels,
        gain,
        use_wscale,
        activation_layer,
        resolution=4,
        in_channels2=None,
        output_features=1,
        last_gain=1,
    ):
        """
        :param mbstd_group_size:
        :param mbstd_num_features:
        :param in_channels:
        :param intermediate_channels:
        :param gain:
        :param use_wscale:
        :param activation_layer:
        :param resolution:
        :param in_channels2:
        :param output_features:
        :param last_gain:
        """

        layers = []
        if mbstd_group_size > 1:
            layers.append(("stddev_layer", StddevLayer(mbstd_group_size, mbstd_num_features)))

        if in_channels2 is None:
            in_channels2 = in_channels

        layers.append(
            (
                "conv",
                EqualizedConv2d(
                    in_channels + mbstd_num_features,
                    in_channels2,
                    kernel_size=3,
                    gain=gain,
                    use_wscale=use_wscale,
                ),
            )
        )
        layers.append(("act0", activation_layer))
        layers.append(("view", View(-1)))
        layers.append(
            (
                "dense0",
                EqualizedLinear(
                    in_channels2 * resolution * resolution,
                    intermediate_channels,
                    gain=gain,
                    use_wscale=use_wscale,
                ),
            )
        )
        layers.append(("act1", activation_layer))
        layers.append(
            (
                "dense1",
                EqualizedLinear(
                    intermediate_channels, output_features, gain=last_gain, use_wscale=use_wscale
                ),
            )
        )

        super().__init__(OrderedDict(layers))


class DiscriminatorBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, gain, use_wscale, activation_layer, blur_kernel):
        super().__init__(
            OrderedDict(
                [
                    (
                        "conv0",
                        EqualizedConv2d(
                            in_channels, in_channels, kernel_size=3, gain=gain, use_wscale=use_wscale
                        ),
                    ),
                    # out channels nf(res-1)
                    ("act0", activation_layer),
                    ("blur", BlurLayer(kernel=blur_kernel)),
                    (
                        "conv1_down",
                        EqualizedConv2d(
                            in_channels,
                            out_channels,
                            kernel_size=3,
                            gain=gain,
                            use_wscale=use_wscale,
                            downscale=True,
                        ),
                    ),
                    ("act1", activation_layer),
                ]
            )
        )


# -------------------------------------CUSTOM BLOCKS------------------------------------#

# ----------------------------------MODEL DEFINITION------------------------------------#


class GMapping(nn.Module):

    def __init__(
        self,
        latent_size=512,
        dlatent_size=512,
        dlatent_broadcast=None,
        mapping_layers=8,
        mapping_fmaps=512,
        mapping_lrmul=0.01,
        mapping_nonlinearity="lrelu",
        use_wscale=True,
        normalize_latents=True,
        **kwargs
    ):
        """
        Mapping network used in the StyleGAN paper.

        :param latent_size: Latent vector(Z) dimensionality.
        # :param label_size: Label dimensionality, 0 if no labels.
        :param dlatent_size: Disentangled latent (W) dimensionality.
        :param dlatent_broadcast: Output disentangled latent (W) as [minibatch, dlatent_size]
                                  or [minibatch, dlatent_broadcast, dlatent_size].
        :param mapping_layers: Number of mapping layers.
        :param mapping_fmaps: Number of activations in the mapping layers.
        :param mapping_lrmul: Learning rate multiplier for the mapping layers.
        :param mapping_nonlinearity: Activation function: 'relu', 'lrelu'.
        :param use_wscale: Enable equalized learning rate?
        :param normalize_latents: Normalize latent vectors (Z) before feeding them to the mapping layers?
        :param kwargs: Ignore unrecognized keyword args.
        """

        super().__init__()

        self.latent_size = latent_size
        self.mapping_fmaps = mapping_fmaps
        self.dlatent_size = dlatent_size
        self.dlatent_broadcast = dlatent_broadcast

        # Activation function.
        act, gain = {
            "relu": (torch.relu, np.sqrt(2)),
            "lrelu": (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2)),
        }[mapping_nonlinearity]

        # Embed labels and concatenate them with latents.
        # TODO

        layers = []
        # Normalize latents.
        if normalize_latents:
            layers.append(("pixel_norm", PixelNormLayer()))

        # Mapping layers. (apply_bias?)
        layers.append(
            (
                "dense0",
                EqualizedLinear(
                    self.latent_size,
                    self.mapping_fmaps,
                    gain=gain,
                    lrmul=mapping_lrmul,
                    use_wscale=use_wscale,
                ),
            )
        )
        layers.append(("dense0_act", act))
        for layer_idx in range(1, mapping_layers):
            fmaps_in = self.mapping_fmaps
            fmaps_out = self.dlatent_size if layer_idx == mapping_layers - 1 else self.mapping_fmaps
            layers.append(
                (
                    "dense{:d}".format(layer_idx),
                    EqualizedLinear(
                        fmaps_in, fmaps_out, gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale
                    ),
                )
            )
            layers.append(("dense{:d}_act".format(layer_idx), act))

        # Output.
        self.map = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # First input: Latent vectors (Z) [mini_batch, latent_size].
        x = self.map(x)

        # Broadcast -> batch_size * dlatent_broadcast * dlatent_size
        if self.dlatent_broadcast is not None:
            x = x.unsqueeze(1).expand(-1, self.dlatent_broadcast, -1)
        return x


class GSynthesis(nn.Module):

    def __init__(
        self,
        dlatent_size=512,
        num_channels=3,
        resolution=1024,
        fmap_base=8192,
        fmap_decay=1.0,
        fmap_max=512,
        use_styles=True,
        const_input_layer=True,
        use_noise=True,
        nonlinearity="lrelu",
        use_wscale=True,
        use_pixel_norm=False,
        use_instance_norm=True,
        blur_filter=None,
        structure="linear",
        **kwargs
    ):
        """
        Synthesis network used in the StyleGAN paper.

        :param dlatent_size: Disentangled latent (W) dimensionality.
        :param num_channels: Number of output color channels.
        :param resolution: Output resolution.
        :param fmap_base: Overall multiplier for the number of feature maps.
        :param fmap_decay: log2 feature map reduction when doubling the resolution.
        :param fmap_max: Maximum number of feature maps in any layer.
        :param use_styles: Enable style inputs?
        :param const_input_layer: First layer is a learned constant?
        :param use_noise: Enable noise inputs?
        # :param randomize_noise: True = randomize noise inputs every time (non-deterministic),
                                  False = read noise inputs from variables.
        :param nonlinearity: Activation function: 'relu', 'lrelu'
        :param use_wscale: Enable equalized learning rate?
        :param use_pixel_norm: Enable pixel_wise feature vector normalization?
        :param use_instance_norm: Enable instance normalization?
        :param blur_filter: Low-pass filter to apply when resampling activations. None = no filtering.
        :param structure: 'fixed' = no progressive growing, 'linear' = human-readable
        :param kwargs: Ignore unrecognized keyword args.
        """

        super().__init__()

        # if blur_filter is None:
        #     blur_filter = [1, 2, 1]

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.structure = structure

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1

        self.num_layers = resolution_log2 * 2 - 2
        self.num_styles = self.num_layers if use_styles else 1

        act, gain = {
            "relu": (torch.relu, np.sqrt(2)),
            "lrelu": (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2)),
        }[nonlinearity]

        # Early layers.
        self.init_block = InputBlock(
            nf(1),
            dlatent_size,
            const_input_layer,
            gain,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            act,
        )
        # create the ToRGB layers for various outputs
        rgb_converters = [EqualizedConv2d(nf(1), num_channels, 1, gain=1, use_wscale=use_wscale)]

        # Building blocks for remaining layers.
        blocks = []
        for res in range(3, resolution_log2 + 1):
            last_channels = nf(res - 2)
            channels = nf(res - 1)
            # name = '{s}x{s}'.format(s=2 ** res)
            blocks.append(
                GSynthesisBlock(
                    last_channels,
                    channels,
                    blur_filter,
                    dlatent_size,
                    gain,
                    use_wscale,
                    use_noise,
                    use_pixel_norm,
                    use_instance_norm,
                    use_styles,
                    act,
                )
            )
            rgb_converters.append(EqualizedConv2d(channels, num_channels, 1, gain=1, use_wscale=use_wscale))

        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.ModuleList(rgb_converters)

        # register the temporary upsampler
        self.temporaryUpsampler = lambda x: interpolate(x, scale_factor=2)

    def forward(self, dlatents_in, depth=0, alpha=0.0, labels_in=None):
        """
        forward pass of the Generator
        :param dlatents_in: Input: Disentangled latents (W) [mini_batch, num_layers, dlatent_size].
        :param labels_in:
        :param depth: current depth from where output is required
        :param alpha: value of alpha for fade-in effect
        :return: y => output
        """

        assert depth < self.depth, "Requested output depth cannot be produced"

        if self.structure == "fixed":
            x = self.init_block(dlatents_in[:, 0:2])
            for i, block in enumerate(self.blocks):
                x = block(x, dlatents_in[:, 2 * (i + 1) : 2 * (i + 2)])
            images_out = self.to_rgb[-1](x)
        elif self.structure == "linear":
            x = self.init_block(dlatents_in[:, 0:2])

            if depth > 0:
                for i, block in enumerate(self.blocks[: depth - 1]):
                    x = block(x, dlatents_in[:, 2 * (i + 1) : 2 * (i + 2)])

                residual = self.to_rgb[depth - 1](self.temporaryUpsampler(x))
                straight = self.to_rgb[depth](
                    self.blocks[depth - 1](x, dlatents_in[:, 2 * depth : 2 * (depth + 1)])
                )

                images_out = (alpha * straight) + ((1 - alpha) * residual)
            else:
                images_out = self.to_rgb[0](x)
        else:
            raise KeyError("Unknown structure: ", self.structure)

        return images_out


class Generator(nn.Module):

    def __init__(
        self,
        resolution,
        latent_size=512,
        dlatent_size=512,
        conditional=False,
        n_classes=0,
        truncation_psi=0.7,
        truncation_cutoff=8,
        dlatent_avg_beta=0.995,
        style_mixing_prob=0.9,
        **kwargs
    ):
        """
        # Style-based generator used in the StyleGAN paper.
        # Composed of two sub-networks (G_mapping and G_synthesis).

        :param resolution:
        :param latent_size:
        :param dlatent_size:
        :param truncation_psi: Style strength multiplier for the truncation trick. None = disable.
        :param truncation_cutoff: Number of layers for which to apply the truncation trick. None = disable.
        :param dlatent_avg_beta: Decay for tracking the moving average of W during training. None = disable.
        :param style_mixing_prob: Probability of mixing styles during training. None = disable.
        :param kwargs: Arguments for sub-networks (G_mapping and G_synthesis).
        """

        super(Generator, self).__init__()

        if conditional:
            assert n_classes > 0, "Conditional generation requires n_class > 0"
            self.class_embedding = nn.Embedding(n_classes, latent_size)
            latent_size *= 2

        self.conditional = conditional
        self.style_mixing_prob = style_mixing_prob

        # Setup components.
        self.num_layers = (int(np.log2(resolution)) - 1) * 2
        self.g_mapping = GMapping(latent_size, dlatent_size, dlatent_broadcast=self.num_layers, **kwargs)
        self.g_synthesis = GSynthesis(resolution=resolution, **kwargs)

        if truncation_psi > 0:
            self.truncation = Truncation(
                avg_latent=torch.zeros(dlatent_size),
                max_layer=truncation_cutoff,
                threshold=truncation_psi,
                beta=dlatent_avg_beta,
            )
        else:
            self.truncation = None

    def forward(self, latents_in, depth, alpha, labels_in=None):
        """
        :param latents_in: First input: Latent vectors (Z) [mini_batch, latent_size].
        :param depth: current depth from where output is required
        :param alpha: value of alpha for fade-in effect
        :param labels_in: Second input: Conditioning labels [mini_batch, label_size].
        :return:
        """

        if not self.conditional:
            if labels_in is not None:
                warnings.warn("Generator is unconditional, labels_in will be ignored")
        else:
            assert labels_in is not None, "Conditional discriminatin requires labels"
            embedding = self.class_embedding(labels_in)
            latents_in = torch.cat([latents_in, embedding], 1)

        dlatents_in = self.g_mapping(latents_in)

        if self.training:
            # Update moving average of W(dlatent).
            # TODO
            if self.truncation is not None:
                self.truncation.update(dlatents_in[0, 0].detach())

            # Perform style mixing regularization.
            if self.style_mixing_prob is not None and self.style_mixing_prob > 0:
                latents2 = torch.randn(latents_in.shape).to(latents_in.device)
                dlatents2 = self.g_mapping(latents2)
                layer_idx = torch.from_numpy(np.arange(self.num_layers)[np.newaxis, :, np.newaxis]).to(
                    latents_in.device
                )
                cur_layers = 2 * (depth + 1)
                mixing_cutoff = (
                    random.randint(1, cur_layers) if random.random() < self.style_mixing_prob else cur_layers
                )
                dlatents_in = torch.where(layer_idx < mixing_cutoff, dlatents_in, dlatents2)

            # Apply truncation trick.
            if self.truncation is not None:
                dlatents_in = self.truncation(dlatents_in)

        fake_images = self.g_synthesis(dlatents_in, depth, alpha)

        return fake_images


class Discriminator(nn.Module):

    def __init__(
        self,
        resolution,
        num_channels=3,
        conditional=False,
        n_classes=0,
        fmap_base=8192,
        fmap_decay=1.0,
        fmap_max=512,
        nonlinearity="lrelu",
        use_wscale=True,
        mbstd_group_size=4,
        mbstd_num_features=1,
        blur_filter=None,
        structure="linear",
        **kwargs
    ):
        """
        Discriminator used in the StyleGAN paper.

        :param num_channels: Number of input color channels. Overridden based on dataset.
        :param resolution: Input resolution. Overridden based on dataset.
        # label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        :param fmap_base: Overall multiplier for the number of feature maps.
        :param fmap_decay: log2 feature map reduction when doubling the resolution.
        :param fmap_max: Maximum number of feature maps in any layer.
        :param nonlinearity: Activation function: 'relu', 'lrelu'
        :param use_wscale: Enable equalized learning rate?
        :param mbstd_group_size: Group size for the mini_batch standard deviation layer, 0 = disable.
        :param mbstd_num_features: Number of features for the mini_batch standard deviation layer.
        :param blur_filter: Low-pass filter to apply when resampling activations. None = no filtering.
        :param structure: 'fixed' = no progressive growing, 'linear' = human-readable
        :param kwargs: Ignore unrecognized keyword args.
        """
        super(Discriminator, self).__init__()

        if conditional:
            assert n_classes > 0, "Conditional Discriminator requires n_class > 0"
            # self.embedding = nn.Embedding(n_classes, num_channels * resolution ** 2)
            num_channels *= 2
            self.embeddings = []

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.img_resolution = resolution
        self.conditional = conditional
        self.mbstd_num_features = mbstd_num_features
        self.mbstd_group_size = mbstd_group_size
        self.structure = structure
        # if blur_filter is None:
        #     blur_filter = [1, 2, 1]

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1

        act, gain = {
            "relu": (torch.relu, np.sqrt(2)),
            "lrelu": (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2)),
        }[nonlinearity]

        # create the remaining layers
        blocks = []
        from_rgb = []
        for res in range(resolution_log2, 2, -1):
            # name = '{s}x{s}'.format(s=2 ** res)
            blocks.append(
                DiscriminatorBlock(
                    nf(res - 1),
                    nf(res - 2),
                    gain=gain,
                    use_wscale=use_wscale,
                    activation_layer=act,
                    blur_kernel=blur_filter,
                )
            )
            # create the fromRGB layers for various inputs:
            from_rgb.append(
                EqualizedConv2d(num_channels, nf(res - 1), kernel_size=1, gain=gain, use_wscale=use_wscale)
            )
            # Create embeddings for various inputs:
            if conditional:
                r = 2 ** (res)
                self.embeddings.append(Embedding(n_classes, (num_channels // 2) * r * r))

        if self.conditional:
            self.embeddings.append(nn.Embedding(n_classes, (num_channels // 2) * 4 * 4))
            self.embeddings = nn.ModuleList(self.embeddings)

        self.blocks = nn.ModuleList(blocks)

        # Building the final block.
        self.final_block = DiscriminatorTop(
            self.mbstd_group_size,
            self.mbstd_num_features,
            in_channels=nf(2),
            intermediate_channels=nf(2),
            gain=gain,
            use_wscale=use_wscale,
            activation_layer=act,
        )
        from_rgb.append(EqualizedConv2d(num_channels, nf(2), kernel_size=1, gain=gain, use_wscale=use_wscale))
        self.from_rgb = nn.ModuleList(from_rgb)

        # register the temporary downSampler
        self.temporaryDownsampler = nn.AvgPool2d(2)

    def forward(self, images_in, depth=None, alpha=1.0, labels_in=None):
        """
        :param images_in: First input: Images [mini_batch, channel, height, width].
        :param labels_in: Second input: Labels [mini_batch, label_size].
        :param depth: current height of operation (Progressive GAN)
        :param alpha: current value of alpha for fade-in
        :return:
        """
        if depth is None:
            depth = self.depth - 1
        assert depth < self.depth, "Requested output depth cannot be produced"

        if self.conditional:
            assert labels_in is not None, "Conditional Discriminator requires labels"
        # print(embedding_in.shape, images_in.shape)
        # exit(0)
        # print(self.embeddings)
        # exit(0)
        if self.structure == "fixed":
            if self.conditional:
                embedding_in = self.embeddings[0](labels_in)
                embedding_in = embedding_in.view(
                    images_in.shape[0], -1, images_in.shape[2], images_in.shape[3]
                )
                images_in = torch.cat([images_in, embedding_in], dim=1)
            x = self.from_rgb[0](images_in)
            for i, block in enumerate(self.blocks):
                x = block(x)
            scores_out = self.final_block(x)

        elif self.structure == "linear":
            if depth > 0:
                if self.conditional:
                    embedding_in = self.embeddings[self.depth - depth - 1](labels_in)
                    embedding_in = embedding_in.view(
                        images_in.shape[0], -1, images_in.shape[2], images_in.shape[3]
                    )
                    images_in = torch.cat([images_in, embedding_in], dim=1)

                residual = self.from_rgb[self.depth - depth](self.temporaryDownsampler(images_in))
                straight = self.blocks[self.depth - depth - 1](
                    self.from_rgb[self.depth - depth - 1](images_in)
                )
                x = (alpha * straight) + ((1 - alpha) * residual)

                for block in self.blocks[(self.depth - depth) :]:
                    x = block(x)
            else:
                if self.conditional:
                    embedding_in = self.embeddings[-1](labels_in)
                    embedding_in = embedding_in.view(
                        images_in.shape[0], -1, images_in.shape[2], images_in.shape[3]
                    )
                    images_in = torch.cat([images_in, embedding_in], dim=1)
                x = self.from_rgb[-1](images_in)

            scores_out = self.final_block(x)
        else:
            raise KeyError("Unknown structure: ", self.structure)

        return scores_out


class StyleGAN:
    def __init__(
        self,
        structure,
        resolution,
        num_channels,
        latent_size,
        g_args=dict(),
        d_args=dict(),
        g_opt_args=dict(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-8),
        d_opt_args=dict(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-8),
        conditional=False,
        n_classes=0,
        loss="relativistic-hinge",
        drift=0.001,
        d_repeats=1,
        use_ema=False,
        ema_decay=0.999,
        device=torch.device("cpu"),
    ):
        """
        Wrapper around the Generator and the Discriminator.

        :param structure: 'fixed' = no progressive growing, 'linear' = human-readable
        :param resolution: Input resolution. Overridden based on dataset.
        :param num_channels: Number of input color channels. Overridden based on dataset.
        :param latent_size: Latent size of the manifold used by the GAN
        :param g_args: Options for generator network.
        :param d_args: Options for discriminator network.
        :param g_opt_args: Options for generator optimizer.
        :param d_opt_args: Options for discriminator optimizer.
        :param loss: the loss function to be used
                     Can either be a string =>
                          ["wgan-gp", "wgan", "lsgan", "lsgan-with-sigmoid",
                          "hinge", "standard-gan" or "relativistic-hinge"]
                     Or an instance of GANLoss
        :param drift: drift penalty for the
                      (Used only if loss is wgan or wgan-gp)
        :param d_repeats: How many times the discriminator is trained per G iteration.
        :param use_ema: boolean for whether to use exponential moving averages
        :param ema_decay: value of mu for ema
        :param device: device to run the GAN on (GPU / CPU)
        """

        # state of the object
        assert structure in ["fixed", "linear"]

        # Check conditional validity
        if conditional:
            assert n_classes > 0, "Conditional GANs require n_classes > 0"
        self.structure = structure
        self.depth = int(np.log2(resolution)) - 1
        self.latent_size = latent_size
        self.device = device
        self.d_repeats = d_repeats
        self.conditional = conditional
        self.n_classes = n_classes

        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # Create the Generator and the Discriminator
        self.gen = Generator(
            num_channels=num_channels,
            resolution=resolution,
            structure=self.structure,
            conditional=self.conditional,
            n_classes=self.n_classes,
            **g_args
        ).to(self.device)

        self.dis = Discriminator(
            num_channels=num_channels,
            resolution=resolution,
            structure=self.structure,
            conditional=self.conditional,
            n_classes=self.n_classes,
            **d_args
        ).to(self.device)

        # if code is to be run on GPU, we can use DataParallel:
        # TODO

        # define the optimizers for the discriminator and generator
        self.__setup_gen_optim(**g_opt_args)
        self.__setup_dis_optim(**d_opt_args)

        # define the loss function used for training the GAN
        self.drift = drift
        self.loss = self.__setup_loss(loss)

        # Use of ema
        if self.use_ema:
            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)
            # updater function:
            self.ema_updater = update_average
            # initialize the gen_shadow weights equal to the weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

    def __setup_gen_optim(self, learning_rate, beta_1, beta_2, eps):
        self.gen_optim = torch.optim.Adam(
            self.gen.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps
        )

    def __setup_dis_optim(self, learning_rate, beta_1, beta_2, eps):
        self.dis_optim = torch.optim.Adam(
            self.dis.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps
        )

    def __setup_loss(self, loss):
        if isinstance(loss, str):
            loss = loss.lower()  # lowercase the string

            if not self.conditional:
                assert loss in [
                    "logistic",
                    "hinge",
                    "standard-gan",
                    "relativistic-hinge",
                ], "Unknown loss function"
                if loss == "logistic":
                    loss_func = LogisticGAN(self.dis)
                elif loss == "hinge":
                    loss_func = HingeGAN(self.dis)
                if loss == "standard-gan":
                    loss_func = StandardGAN(self.dis)
                elif loss == "relativistic-hinge":
                    loss_func = RelativisticAverageHingeGAN(self.dis)
            else:
                assert loss in ["conditional-loss"]
                if loss == "conditional-loss":
                    loss_func = ConditionalGANLoss(self.dis)

        return loss_func

    def __progressive_down_sampling(self, real_batch, depth, alpha):
        """
        private helper for down_sampling the original images in order to facilitate the
        progressive growing of the layers.

        :param real_batch: batch of real samples
        :param depth: depth at which training is going on
        :param alpha: current value of the fade-in alpha
        :return: real_samples => modified real batch of samples
        """

        from torch.nn import AvgPool2d
        from torch.nn.functional import interpolate

        if self.structure == "fixed":
            return real_batch

        # down_sample the real_batch for the given depth
        down_sample_factor = int(np.power(2, self.depth - depth - 1))
        prior_down_sample_factor = max(int(np.power(2, self.depth - depth)), 0)

        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)

        if depth > 0:
            prior_ds_real_samples = interpolate(
                AvgPool2d(prior_down_sample_factor)(real_batch), scale_factor=2
            )
        else:
            prior_ds_real_samples = ds_real_samples

        # real samples are a combination of ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        # return the so computed real_samples
        return real_samples

    def optimize_discriminator(self, noise, real_batch, depth, alpha, labels=None):
        """
        performs one step of weight update on discriminator using the batch of data

        :param noise: input noise of sample generation
        :param real_batch: real samples batch
        :param depth: current depth of optimization
        :param alpha: current alpha for fade-in
        :return: current loss (Wasserstein loss)
        """

        real_samples = self.__progressive_down_sampling(real_batch, depth, alpha)

        loss_val = 0
        for _ in range(self.d_repeats):
            # generate a batch of samples
            fake_samples = self.gen(noise, depth, alpha, labels).detach()

            if not self.conditional:
                loss = self.loss.dis_loss(real_samples, fake_samples, depth, alpha)
            else:
                loss = self.loss.dis_loss(real_samples, fake_samples, labels, depth, alpha)
            # optimize discriminator
            self.dis_optim.zero_grad()
            loss.backward()
            self.dis_optim.step()

            loss_val += loss.item()

        return loss_val / self.d_repeats

    def optimize_generator(self, noise, real_batch, depth, alpha, labels=None):
        """
        performs one step of weight update on generator for the given batch_size

        :param noise: input random noise required for generating samples
        :param real_batch: batch of real samples
        :param depth: depth of the network at which optimization is done
        :param alpha: value of alpha for fade-in effect
        :return: current loss (Wasserstein estimate)
        """

        real_samples = self.__progressive_down_sampling(real_batch, depth, alpha)

        # generate fake samples:
        fake_samples = self.gen(noise, depth, alpha, labels)

        # Change this implementation for making it compatible for relativisticGAN
        if not self.conditional:
            loss = self.loss.gen_loss(real_samples, fake_samples, depth, alpha)
        else:
            loss = self.loss.gen_loss(real_samples, fake_samples, labels, depth, alpha)

        # optimize the generator
        self.gen_optim.zero_grad()
        loss.backward()
        # Gradient Clipping
        nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=10.0)
        self.gen_optim.step()

        # if use_ema is true, apply ema to the generator parameters
        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        # return the loss value
        return loss.item()

    @staticmethod
    def create_grid(samples, scale_factor, img_file):
        """
        utility function to create a grid of GAN samples

        :param samples: generated samples for storing
        :param scale_factor: factor for upscaling the image
        :param img_file: name of file to write
        :return: None (saves a file)
        """
        from torch.nn.functional import interpolate
        from torchvision.utils import save_image

        # upsample the image
        if scale_factor > 1:
            samples = interpolate(samples, scale_factor=scale_factor)

        # save the images:
        save_image(
            samples,
            img_file,
            nrow=int(np.sqrt(len(samples))),
            normalize=True,
            scale_each=True,
            pad_value=128,
            padding=1,
        )

    def train(
        self,
        dataset,
        num_workers,
        epochs,
        batch_sizes,
        fade_in_percentage,
        logger,
        output,
        num_samples=36,
        start_depth=0,
        feedback_factor=100,
        checkpoint_factor=1,
    ):
        raise NotImplementedError
        """
        Utility method for training the GAN. Note that you don't have to necessarily use this
        you can use the optimize_generator and optimize_discriminator for your own training routine.

        :param dataset: object of the dataset used for training.
                        Note that this is not the data loader (we create data loader in this method
                        since the batch_sizes for resolutions can be different)
        :param num_workers: number of workers for reading the data. def=3
        :param epochs: list of number of epochs to train the network for every resolution
        :param batch_sizes: list of batch_sizes for every resolution
        :param fade_in_percentage: list of percentages of epochs per resolution used for fading in the new layer
                                   not used for first resolution, but dummy value still needed.
        :param logger:
        :param output: Output dir for samples,models,and log.
        :param num_samples: number of samples generated in sample_sheet. def=36
        :param start_depth: start training from this depth. def=0
        :param feedback_factor: number of logs per epoch. def=100
        :param checkpoint_factor:
        :return: None (Writes multiple files to disk)
        """

        assert self.depth <= len(epochs), "epochs not compatible with depth"
        assert self.depth <= len(batch_sizes), "batch_sizes not compatible with depth"
        assert self.depth <= len(fade_in_percentage), "fade_in_percentage not compatible with depth"

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()
        if self.use_ema:
            self.gen_shadow.train()

        # create a global time counter
        global_time = time.time()

        # create fixed_input for debugging
        fixed_input = torch.randn(num_samples, self.latent_size).to(self.device)

        fixed_labels = None
        if self.conditional:
            fixed_labels = torch.linspace(0, self.n_classes - 1, num_samples).to(torch.int64).to(self.device)
        # config depend on structure
        logger.info("Starting the training process ... \n")
        if self.structure == "fixed":
            start_depth = self.depth - 1
        step = 1  # counter for number of iterations
        for current_depth in range(start_depth, self.depth):
            current_res = np.power(2, current_depth + 2)
            logger.info("Currently working on depth: %d", current_depth + 1)
            logger.info("Current resolution: %d x %d" % (current_res, current_res))

            ticker = 1

            # Choose training parameters and configure training ops.
            # TODO
            data = get_data_loader(dataset, batch_sizes[current_depth], num_workers)

            for epoch in range(1, epochs[current_depth] + 1):
                start = timeit.default_timer()  # record time at the start of epoch

                logger.info("Epoch: [%d]" % epoch)
                # total_batches = len(iter(data))
                total_batches = len(data)

                fade_point = int(
                    (fade_in_percentage[current_depth] / 100) * epochs[current_depth] * total_batches
                )

                for i, batch in enumerate(data, 1):
                    # calculate the alpha for fading in the layers
                    alpha = ticker / fade_point if ticker <= fade_point else 1

                    # extract current batch of data for training
                    if self.conditional:
                        images, labels = batch
                        labels = labels.to(self.device)
                    else:
                        images = batch
                        labels = None

                    images = images.to(self.device)

                    gan_input = torch.randn(images.shape[0], self.latent_size).to(self.device)

                    # optimize the discriminator:
                    dis_loss = self.optimize_discriminator(gan_input, images, current_depth, alpha, labels)

                    # optimize the generator:
                    gen_loss = self.optimize_generator(gan_input, images, current_depth, alpha, labels)

                    # provide a loss feedback
                    if i % int(total_batches / feedback_factor + 1) == 0 or i == 1:
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed)).split(".")[0]
                        logger.info(
                            "Elapsed: [%s] Step: %d  Batch: %d  D_Loss: %f  G_Loss: %f"
                            % (elapsed, step, i, dis_loss, gen_loss)
                        )

                        # create a grid of samples and save it
                        os.makedirs(os.path.join(output, "samples"), exist_ok=True)
                        gen_img_file = os.path.join(
                            output,
                            "samples",
                            "gen_" + str(current_depth) + "_" + str(epoch) + "_" + str(i) + ".png",
                        )

                        with torch.no_grad():
                            self.create_grid(
                                samples=(
                                    self.gen(
                                        fixed_input, current_depth, alpha, labels_in=fixed_labels
                                    ).detach()
                                    if not self.use_ema
                                    else self.gen_shadow(
                                        fixed_input, current_depth, alpha, labels_in=fixed_labels
                                    ).detach()
                                ),
                                scale_factor=(
                                    int(np.power(2, self.depth - current_depth - 1))
                                    if self.structure == "linear"
                                    else 1
                                ),
                                img_file=gen_img_file,
                            )

                    # increment the alpha ticker and the step
                    ticker += 1
                    step += 1

                elapsed = timeit.default_timer() - start
                elapsed = str(datetime.timedelta(seconds=elapsed)).split(".")[0]
                logger.info("Time taken for epoch: %s\n" % elapsed)

                if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == epochs[current_depth]:
                    save_dir = os.path.join(output, "models")
                    os.makedirs(save_dir, exist_ok=True)
                    gen_save_file = os.path.join(
                        save_dir, "GAN_GEN_" + str(current_depth) + "_" + str(epoch) + ".pth"
                    )
                    dis_save_file = os.path.join(
                        save_dir, "GAN_DIS_" + str(current_depth) + "_" + str(epoch) + ".pth"
                    )
                    gen_optim_save_file = os.path.join(
                        save_dir, "GAN_GEN_OPTIM_" + str(current_depth) + "_" + str(epoch) + ".pth"
                    )
                    dis_optim_save_file = os.path.join(
                        save_dir, "GAN_DIS_OPTIM_" + str(current_depth) + "_" + str(epoch) + ".pth"
                    )

                    torch.save(self.gen.state_dict(), gen_save_file)
                    logger.info("Saving the model to: %s\n" % gen_save_file)
                    torch.save(self.dis.state_dict(), dis_save_file)
                    torch.save(self.gen_optim.state_dict(), gen_optim_save_file)
                    torch.save(self.dis_optim.state_dict(), dis_optim_save_file)

                    # also save the shadow generator if use_ema is True
                    if self.use_ema:
                        gen_shadow_save_file = os.path.join(
                            save_dir, "GAN_GEN_SHADOW_" + str(current_depth) + "_" + str(epoch) + ".pth"
                        )
                        torch.save(self.gen_shadow.state_dict(), gen_shadow_save_file)
                        logger.info("Saving the model to: %s\n" % gen_shadow_save_file)

        logger.info("Training completed.\n")


# ----------------------------------MODEL DEFINITION------------------------------------#


if __name__ == "__main__":
    print("Done.")
