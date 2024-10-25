import torch
import os
from argparse import ArgumentParser
from lightning.pytorch import Trainer
from methods.method2_lgrad import (
    resnet50,
    StyleGan1Discriminator,
    StyleGan2Discriminator,
    LGradModel,
    LGradEvalWrapper,
)
from cmd_parser import parse_args
from prepare_data import prepare_data


variant_choices = ["default", "progan4c", "progan2c", "progan1c"]
sg2_dis_sd_paths = [
    "methods/method2_lgrad/weights/stylegan2-ffhq-256x256.discriminator.pth",
    "methods/method2_lgrad/weights/stylegan2-ffhqu-256x256.discriminator.pth",
    "methods/method2_lgrad/weights/stylegan2-celebahq-256x256.discriminator.pth",
]
sg1_dis_sd_paths = [
    "methods/method2_lgrad/weights/stylegan1-bedrooms-256x256.discriminator.pth",
    "methods/method2_lgrad/weights/stylegan1-cats-256x256.discriminator.pth",
]

use_stylegan2 = False


def main():
    parser = ArgumentParser()
    args = parse_args(parser, variant_choices)

    if args.variant in ["default", "progan4c"]:
        clf_sd_path = "methods/method2_lgrad/weights/LGrad-4class-Trainon-Progan_car_cat_chair_horse.pth"
    elif args.variant == "progan2c":
        clf_sd_path = "methods/method2_lgrad/weights/LGrad-2class-Trainon-Progan_chair_horse.pth"
    elif args.variant == "progan1c":
        clf_sd_path = "methods/method2_lgrad/weights/LGrad-1class-Trainon-Progan_horse.pth"

    clf_state_dict = torch.load(clf_sd_path, map_location="cpu")
    clf = resnet50(num_classes=1)
    clf.load_state_dict(clf_state_dict)
    clf.eval()

    if use_stylegan2:
        dis_state_dict = torch.load(sg2_dis_sd_paths[1], map_location="cpu")
        discriminator = StyleGan2Discriminator(
            c_dim=0, img_resolution=256, img_channels=3, channel_base=32768 // 2
        )
    else:
        dis_state_dict = torch.load(sg1_dis_sd_paths[0], map_location="cpu")
        discriminator = StyleGan1Discriminator(resolution=256)
    discriminator.load_state_dict(dis_state_dict)
    discriminator.eval()

    model = LGradModel(discriminator, clf)

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_wrapper = LGradEvalWrapper(model, args.output_dir, args.is_save_logits, args.is_save_embeds)
    dl = prepare_data(args)

    trainer = Trainer(inference_mode=False, logger=False)
    trainer.test(eval_wrapper, dl)


if __name__ == "__main__":
    main()
