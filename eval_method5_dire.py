import torch
import os
from argparse import ArgumentParser
from lightning.pytorch import Trainer
from methods.method5_dire import GUIDED_DIFFUSION_WEIGHTS_NAME, GUIDED_DIFFUSION_CONFIGS, create_model_and_diffusion, dire_resnet50, DireModel, DIREEvalWrapper
from cmd_parser import parse_args
from prepare_data import prepare_data


torch.set_float32_matmul_precision("medium")
variant_choices = ["default"]


def main():
    parser = ArgumentParser()
    args = parse_args(parser, variant_choices)

    diff_sd_path = f"methods/method5_dire/weights/{GUIDED_DIFFUSION_WEIGHTS_NAME}.pth"
    diff_sd = torch.load(diff_sd_path, map_location="cpu")

    unet, diffusion = create_model_and_diffusion(**GUIDED_DIFFUSION_CONFIGS)
    unet.load_state_dict(diff_sd)
    unet.convert_to_fp16()

    clf_sd_path = "methods/method5_dire/weights/lsun_pndm.pth"
    clf_sd = torch.load(clf_sd_path, map_location="cpu")["model"]
    clf = dire_resnet50(num_classes=1)
    clf.load_state_dict(clf_sd)

    model = DireModel(unet, diffusion, clf)
    model.eval()

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_wrapper = DIREEvalWrapper(model, args.output_dir, args.is_save_logits, args.is_save_embeds)
    dl = prepare_data(args)

    trainer = Trainer(logger=False)
    trainer.test(eval_wrapper, dl)
    

if __name__ == "__main__":
    main()