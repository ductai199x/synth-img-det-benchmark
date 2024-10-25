import torch
import os
from argparse import ArgumentParser
from lightning.pytorch import Trainer
from methods.method1_cnndet import resnet50, CnnDetEvalWrapper
from cmd_parser import parse_args
from prepare_data import prepare_data


variant_choices = ["default", "blur_jpg_prob0.1", "blur_jpb_prob0.5"]


def main():
    parser = ArgumentParser()
    args = parse_args(parser, variant_choices)

    if args.variant in ["default", "blur_jpg_prob0.1"]:
        ckpt_path = "methods/method1_cnndet/weights/blur_jpg_prob0.1.pth"
    elif args.variant == "blur_jpg_prob0.5":
        ckpt_path = "methods/method1_cnndet/weights/blur_jpg_prob0.5.pth"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"]
    model = resnet50(num_classes=1)
    model.load_state_dict(state_dict)
    model.eval()

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_wrapper = CnnDetEvalWrapper(model, args.output_dir, args.is_save_logits, args.is_save_embeds)
    dl = prepare_data(args)

    trainer = Trainer(logger=False)
    trainer.test(eval_wrapper, dl)
    

if __name__ == "__main__":
    main()