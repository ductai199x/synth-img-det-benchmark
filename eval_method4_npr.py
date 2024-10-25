import torch
import os
from argparse import ArgumentParser
from lightning.pytorch import Trainer
from methods.method4_npr import npr_resnet50, DummyWrapper, NPREvalWrapper
from cmd_parser import parse_args
from prepare_data import prepare_data


variant_choices = ["default", "default_ckpt", "ckpt2", "ckpt3"]


def main():
    parser = ArgumentParser()
    args = parse_args(parser, variant_choices)

    if args.variant in ["default", "default_ckpt"]:
        ckpt_path = "methods/method4_npr/weights/NPR.pth"
    elif args.variant == "ckpt2":
        ckpt_path = "methods/method4_npr/weights/model_epoch_last_3090.pth"
    elif args.variant == "ckpt3":
        ckpt_path = "methods/method4_npr/weights/NPR_GenImage_sdv4.pth"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    state_dict_new = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")
        state_dict_new[new_k] = v

    model = npr_resnet50(num_classes=1)
    model.load_state_dict(state_dict_new)
    model.eval()

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_wrapper = NPREvalWrapper(model, args.output_dir, args.is_save_logits, args.is_save_embeds)
    dl = prepare_data(args)

    trainer = Trainer(logger=False)
    trainer.test(eval_wrapper, dl)
    

if __name__ == "__main__":
    main()