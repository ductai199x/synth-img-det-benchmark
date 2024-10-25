import torch
import os
from argparse import ArgumentParser
from lightning.pytorch import Trainer
from methods.method7_patchfor import make_patch_xceptionnet, PatchForModel, PatchForEvalWrapper
from cmd_parser import parse_args
from prepare_data import prepare_data


variant_choices = ["default", "block1", "block2", "block3", "block4", "block5"]
weight_path = "methods/method7_patchfor/weights/checkpoints/xception_{block}/bestval_net_D.pth"


def main():
    parser = ArgumentParser()
    args = parse_args(parser, variant_choices)

    args.variant = "block2" if args.variant == "default" else args.variant
    ckpt_path = weight_path.format(block=args.variant)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    patch_clf = make_patch_xceptionnet(args.variant)
    patch_clf.load_state_dict(state_dict)

    model = PatchForModel(patch_clf)
    model.eval()

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_wrapper = PatchForEvalWrapper(model, args.output_dir, args.is_save_logits, args.is_save_embeds)
    dl = prepare_data(args)

    trainer = Trainer(enable_checkpointing=False, logger=False)
    trainer.test(eval_wrapper, dl)
    

if __name__ == "__main__":
    main()