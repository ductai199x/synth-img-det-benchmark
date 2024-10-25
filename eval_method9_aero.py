import torch
import os
from argparse import ArgumentParser
from lightning.pytorch import Trainer
from methods.method9_aero import get_autoenc, AerobladeModel, AerobladeEvalWrapper
from cmd_parser import parse_args
from prepare_data import prepare_data


variant_choices = ["default"]
cache_dir = "methods/method9_aero/weights"
repo_ids = [
    "CompVis/stable-diffusion-v1-1",  # SD1
    "stabilityai/stable-diffusion-2-base",  # SD2
    "kandinsky-community/kandinsky-2-1",  # KD2.1
]


def main():
    parser = ArgumentParser()
    args = parse_args(parser, variant_choices)

    autoencs, decode_dtypes = zip(*[get_autoenc(repo_id, cache_dir) for repo_id in repo_ids])
    model = AerobladeModel(autoencs, decode_dtypes)

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_wrapper = AerobladeEvalWrapper(model, args.output_dir, args.is_save_logits, args.is_save_embeds)
    dl = prepare_data(args)

    trainer = Trainer(enable_checkpointing=False, logger=False)
    trainer.test(eval_wrapper, dl)
    

if __name__ == "__main__":
    main()