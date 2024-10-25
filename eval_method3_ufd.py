import torch
import os
from argparse import ArgumentParser
from lightning.pytorch import Trainer
from methods.method3_ufd import CLIPModel, UFDModel, UFDEvalWrapper
from cmd_parser import parse_args
from prepare_data import prepare_data


variant_choices = ["default"]
clip_model_name = "openai/clip-vit-large-patch14"
clip_model_path = "methods/method3_ufd/weights"
fc_sd_path = "methods/method3_ufd/weights/fc_weights.pth"


def main():
    parser = ArgumentParser()
    args = parse_args(parser, variant_choices)

    clip_model = CLIPModel.from_pretrained(clip_model_name, cache_dir=clip_model_path)
    fc = torch.nn.Linear(clip_model.projection_dim, 1)
    
    fc_state_dict = torch.load(fc_sd_path, map_location="cpu")
    fc.load_state_dict(fc_state_dict)

    model = UFDModel(clip_model, fc)
    model.eval()

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_wrapper = UFDEvalWrapper(model, args.output_dir, args.is_save_logits, args.is_save_embeds)
    dl = prepare_data(args)

    trainer = Trainer(logger=False)
    trainer.test(eval_wrapper, dl)
    

if __name__ == "__main__":
    main()