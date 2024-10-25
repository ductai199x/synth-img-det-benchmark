import torch
import os
from argparse import ArgumentParser
from lightning.pytorch import Trainer
from methods.method6_zed import ZEDModel, ZEDEvalWrapper, SrecTrainWrapper
from cmd_parser import parse_args
from prepare_data import prepare_data


variant_choices = ["default", "midb", "coco2017", "imagenet1k", "imagnet22k", "laion"]
weight_dir = "methods/method6_zed/weights/version_{ds}_random_crop_128_ft_openimages_author/checkpoints/{ckptname}"
selected_feature = "dlt0"


def main():
    parser = ArgumentParser()
    args = parse_args(parser, variant_choices)

    if args.variant in ["default", "midb"]:
        ckpt_path = weight_dir.format(ds="midb", ckptname="epoch=11-val_bpsp=1.8834.ckpt")
    elif args.variant == "coco2017":
        ckpt_path = weight_dir.format(ds=args.variant, ckptname="epoch=27-val_bpsp=3.3728.ckpt")
    elif args.variant == "imagenet1k":
        ckpt_path = weight_dir.format(ds=args.variant, ckptname="epoch=22-val_bpsp=3.4713.ckpt")
    elif args.variant == "imagnet22k":
        ckpt_path = weight_dir.format(ds=args.variant, ckptname="epoch=31-val_bpsp=3.1029.ckpt")
    elif args.variant == "laion":
        ckpt_path = weight_dir.format(ds=args.variant, ckptname="epoch=19-val_bpsp=1.8512.ckpt")

    srec_wrapper = SrecTrainWrapper.load_from_checkpoint(ckpt_path, map_location="cpu")
    srec_model = srec_wrapper.model
    model = ZEDModel(srec_model, selected_feature=selected_feature)
    model.eval()

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_wrapper = ZEDEvalWrapper(model, args.output_dir, args.is_save_logits, args.is_save_embeds)
    dl = prepare_data(args)

    trainer = Trainer(logger=False)
    trainer.test(eval_wrapper, dl)
    

if __name__ == "__main__":
    main()