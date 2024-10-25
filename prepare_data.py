import webdataset as wds
from argparse import Namespace
from data import (
    CamIdWebdataset,
    ImageNet1kWebdataset,
    ImageNet22kWebdataset,
    Coco2017Webdataset,
    LaionHiResWebdataset,
    SynthImgDetWebdataset,
    SynthImgExtraWebdataset,
    SynthbusterWebdataset,
    SynthImgWebdataset,
)
from cmd_parser import arch_choices
from torchvision.transforms import Resize


def prepare_data(args: Namespace):
    if args.real == "midb":
        real_ds = CamIdWebdataset(split=args.real_split, shardshuffle=args.shardshuffle)
    elif args.real == "imagenet1k":
        real_ds = ImageNet1kWebdataset(split=args.real_split, shardshuffle=args.shardshuffle)
    elif args.real == "imagenet22k":
        real_ds = ImageNet22kWebdataset(split=args.real_split, shardshuffle=args.shardshuffle)
    elif args.real == "coco2017":
        real_ds = Coco2017Webdataset(split=args.real_split, shardshuffle=args.shardshuffle)
    elif args.real == "laion":
        real_ds = LaionHiResWebdataset(split=args.real_split, shardshuffle=args.shardshuffle)

    chosen_arch_idx = arch_choices[args.synth].index(args.arch)
    ignore_classes = [c for c in range(len(arch_choices[args.synth])) if c != chosen_arch_idx]
    if args.synth == "synth_img":
        synth_ds = SynthImgWebdataset(
            split=args.synth_split,
            ignore_classes=ignore_classes,
            shardshuffle=args.shardshuffle,
        )
    elif args.synth == "synth_img_extra":
        synth_ds = SynthImgExtraWebdataset(
            split=args.synth_split,
            ignore_classes=ignore_classes,
            shardshuffle=args.shardshuffle,
        )
    elif args.synth == "synthbuster":
        synth_ds = SynthbusterWebdataset(
            split=args.synth_split,
            ignore_classes=ignore_classes,
            shardshuffle=args.shardshuffle,
        )

    real_wl = (
        wds.WebLoader(real_ds, batch_size=1, num_workers=args.num_workers)
        .with_length(args.num_real)
        .with_epoch(args.num_real)
    )
    synth_wl = (
        wds.WebLoader(synth_ds, batch_size=1, num_workers=args.num_workers)
        .with_length(args.num_synth)
        .with_epoch(args.num_synth)
    )

    dl = SynthImgDetWebdataset(real_wl, synth_wl, is_balanced=False)
    return dl
