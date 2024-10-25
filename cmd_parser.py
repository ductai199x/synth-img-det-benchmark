import argparse
from typing import List


real_dataset_choices = ["midb", "imagenet1k", "imagenet22k", "coco2017", "laion"]
synth_dataset_choices = ["synth_img", "synth_img_extra", "synthbuster"]
arch_choices = {
    "synth_img": [
        "dalle3",
        "midjourney_v6",
        "progan",
        "projected_gan",
        "stable_diffusion",
        "stable_diffusion_15",
        "stable_diffusion_3_medium",
        "stable_diffusion_xl_10",
        "stylegan",
        "stylegan2",
        "stylegan3",
        "tam_trans",
    ],
    "synth_img_extra": [
        "biggan",
        "dalle_mini",
        "dalle2",
        "eg3d",
        "gigagan",
        "guided_diffusion",
        "latent_diffusion",
        "glide",
    ],
    "synthbuster": [
        "dalle2",
        "dalle3",
        "firefly",
        "glide",
        "midjourney_v5",
        "stable_diffusion_1_3",
        "stable_diffusion_1_4",
        "stable_diffusion_2",
        "stable_diffusion_xl",
    ]
}


def list_archs(args: argparse.Namespace):
    print("\n".join(arch_choices[args.synth]))


def parse_args(parser: argparse.ArgumentParser, variant_choices: List[str]) -> argparse.Namespace:
    subparsers = parser.add_subparsers(title="commands", dest="command")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("-v", "--variant", type=str, default="default", choices=variant_choices)
    run_parser.add_argument("-r", "--real", type=str, required=True, choices=real_dataset_choices)
    run_parser.add_argument("-rn", "--num-real", type=int, required=True)
    run_parser.add_argument("-rs", "--real-split", type=str, default="test")
    run_parser.add_argument("-s", "--synth", type=str, required=True, choices=synth_dataset_choices)
    run_parser.add_argument("-a", "--arch", type=str, required=True)
    run_parser.add_argument("-sn", "--num-synth", type=int, required=True)
    run_parser.add_argument("-ss", "--synth-split", type=str, default="test")
    run_parser.add_argument("-o", "--output-dir", type=str, required=False, default=None)
    run_parser.add_argument("-w", "--num-workers", type=int, default=4)
    run_parser.add_argument("--shardshuffle", action="store_true")
    run_parser.add_argument("--no-save-logits", action="store_true")
    run_parser.add_argument("--no-save-embeds", action="store_true")

    # Add subparser to list architectures for a given synthetic dataset
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("-s", "--synth", type=str, required=True, choices=synth_dataset_choices)
    list_parser.set_defaults(func=list_archs)

    args = parser.parse_args()

    if args.command == "list":
        args.func(args)
        
    if args.command == "run":
        if args.arch not in arch_choices[args.synth]:
            raise ValueError(f"Invalid architecture {args.arch} for synthetic dataset {args.synth}")
        args.is_save_logits = not args.no_save_logits
        args.is_save_embeds = not args.no_save_embeds
    
    print(args)
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_args(parser)