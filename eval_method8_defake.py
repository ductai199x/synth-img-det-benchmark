import torch
import os
from argparse import ArgumentParser
from lightning.pytorch import Trainer
from methods.method8_defake import Classifier, DeFakeModel, DeFakeEvalWrapper, CLIPModel, CLIPProcessor, BlipModel, BlipProcessor
from cmd_parser import parse_args
from prepare_data import prepare_data


variant_choices = ["default"]
clf_sd_path = "methods/method8_defake/weights/classifier.pth"
cache_dir = "methods/method8_defake/weights"


def main():
    parser = ArgumentParser()
    args = parse_args(parser, variant_choices)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)
    blip_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir=cache_dir)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir=cache_dir)

    clf = Classifier(input_size=1024, hidden_size_list=[512, 256], num_classes=2)
    clf_state_dict = torch.load(clf_sd_path, map_location="cpu")
    clf.load_state_dict(clf_state_dict)
    clf.eval()

    model = DeFakeModel(clf, blip_model, blip_processor, clip_model, clip_processor)
    model.eval()

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_wrapper = DeFakeEvalWrapper(model, args.output_dir, args.is_save_logits, args.is_save_embeds)
    dl = prepare_data(args)

    trainer = Trainer(enable_checkpointing=False, logger=False)
    trainer.test(eval_wrapper, dl)
    

if __name__ == "__main__":
    main()