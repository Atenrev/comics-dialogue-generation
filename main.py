import argparse
import torch
import importlib

from transformers import T5Tokenizer

from src.configuration import get_configuration
from src.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', type=str, default="configs/text_cloze_text_only.yaml",
                        help='YAML model config')
    parser.add_argument('--mode', type=str, default="training",
                        help='Execution mode ("training" or "inference")')
    parser.add_argument('--model', type=str, default="text_cloze_model",
                        help='Model to run')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to model checkpoint')

    args = parser.parse_args()
    return args


def main(args) -> None:
    """
    T5-small output size = 512
    """
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}.')

    # Configuration and checkpoint loading
    config = get_configuration(args.yaml_config)
    checkpoint = None

    if args.load_checkpoint is not None:
        print("Loading checkpoint.")

        try:
            checkpoint = torch.load(args.load_checkpoint, map_location=device)
        except Exception as e:
            print("ERROR: The checkpoint could not be loaded.")
            print(e)
            return

    # Tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    ModelClass = getattr(importlib.import_module(
        f"src.models.{args.model}"), config.model.classname)
    model = ModelClass(config.model).to(device)

    if args.mode == "training":
        trainer = Trainer(model, tokenizer, device, config.trainer, checkpoint)
        trainer.train(config.trainer.epochs)
    elif args.mode == "inference":
        pass


if __name__ == "__main__":
    args = parse_args()
    main(args)
