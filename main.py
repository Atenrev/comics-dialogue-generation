import argparse
import torch
import importlib

from transformers import T5Tokenizer, RobertaTokenizer

from src.configuration import get_configuration
from src.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', type=str, default="configs/text_cloze_text_only_t5.yaml",
                        help='YAML model config')
    parser.add_argument('--mode', type=str, default="train",
                        help='Execution mode ("training" or "inference")')
    parser.add_argument('--model', type=str, default="text_cloze_model_t5",
                        help='Model to run')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to model checkpoint')

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    """
    T5-small output size = 512
    """
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'SELECTED DEVICE: {device}.')

    # Configuration and checkpoint loading
    config = get_configuration(args.yaml_config)
    print("SELECTED MODEL:", config.model.classname)
    print("SELECTED DATASET:", config.trainer.dataset.name)
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
    # TODO: Load tokenizer dynamically
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    # tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    ModelClass = getattr(importlib.import_module(
        f"src.models.{args.model}"), config.model.classname)
    model = ModelClass(config.model).to(device)

    if args.mode == "train":
        trainer = Trainer(model, tokenizer, device, config.trainer, checkpoint)
        trainer.train(config.trainer.epochs)
    elif args.mode == "eval":
        trainer = Trainer(model, tokenizer, device, config.trainer, checkpoint)
        trainer.eval()
    elif args.mode == "inference":
        pass


if __name__ == "__main__":
    args = parse_args()
    main(args)
