import argparse
import torch
import importlib

from transformers import AutoTokenizer

from src.common.configuration import get_dataset_configuration, get_model_configuration, get_trainer_configuration
from src.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="text_cloze_text_only_t5_hierarchy",
                        help='Model to run')
    parser.add_argument('--dataset', type=str, default="text_cloze_text_only_easy",
                        help='Dataset to use')
    parser.add_argument('--trainer', type=str, default="default",
                        help='Trainer params to use') 
    parser.add_argument('--mode', type=str, default="train",
                        help='Execution mode ("training" or "inference")')
    parser.add_argument('--load_checkpoint', type=str, default=None,  # "models/t5smallpo256posemb04dropout_shufflecandidates_6577acc.pt",
                        help='Path to model checkpoint')

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'INFO: SELECTED DEVICE: {device}.')

    # Configuration and checkpoint loading
    model_config = get_model_configuration(args.model)
    dataset_config = get_dataset_configuration(args.dataset)
    trainer_config = get_trainer_configuration(args.trainer)
    print("INFO: SELECTED MODEL:", model_config.classname)
    print("INFO: SELECTED DATASET:", dataset_config.name)
    checkpoint = None

    if args.load_checkpoint is not None:
        print("INFO: Loading checkpoint.")

        try:
            checkpoint = torch.load(args.load_checkpoint, map_location=device)
        except Exception as e:
            print("ERROR: The checkpoint could not be loaded.")
            print(e)
            return

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer)
    ModelClass = getattr(importlib.import_module(
        f"src.models.{args.model}"), model_config.classname)
    model = ModelClass(model_config).to(device)

    # Load model checkpoint
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

    if args.mode == "train":
        trainer = Trainer(model, dataset_config, tokenizer,
                          device, trainer_config, checkpoint)
        trainer.train(trainer_config.epochs)
    elif args.mode == "eval":
        assert checkpoint is not None
        trainer = Trainer(model, dataset_config, tokenizer,
                          device, trainer_config, checkpoint)
        trainer.eval()
    elif args.mode == "inference":
        assert checkpoint is not None
        raise NotImplementedError


if __name__ == "__main__":
    args = parse_args()
    main(args)
