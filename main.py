import argparse
import torch
import importlib
import logging

from transformers import AutoTokenizer

from src.common.registry import Registry
from src.common.configuration import get_dataset_configuration, get_model_configuration, get_trainer_configuration
from src.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="text_cloze_text_only_t5",
                        help='Model to run')
    parser.add_argument('--dataset_config', type=str, default="text_cloze_text_only_easy",
                        help='Dataset config to use')
    parser.add_argument('--trainer_config', type=str, default="default",
                        help='Trainer params to use')
    parser.add_argument('--dataset_dir', type=str, default="datasets/COMICS/",
                        help='Dataset directory path')
    parser.add_argument('--mode', type=str, default="eval",
                        help='Execution mode ("training", "eval" or "inference")')
    parser.add_argument('--load_checkpoint', type=str, default="runs/9-TextClozeTextOnlyT5Model_text_cloze_text_only_74428095-e2c3-4f02-9dcb-0e5e0f8264d4/models/epoch_50.pt",
                        help='Path to model checkpoint')

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"SELECTED DEVICE: {device}")

    # Configuration and checkpoint loading
    model_config = get_model_configuration(args.model)
    dataset_config = get_dataset_configuration(args.dataset_config)
    trainer_config = get_trainer_configuration(args.trainer_config)

    # Register configs
    Registry.register("model_config", model_config)
    Registry.register("dataset_config", dataset_config)
    Registry.register("trainer_config", trainer_config)

    logging.info(f"SELECTED MODEL: {model_config.classname}")
    logging.info(f"SELECTED DATASET: {dataset_config.name}")
    checkpoint = None
    # is_parallel = False

    if args.load_checkpoint is not None:
        logging.info("Loading checkpoint.")

        try:
            checkpoint = torch.load(args.load_checkpoint, map_location=device)
            checkpoint["model_state_dict"] = {
                (k.replace("module.", "") if k.startswith("module.") else k) : v
                for k, v in checkpoint["model_state_dict"].items()
            }
            # is_parallel = True
        except Exception as e:
            logging.error("The checkpoint could not be loaded.")
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

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if args.mode == "train":
        trainer = Trainer(model, args.dataset_dir, dataset_config, tokenizer,
                          device, trainer_config, checkpoint)
        trainer.train(trainer_config.epochs)
    elif args.mode == "eval":
        assert checkpoint is not None, "ERROR: No checkpoint provided."
        trainer = Trainer(model, args.dataset_dir, dataset_config, tokenizer,
                          device, trainer_config, checkpoint)
        trainer.eval()
    elif args.mode == "inference":
        assert checkpoint is not None, "ERROR: No checkpoint provided."
        raise NotImplementedError


if __name__ == "__main__":
    args = parse_args()
    main(args)
