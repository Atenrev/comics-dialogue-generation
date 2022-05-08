import argparse
import torch
import numpy as np
import importlib
import logging

from transformers import AutoTokenizer

from src.common.registry import Registry
from src.common.configuration import get_dataset_configuration, get_model_configuration, get_trainer_configuration
from src.inference import InferenceEngine
from src.trainer import Trainer
from src.tokenizers.vlt5_tokenizers import VLT5TokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="text_cloze_image_text_vlt5",
                        help='Model to run')
    parser.add_argument('--dataset_config', type=str, default="text_cloze_image_text_vlt5_easy",
                        help='Dataset config to use')
    parser.add_argument('--trainer_config', type=str, default="default",
                        help='Trainer params to use')
    parser.add_argument('--dataset_dir', type=str, default="datasets/COMICS/",
                        help='Dataset directory path')
    parser.add_argument('--mode', type=str, default="train",
                        help='Execution mode ("training", "eval" or "inference")')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Seed to use')

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    torch.manual_seed(0)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"SELECTED DEVICE: {device}")

    # Configuration loading
    model_config = get_model_configuration(args.model)
    Registry.register("model_config", model_config)
    dataset_config = get_dataset_configuration(args.dataset_config)
    Registry.register("dataset_config", dataset_config)

    logging.info(f"SELECTED MODEL: {model_config.classname}")
    logging.info(f"SELECTED DATASET: {dataset_config.name}")

    # Dataset preprocessing
    tokenizer = None
    if model_config.tokenizer:
        if model_config.tokenizer == "vlt5":
            tokenizer = VLT5TokenizerFast.from_pretrained(
                model_config.backbone,
                max_length=model_config.max_text_length,
                do_lower_case=model_config.do_lower_case,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer)

    feature_extractor = None
    if model_config.feature_extractor:
        from transformers import BeitFeatureExtractor
        feature_extractor = BeitFeatureExtractor.from_pretrained(
            model_config.feature_extractor)

    transform = None
    if model_config.transforms:
        raise NotImplementedError("Transforms are not implemented yet.")

    dataset_kwargs = {}
    if tokenizer:
        dataset_kwargs["tokenizer"] = tokenizer
    if feature_extractor:
        dataset_kwargs["feature_extractor"] = feature_extractor
    if transform:
        dataset_kwargs["transform"] = transform

    # Model loading
    ModelClass = getattr(importlib.import_module(
        f"src.models.{args.model}"), model_config.classname)
    model = ModelClass(model_config).to(device)

    # Load model checkpoint
    checkpoint = None

    if args.load_checkpoint is not None:
        logging.info("Loading checkpoint.")

        try:
            checkpoint = torch.load(args.load_checkpoint, map_location=device)
        except Exception as e:
            logging.error("The checkpoint could not be loaded.")
            print(e)
            return

        # checkpoint["model_state_dict"] = {
        #     (k.replace("module.", "") if k.startswith("module.") else k): v
        #     for k, v in checkpoint["model_state_dict"].items()
        # }
        # model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.load_checkpoint(checkpoint["model_state_dict"])

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if args.mode != "inference":
        # Trainer specific configuration loading
        trainer_config = get_trainer_configuration(args.trainer_config)
        Registry.register("trainer_config", trainer_config)

        # DataLoaders
        create_dataloader = getattr(importlib.import_module(
            f"src.datasets.{dataset_config.name}"), "create_dataloader")
        train_dataloader, val_dataloader, test_dataloader = create_dataloader(
            args.batch_size,
            args.dataset_dir,
            device,
            dataset_config,
            dataset_kwargs=dataset_kwargs
        )

        trainer = Trainer(model, device, trainer_config, checkpoint)

        if args.mode == "train":
            trainer.train(train_dataloader, val_dataloader,
                          trainer_config.epochs)
        elif args.mode == "eval":
            assert checkpoint is not None, "ERROR: No checkpoint provided."
            trainer.eval(test_dataloader)
        else:
            raise

    elif args.mode == "inference":
        # DataLoaders
        create_dataloader = getattr(importlib.import_module(
            f"src.datasets.{dataset_config.name}"), "create_dataloader")
        dataloader, _, _ = create_dataloader(
            args.batch_size,
            args.dataset_dir,
            device,
            dataset_config,
            inference=True,
            dataset_kwargs=dataset_kwargs
        )

        inference_engine = InferenceEngine(model, device)
        inference_engine.run(dataloader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
