import argparse
import torch

from transformers import T5Tokenizer

from configuration import get_configuration
from models.text_cloze_model import TextClozeTextOnlyModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', type=str,
                        default="configs/text_cloze_text_only.yaml", help='YAML model config')

    args = parser.parse_args()
    return args


def main(args) -> None:
    """
    T5-small output size = 512
    """
    config = get_configuration(args.yaml_config)
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = TextClozeTextOnlyModel(config.model)

    import torch.nn.functional as F
    input_ids = tokenizer("Hello", return_tensors="pt").input_ids
    answers_ids = tokenizer(["World War", "Cold"], return_tensors="pt", max_length=config.model.answer_max_tokens, padding="max_length").input_ids
    targets = torch.tensor([[1.0, 0.0]])
    outputs = model(input_ids, answers_ids, targets)
    pass


if __name__ == "__main__":
    args = parse_args()
    main(args)
