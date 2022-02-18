import argparse

from configuration import get_configuration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', type=str,
                        default="configs/text_cloze_text_only.yml", help='YAML model config')

    args = parser.parse_args()
    return args


def main(args) -> None:
    config = get_configuration(args.get("DATASET".get("PATH")))
    pass


if __name__ == "__main__":
    args = parse_args()
    main(args)