import yaml


def get_configuration(yaml_path: str) -> any: 
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)
