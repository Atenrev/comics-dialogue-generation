import yaml
from munch import DefaultMunch


def get_configuration(yaml_path: str) -> any: 
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
        return DefaultMunch.fromDict(yaml_dict)
