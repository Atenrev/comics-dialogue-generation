"""
Some functions from https://github.com/ArjanCodes/2021-data-science-refactor/blob/main/after/ds/utils.py
"""
import json
import pathlib
import uuid

from typing import List


def create_experiment_dir(root: str, experiment_uuid: str = "", parents: bool = True) -> str:
    root_path = pathlib.Path(root).resolve()
    child = (
        create_from_missing(root_path, experiment_uuid)
        if not root_path.exists()
        else create_from_existing(root_path, experiment_uuid)
    )
    child.mkdir(parents=parents)
    return child.as_posix()


def create_from_missing(root: pathlib.Path, experiment_uuid: str = "") -> pathlib.Path:
    return root / f"0-{experiment_uuid}"


def create_from_existing(root: pathlib.Path, experiment_uuid: str = "") -> pathlib.Path:
    children = [
        int(c.name.split("-")[0]) for c in root.glob("*")
        if (c.is_dir() and c.name.split("-")[0].isnumeric())
    ]
    if is_first_experiment(children):
        child = create_from_missing(root, experiment_uuid)
    else:
        child = root / \
            f"{increment_experiment_number(children)}-{experiment_uuid}"
    return child


def is_first_experiment(children: List[int]) -> bool:
    return len(children) == 0


def increment_experiment_number(children: List[int]) -> str:
    return str(len(children) + 1)


def add_new_experiment(new_data, filename='experiments.json'):
    root_path = pathlib.Path(filename).resolve()

    if not root_path.exists():
        with open(filename,'w') as file:
            json.dump({"experiments": []}, file)

    with open(filename,'r+') as file:
        file_data = json.load(file)
        file_data["experiments"].append(new_data)
        file.seek(0)
        json.dump(file_data, file, indent = 4)


def generate_experiment_uuid(trainer_config, dataset_config, model_config):
    experiment_uuid = f"{model_config.classname}_{dataset_config.name}_{str(uuid.uuid4())}"
    config = {
        "name": experiment_uuid,
        "trainer": trainer_config,
        "dataset": dataset_config,
        "model": model_config
    }
    
    add_new_experiment(config)

    return experiment_uuid
