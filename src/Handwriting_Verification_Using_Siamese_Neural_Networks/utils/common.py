import yaml
from pathlib import Path
from box import ConfigBox


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read a YAML configuration file."""

    with open(path_to_yaml, "r", encoding="utf-8") as file:
        content = yaml.safe_load(file)

    return ConfigBox(content)


def create_directories(path_to_directories: list):
    """Create directories if they do not exist."""

    for path in path_to_directories:
        Path(path).mkdir(parents=True, exist_ok=True)