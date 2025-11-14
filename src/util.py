import string
from pathlib import Path

"""
Helper functions for the project.
"""

def _get_project_root(marker=".git") -> Path:
    """
    Finds the root of the project by searching for a file at the root.
    :param marker: The file that exists at the root of the project.
    :return: The path of the root.
    """
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Marker: {marker} not found")

def get_path(relative_path: string) -> Path:
    """
    Converts a relative path to a Path object.
    The purpose of this function is to avoid issues arising from scripts being run from different directories.
    Usage: filepath = get_path("data/file.txt")
    :param relative_path: The relative path to the desired file or directory (starting from root directory).
    :return: The corresponding Path object.
    """
    project_root = _get_project_root()
    return project_root.joinpath(relative_path)