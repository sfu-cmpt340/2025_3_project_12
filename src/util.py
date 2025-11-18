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

def _get_correct_case_path(relative_path: str, project_root: Path) -> Path:
    """
    Helper function to handle case mismatches between meta.csv and actual data folders.
    :param relative_path: The path with possible incorrect casing.
    :param project_root: The root of the project as a Path object.
    :return: The actual Path object with correct casing.
    """
    path_parts = Path(relative_path).parts
    current = project_root
    for part in path_parts:
        lowercase_path_to_correct_case_path_dict = {p.name.lower(): p for p in current.iterdir()}
        if part.lower() not in lowercase_path_to_correct_case_path_dict:
            raise FileNotFoundError(f"{part} not found in {current}")

        # append correct case
        current = lowercase_path_to_correct_case_path_dict[part.lower()]

    return current

def get_path(relative_path: str) -> Path:
    """
    Converts a relative path to a Path object.
    The purpose of this function is to avoid issues arising from scripts being run from different directories.
    Usage: filepath = get_path("data/file.txt")
    :param relative_path: The relative path to the desired file or directory (starting from root directory). Case-insensitive.
    :return: The corresponding Path object.
    """
    project_root = _get_project_root()
    return _get_correct_case_path(relative_path, project_root)
