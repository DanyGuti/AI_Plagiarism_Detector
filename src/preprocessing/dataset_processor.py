'''
- Dataser Processor for the Java AST dataset.
'''

from pathlib import Path
from typing import Dict, Tuple
import os
import random
import json
import concurrent.futures
import numpy as np
from tqdm import tqdm
from features.ast_encoding import traverse_ast, encode_features, read_ast_from_file
from .parser_utils import parse_java_code

CATEGORIES: list[str] = [
    "plagiarized",
    "non-plagiarized",
    "original"
]

def append_java_files_to_list(
    all_files: list[Tuple[Path, Path]],
    cat_folder: Path,
    src_root: Path
) -> list[Tuple[Path, Path]]:
    '''
    Appends Java files from a category folder to the list of all files.
    Args:
        all_files (list): List to append the Java files to.
        cat_folder (Path): Path to the category folder containing Java files.
        src_root (Path): Root path of the source files.
    Returns:
        list: Updated list of all files with Java files and their output paths.
    '''
    for java_file in cat_folder.glob("*.java"):
        rel_path = java_file.relative_to(src_root)
        output_path = rel_path.with_suffix(".json")
        all_files.append((java_file, output_path))
    return all_files

def collect_java_files(
    src_root: Path,
) -> list[tuple[Path, Path]]:
    '''
    Collects all Java files from the source root directory and its subdirectories.
    Args:
        src_root (Path): Root path of the source files.
    Returns:
        list: List of tuples containing Java file paths and their corresponding output paths.
    '''
    all_files = []

    for case_folder in src_root.iterdir():
        if not case_folder.is_dir():
            continue

        for category in CATEGORIES:
            cat_folder = case_folder / category
            if not cat_folder.is_dir():
                continue

            match category:
                case "plagiarized":
                    if len(list(case_folder.iterdir())) == 2:
                        append_java_files_to_list(all_files, cat_folder, src_root)
                    for folder in cat_folder.iterdir():
                        if not folder.is_dir():
                            continue
                        for version_folder in folder.iterdir():
                            if not version_folder.is_dir():
                                continue
                            append_java_files_to_list(all_files, version_folder, src_root)

                case "non-plagiarized":
                    if len(list(case_folder.iterdir())) == 2:
                        append_java_files_to_list(all_files, cat_folder, src_root)
                        continue
                    for folder in cat_folder.iterdir():
                        if not folder.is_dir():
                            continue
                        append_java_files_to_list(all_files, folder, src_root)

                case "original":
                    if len(list(case_folder.iterdir())) > 2:
                        append_java_files_to_list(all_files, cat_folder, src_root)
                case _:
                    raise ValueError(f"Unknown category: {category}")
    return all_files

def _process_single_file(
    args: tuple[Path, Path, Path]
) -> None | str:
    '''
    Processes a single Java file, parses it, and saves the AST to a JSON file.
    Args:
        args (tuple): A tuple containing the Java file path, relative output path,\
        and destination root.
    Returns:
        None or str: Returns None if successful, or an error message if an exception occurs.
    '''
    java_file, rel_output, dst_root = args
    try:
        with open(java_file, "r", encoding="utf-8") as f:
            code = f.read()
        ast = parse_java_code(code)
        output_path = dst_root / rel_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(ast, out_f, indent=2)
        return None
    except (IOError, ValueError) as e:
        return f"Error processing {java_file}: {e}"

def process_all_files(
    src_root: str,
    dst_root: str,
    max_workers: int = None
) -> None:
    '''
    Processes all Java files in the source root directory and saves their ASTs\
        to JSON files in the destination root directory.
    Args:
        src_root (str): Root path of the source files.
        dst_root (str): Root path where the output JSON files will be saved.
        max_workers (int): Maximum number of worker threads to use for processing.
    Returns:
        None
    '''
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    if not src_root.is_dir():
        raise ValueError(f"Source root {src_root} is not a directory.")
    if not dst_root.is_dir():
        raise ValueError(f"Destination root {dst_root} is not a directory.")

    all_files = collect_java_files(src_root)
    tasks = [(java_file, rel_output, dst_root) for java_file, rel_output in all_files]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(tasks), desc="Processing Java files", unit="file") as pbar:
            for result in executor.map(_process_single_file, tasks):
                if result is not None:
                    tqdm.write(result)
                pbar.update(1)

def update_matrix_dict(
    matrix_dict: Dict[str, Tuple[str, np.ndarray, int]],
    version_folder: Path,
    case_name: str,
    key_prefix: str,
    label: int
) -> None:
    '''
    Helper function that updates the matrix dictionary with features extracted from JSON\
    files in the specified version folder.
    Args:
        matrix_dict (dict): Dictionary to update with features.
        version_folder (Path): Path to the folder containing JSON files.
        case_name (str): Name of the case to use in the key.
        key_prefix (str): Prefix for the key in the matrix dictionary.
        label (int): Label for the case (1 for plagiarized, 0 for non-plagiarized).
    Returns:
        None
    '''
    for file in version_folder.glob("*.json"):
        tree = read_ast_from_file(str(file))
        if tree is None:
            continue
        graph = traverse_ast(tree)
        matrix = encode_features(graph)
        key = f"{key_prefix}-{case_name}-{file.stem}"
        matrix_dict[key] = (case_name, matrix, label)

def create_dictionary_from_files(
    input_path: str,
) -> Tuple[Dict[str, Tuple[str, np.ndarray, int]], Dict[str, Tuple[str, np.ndarray]]]:
    '''
    Creates a dictionary of matrices from the JSON files in the input path.
    Args:
        input_path (str): Path to the directory containing case folders.
    Returns:
        tuple: A tuple containing two dictionaries:
            - matrix_dict: Dictionary with keys as case names and values as tuples of\
                (case name, matrix, label).
            - original_dict: Dictionary with keys as original case names and values as\
                tuples of (case name, matrix).
    '''
    input_path = Path(input_path)
    matrix_dict = {}
    original_dict = {}

    for case_folder in tqdm(list(input_path.iterdir()), desc="Parsing Cases"):
        if not case_folder.is_dir():
            continue
        case_name = case_folder.name

        for category in CATEGORIES:
            cat_path = case_folder / category
            if not cat_path.exists():
                continue

            match category:
                case "plagiarized":
                    if len(list(case_folder.iterdir())) == 2:
                        update_matrix_dict(matrix_dict, cat_path, case_name, "plagiarized", 1)
                    for level_folder in cat_path.iterdir():
                        if not level_folder.is_dir():
                            continue
                        for version_folder in level_folder.iterdir():
                            if not version_folder.is_dir():
                                continue
                            update_matrix_dict(
                                matrix_dict, version_folder,
                                case_name, "plagiarized", 1
                            )

                case "non-plagiarized":
                    if len(list(case_folder.iterdir())) == 2:
                        update_matrix_dict(matrix_dict, cat_path, case_name, "non-plagiarized", 0)
                        continue
                    for version_folder in cat_path.iterdir():
                        if not version_folder.is_dir():
                            continue
                        update_matrix_dict(
                            matrix_dict, version_folder,
                            case_name, "non-plagiarized", 0
                        )

                case "original":
                    if len(list(case_folder.iterdir())) > 2:
                        for file in cat_path.glob("*.json"):
                            tree = read_ast_from_file(str(file))
                            if tree is None:
                                continue
                            graph = traverse_ast(tree)
                            matrix = encode_features(graph)
                            key = f"original-{case_name}-{file.stem}"
                            original_dict[key] = (case_name, matrix)
                case _:
                    raise ValueError(f"Unknown category: {category}")

    return matrix_dict, original_dict

def write_matrix(file_path: str, matrix: np.ndarray):
    '''
    Writes a matrix to a text file.
    Args:
        file_path (str): Path to the output text file.
        matrix (np.ndarray): Matrix to write to the file.
    Returns:
        None
    '''
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for row in matrix:
            f.write(" ".join(map(str, row.tolist())) + "\n")

def create_data_split_from_dict(
    matrix_dict: Dict[str, Tuple[str, np.ndarray, int]],
    original_dict: Dict[str, Tuple[str, np.ndarray]],
    train_split: float = 0.6,
    val_split: float = 0.2,
    test_split: float = 0.2,
):
    '''
    Splits the data into training, validation, and test sets based on\
        the provided splits.
    Args:
        matrix_dict (dict): Dictionary containing matrices with keys as case names.
        original_dict (dict): Dictionary containing original matrices with keys as case names.
        train_split (float): Proportion of data to use for training.
        val_split (float): Proportion of data to use for validation.
        test_split (float): Proportion of data to use for testing.
    Returns:
        None
    '''
    if not np.isclose(train_split + val_split + test_split, 1.0):
        raise ValueError("Splits must sum to 1.0")

    base_dir = Path(os.getcwd()).parent / "matrix_data"
    label_data = {"train": {}, "validation": {}, "test": {}}

    grouped: Dict[str, list[Tuple[str, np.ndarray, int]]] = {}
    for key, (case, matrix, label) in matrix_dict.items():
        grouped.setdefault(case, []).append((key, matrix, label))

    for case, items in grouped.items():
        if not items:
            continue
        random.shuffle(items)
        total = len(items)
        train_end = int(total * train_split)
        val_end = train_end + int(total * val_split)

        splits = {
            "train": items[:train_end],
            "validation": items[train_end:val_end],
            "test": items[val_end:]
        }

        for split_name, samples in splits.items():
            for key, matrix, label in samples:
                file_path = base_dir / split_name / case / f"{key}.txt"
                write_matrix(str(file_path), matrix)
                label_data[split_name][f"{case}/{key}.txt"] = label

    for split_name, labels in label_data.items():
        label_file = base_dir / f"{split_name}_labels.json"
        with open(label_file, "w", encoding='utf-8') as f:
            json.dump(labels, f, indent=2)

    for key, (case, matrix) in original_dict.items():
        file_path = base_dir / "original" / case / f"{key}.txt"
        write_matrix(str(file_path), matrix)
