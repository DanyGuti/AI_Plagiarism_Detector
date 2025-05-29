'''
Module to make pre-processing of the code files
'''

from pathlib import Path
from typing import Dict, Tuple
import os
import random
import json
import numpy as np
from tqdm import tqdm
from features.ast_encoding import traverse_ast, encode_features, read_ast_from_file
from .parser_utils import parse_java_code
import concurrent.futures

CATEGORIES: list[str] = [
    "plagiarized",
    "non-plagiarized",
    "original"
]

def append_java_files_to_list(
    all_files: list[tuple[Path, Path]],
    cat_folder: Path,
    src_root: Path
) -> list[tuple[Path, Path]]:
    '''
    Appends Java files from a category folder to the all_files list.
    Args:
        all_files: List to append the tuples of\
            (input_java_path, output_json_path_relative_to_src_root).
        cat_folder: Path to the category folder containing Java files.
        src_root: Root path of the source directory, used to compute relative paths.
    Returns:
        all_files:
        Updated list of tuples with Java files and their corresponding output paths.
        '''
    for java_file in cat_folder.glob("*.java"):
        rel_path = java_file.relative_to(src_root)
        output_path = rel_path.with_suffix(".json")
        all_files.append((java_file, output_path))
    return all_files

def collect_java_files(
    src_root: Path,
    has_nested_dirs: bool = True
) -> list[tuple[Path, Path]]:
    """
    Traverse all case folders and collect Java files with their target output paths.
    Returns:
        A list of tuples: (input_java_path, output_json_path_relative_to_src_root)
    """
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
                        append_java_files_to_list(
                            all_files,
                            cat_folder,
                            src_root
                        )
                    for folder in cat_folder.iterdir():
                        if not folder.is_dir():
                            continue
                        for version_folder in folder.iterdir():
                            if not version_folder.is_dir():
                                continue
                            append_java_files_to_list(
                                all_files,
                                version_folder,
                                src_root
                            )
                case "non-plagiarized":
                    if len(list(case_folder.iterdir())) == 2:
                        append_java_files_to_list(
                            all_files,
                            cat_folder,
                            src_root
                        )
                        continue
                    for folder in cat_folder.iterdir():
                        if not folder.is_dir():
                            continue
                        append_java_files_to_list(
                            all_files,
                            folder,
                            src_root
                        )
                case "original":
                    if len(list(case_folder.iterdir())) > 2:
                        append_java_files_to_list(
                            all_files,
                            cat_folder,
                            src_root
                        )
                case _:
                    raise ValueError(f"Unknown category: {category}")
    return all_files

def _process_single_file(args):
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
    except Exception as e:
        return f"Error processing {java_file}: {e}"


def process_all_files(src_root: str, dst_root: str, max_workers: int = None):
    """
    Process all Java files from the source root and write their parsed
    AST as JSON in the destination, in parallel.
    Shows a single global progress bar.
    """
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
    matrix_dict: Dict[str, Tuple[str, np.ndarray]],
    version_folder: Path,
    case_name: str,
    key_prefix: str
) -> None:
    '''
    Updates the matrix_dict with matrices from the given version folder.
    Args:
        matrix_dict: Dictionary to update with new matrices.
        version_folder: Path to the folder containing JSON files.
        case_name: Name of the case to use in the keys.
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
        matrix_dict[key] = (case_name, matrix)

def create_dictionary_from_files(
    input_path: str,
    has_nested_dirs: bool = True
) -> Tuple[Dict[str, Tuple[str, np.ndarray]], Dict[str, Tuple[str, np.ndarray]]]:
    """
    Parses AST JSON files into a dictionary of encoded matrices.
    Returns:
        (main_dict, original_dict)
        - main_dict: samples for train/val/test
        - original_dict: only the original files
    """
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
                        update_matrix_dict(
                            matrix_dict,
                            cat_path,
                            case_name,
                            key_prefix="plagiarized"
                        )

                    for level_folder in cat_path.iterdir():
                        if not level_folder.is_dir():
                            continue
                        for version_folder in level_folder.iterdir():
                            if not version_folder.is_dir():
                                continue
                            update_matrix_dict(
                                matrix_dict,
                                version_folder,
                                case_name,
                                key_prefix="plagiarized"
                            )

                case "non-plagiarized":
                    if len(list(case_folder.iterdir())) == 2:
                        update_matrix_dict(
                            matrix_dict,
                            cat_path,
                            case_name,
                            key_prefix="non-plagiarized"
                        )
                        continue
                    for version_folder in cat_path.iterdir():
                        if not version_folder.is_dir():
                            continue
                        update_matrix_dict(
                            matrix_dict,
                            version_folder,
                            case_name,
                            key_prefix="non-plagiarized"
                        )

                case "original":
                    if len(list(case_folder.iterdir())) > 2:
                        update_matrix_dict(
                            original_dict,
                            cat_path,
                            case_name,
                            key_prefix="original"
                        )
                case _:
                    raise ValueError(f"Unknown category: {category}")

    return matrix_dict, original_dict


def write_matrix(file_path: str, matrix: np.ndarray):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for row in matrix:
            f.write(" ".join(map(str, row.tolist())) + "\n")


def create_data_split_from_dict(
    matrix_dict: Dict[str, Tuple[str, np.ndarray]],
    original_dict: Dict[str, Tuple[str, np.ndarray]],
    train_split: float = 0.6,
    val_split: float = 0.2,
    test_split: float = 0.2,
):
    """
    Splits main_dict into train/val/test and writes them to matrix_data/{split}/{case}/.
    Writes original_dict to matrix_data/original/{case}/.
    """
    if not np.isclose(train_split + val_split + test_split, 1.0):
        raise ValueError("Splits must sum to 1.0")

    base_dir = Path(os.getcwd()).parent / "matrix_data"

    # Group main dict by case
    grouped: Dict[str, list[Tuple[str, np.ndarray]]] = {}
    for key, (case, matrix) in matrix_dict.items():
        grouped.setdefault(case, []).append((key, matrix))

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
            for key, matrix in samples:
                file_path = base_dir / split_name / case / f"{key}.txt"
                write_matrix(str(file_path), matrix)

    # Write original matrices
    for key, (case, matrix) in original_dict.items():
        file_path = base_dir / "original" / case / f"{key}.txt"
        write_matrix(str(file_path), matrix)
