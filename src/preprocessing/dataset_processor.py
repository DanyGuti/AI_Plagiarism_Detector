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
from features.ast_embedding import traverse_ast, encode_features, read_ast_from_file
from .parser_utils import parse_java_code

CATEGORIES: list[str] = [
    "plagiarized",
    "non-plagiarized",
    "original"
]

def collect_java_files(src_root: Path) -> list[tuple[Path, Path]]:
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

            if category == "plagiarized":
                for folder in cat_folder.iterdir():
                    if not folder.is_dir():
                        continue
                    for version_folder in folder.iterdir():
                        if not version_folder.is_dir():
                            continue
                        for java_file in version_folder.glob("*.java"):
                            rel_path = java_file.relative_to(src_root)
                            output_path = rel_path.with_suffix(".json")
                            all_files.append((java_file, output_path))

            elif category == "non-plagiarized":
                for folder in cat_folder.iterdir():
                    if not folder.is_dir():
                        continue
                    for java_file in folder.glob("*.java"):
                        rel_path = java_file.relative_to(src_root)
                        output_path = rel_path.with_suffix(".json")
                        all_files.append((java_file, output_path))

            elif category == "original":
                for java_file in cat_folder.glob("*.java"):
                    rel_path = java_file.relative_to(src_root)
                    output_path = rel_path.with_suffix(".json")
                    all_files.append((java_file, output_path))

    return all_files

def process_all_files(src_root: str, dst_root: str):
    """
    Process all Java files from the source root and write their parsed AST as JSON in the destination.
    Shows a single global progress bar.
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    if not src_root.is_dir():
        raise ValueError(f"Source root {src_root} is not a directory.")
    if not dst_root.is_dir():
        raise ValueError(f"Destination root {dst_root} is not a directory.")

    all_files = collect_java_files(src_root)

    with tqdm(total=len(all_files), desc="Processing Java files", unit="file") as pbar:
        for java_file, rel_output in all_files:
            try:
                with open(java_file, "r", encoding="utf-8") as f:
                    code = f.read()
                ast = parse_java_code(code)

                output_path = dst_root / rel_output
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as out_f:
                    json.dump(ast, out_f, indent=2)
            except Exception as e:
                tqdm.write(f"Error processing {java_file}: {e}")
            pbar.update(1)

def create_dictionary_from_files(input_path: str) -> Tuple[Dict[str, Tuple[str, np.ndarray]], Dict[str, Tuple[str, np.ndarray]]]:
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

            if category == "plagiarized":
                for level_folder in cat_path.iterdir():
                    if not level_folder.is_dir():
                        continue
                    for version_folder in level_folder.iterdir():
                        if not version_folder.is_dir():
                            continue
                        for file in version_folder.glob("*.json"):
                            tree = read_ast_from_file(str(file))
                            if tree is None:
                                continue
                            graph = traverse_ast(tree)
                            matrix = encode_features(graph)
                            key = f"plag-{case_name}-{file.stem}"
                            matrix_dict[key] = (case_name, matrix)

            elif category == "non-plagiarized":
                for version_folder in cat_path.iterdir():
                    if not version_folder.is_dir():
                        continue
                    for file in version_folder.glob("*.json"):
                        tree = read_ast_from_file(str(file))
                        if tree is None:
                            continue
                        graph = traverse_ast(tree)
                        matrix = encode_features(graph)
                        key = f"nonplag-{case_name}-{file.stem}"
                        matrix_dict[key] = (case_name, matrix)

            elif category == "original":
                for file in cat_path.glob("*.json"):
                    tree = read_ast_from_file(str(file))
                    if tree is None:
                        continue
                    graph = traverse_ast(tree)
                    matrix = encode_features(graph)
                    key = f"orig-{case_name}-{file.stem}"
                    original_dict[key] = (case_name, matrix)

    return matrix_dict, original_dict


def write_matrix(file_path: str, matrix: np.ndarray):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for row in matrix:
            f.write(" ".join(map(str, row.tolist())) + "\n")


def create_data_split_from_dict(
    matrix_dict: Dict[str, Tuple[str, np.ndarray]],
    original_dict: Dict[str, Tuple[str, np.ndarray]],
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
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

