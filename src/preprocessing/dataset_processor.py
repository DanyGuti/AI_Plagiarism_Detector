import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm
from features.ast_embedding import traverse_ast, encode_features, read_ast_from_file

CATEGORIES = ["plagiarized", "non-plagiarized", "original"]

def create_dictionary_from_files(input_path: str) -> Dict[str, Tuple[str, np.ndarray]]:
    """
    Parses AST JSON files from the specified directory into a dictionary of matrices.
    Returns:
        A dictionary {sample_key: (case_name, encoded_matrix)}
    """
    input_path = Path(input_path)
    matrix_dict = {}

    for case_folder in input_path.iterdir():
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
                            key = f"plag-{case_name}-{file.stem}"
                            tree = read_ast_from_file(str(file))
                            graph = traverse_ast(tree)
                            matrix = encode_features(graph)
                            matrix_dict[key] = (case_name, matrix)

            elif category == "non-plagiarized":
                for version_folder in cat_path.iterdir():
                    if not version_folder.is_dir():
                        continue
                    for file in version_folder.glob("*.json"):
                        key = f"nonplag-{case_name}-{file.stem}"
                        tree = read_ast_from_file(str(file))
                        graph = traverse_ast(tree)
                        matrix = encode_features(graph)
                        matrix_dict[key] = (case_name, matrix)

            elif category == "original":
                for file in cat_path.glob("*.json"):
                    key = f"orig-{case_name}-{file.stem}"
                    tree = read_ast_from_file(str(file))
                    graph = traverse_ast(tree)
                    matrix = encode_features(graph)
                    matrix_dict[key] = (case_name, matrix)

    return matrix_dict

def write_matrix(file_path: str, matrix: np.ndarray):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for row in matrix:
            f.write(" ".join(map(str, row.tolist())) + "\n")

def create_data_split_from_dict(
    matrix_dict: Dict[str, Tuple[str, np.ndarray]],
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
):
    """
    Splits matrices into train/val/test and writes them to matrix_data/{split}/{case}/ files.
    """
    if not np.isclose(train_split + val_split + test_split, 1.0):
        raise ValueError("Splits must sum to 1.0")

    # Group by case
    grouped: Dict[str, list[Tuple[str, np.ndarray]]] = {}
    for key, (case, matrix) in matrix_dict.items():
        grouped.setdefault(case, []).append((key, matrix))

    base_dir = Path(os.getcwd()).parent / "matrix_data"
    for case, items in grouped.items():
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
