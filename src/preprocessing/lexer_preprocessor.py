'''
- Dataser Processor for the Java token generator.
'''

from pathlib import Path
from typing import Dict, Tuple
import os
import random
import json
import concurrent.futures
import numpy as np
from tqdm import tqdm
from features.lstm_embedding import (
    read_tokens_from_file,
    prepare_corpus,
    train_word2vec_model,
    generate_embedding_vector,
)
from .parser_utils import parse_java_code_token_stream

CATEGORIES: list[str] = [
    "plagiarized",
    "non-plagiarized",
    "original"
]

EXCLUDED_TOKENS = {"COMMENT", "LINE_COMMENT", "BLOCK_COMMENT", "WS", "SEMI", "DOT"}


def append_java_files_to_list_lexer(
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
        output_path = rel_path.with_suffix(".txt")
        all_files.append((java_file, output_path))
    return all_files

def collect_java_files_lexer(
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
                        append_java_files_to_list_lexer(all_files, cat_folder, src_root)
                    for folder in cat_folder.iterdir():
                        if not folder.is_dir():
                            continue
                        for version_folder in folder.iterdir():
                            if not version_folder.is_dir():
                                continue
                            append_java_files_to_list_lexer(all_files, version_folder, src_root)

                case "non-plagiarized":
                    if len(list(case_folder.iterdir())) == 2:
                        append_java_files_to_list_lexer(all_files, cat_folder, src_root)
                        continue
                    for folder in cat_folder.iterdir():
                        if not folder.is_dir():
                            continue
                        append_java_files_to_list_lexer(all_files, folder, src_root)

                case "original":
                    if len(list(case_folder.iterdir())) > 2:
                        append_java_files_to_list_lexer(all_files, cat_folder, src_root)
                case _:
                    raise ValueError(f"Unknown category: {category}")
    return all_files

def _process_single_file(
    args: tuple[Path, Path, Path]
) -> None | str:
    '''
    Processes a single Java file, parses it into a token stream, and writes the\
    token stream to a txt file in the specified output path.
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
        token_stream = parse_java_code_token_stream(code)
        output_path = dst_root / rel_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out_f:
            for text, token_type in token_stream:
                if token_type not in EXCLUDED_TOKENS:
                    out_f.write(f"{text}\t{token_type}\n")
        return None
    except (IOError, ValueError) as e:
        return f"Error processing {java_file}: {e}"

def process_all_files_lexer(
    src_root: str,
    dst_root: str,
    max_workers: int = None
) -> None:
    '''
    Processes all Java files in the source root directory and saves their tokens\
        to txt files in the destination root directory.
    Args:
        src_root (str): Root path of the source files.
        dst_root (str): Root path where the output txts files will be saved.
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

    all_files = collect_java_files_lexer(src_root)
    tasks = [(java_file, rel_output, dst_root) for java_file, rel_output in all_files]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(tasks), desc="Processing Java files", unit="file") as pbar:
            for result in executor.map(_process_single_file, tasks):
                if result is not None:
                    tqdm.write(result)
                pbar.update(1)

def update_matrix_dict_lexer(
    matrix_dict: Dict[str, Tuple[str, np.ndarray, int]],
    version_folder: Path,
    case_name: str,
    key_prefix: str,
    label: int
) -> None:
    '''
    Helper function that updates the matrix dictionary with vector embeddings from txt\
    files in the specified version folder.
    Args:
        matrix_dict (dict): Dictionary to update with features.
        version_folder (Path): Path to the folder containing txt files.
        case_name (str): Name of the case to use in the key.
        key_prefix (str): Prefix for the key in the matrix dictionary.
        label (int): Label for the case (1 for plagiarized, 0 for non-plagiarized).
    Returns:
        None
    '''
    for file in version_folder.glob("*.txt"):
        tokens = read_tokens_from_file(str(file))
        corpus = prepare_corpus(tokens)
        model = train_word2vec_model(corpus)
        vector = generate_embedding_vector(tokens, model)
        key = f"{key_prefix}-{case_name}-{file.stem}"
        matrix_dict[key] = (case_name, vector, label)

def create_dictionary_from_files_lexer(
    input_path: str,
) -> Tuple[Dict[str, Tuple[str, np.ndarray, int]], Dict[str, Tuple[str, np.ndarray]]]:
    '''
    Creates a dictionary of matrices from the txt files in the input path.
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

    for case_folder in tqdm(list(input_path.iterdir()), desc="Embedding Tokens Cases"):
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
                        update_matrix_dict_lexer(matrix_dict, cat_path, case_name, "plagiarized", 1)
                    for level_folder in cat_path.iterdir():
                        if not level_folder.is_dir():
                            continue
                        for version_folder in level_folder.iterdir():
                            if not version_folder.is_dir():
                                continue
                            update_matrix_dict_lexer(
                                matrix_dict, version_folder,
                                case_name, "plagiarized", 1
                            )

                case "non-plagiarized":
                    if len(list(case_folder.iterdir())) == 2:
                        update_matrix_dict_lexer(
                            matrix_dict, cat_path,
                            case_name, "non-plagiarized", 0
                        )
                        continue
                    for version_folder in cat_path.iterdir():
                        if not version_folder.is_dir():
                            continue
                        update_matrix_dict_lexer(
                            matrix_dict, version_folder,
                            case_name, "non-plagiarized", 0
                        )

                case "original":
                    if len(list(case_folder.iterdir())) > 2:
                        for file in cat_path.glob("*.txt"):
                            tokens = read_tokens_from_file(str(file))
                            corpus = prepare_corpus(tokens)
                            model = train_word2vec_model(corpus)
                            vector = generate_embedding_vector(tokens, model)
                            key = f"original-{case_name}-{file.stem}"
                            original_dict[key] = (case_name, vector)
                case _:
                    raise ValueError(f"Unknown category: {category}")

    return matrix_dict, original_dict

def write_matrix_lexer(file_path: str, matrix: np.ndarray):
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

def create_data_split_from_dict_lexer(
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
    base_dir = Path(os.getcwd()).parent / "embed_data"
    matrix_data_dir = base_dir.parent / "matrix_data"
    label_data = {"train": {}, "validation": {}, "test": {}}

    print("Matrix data dir:", matrix_data_dir)
    print("Exists:", matrix_data_dir.exists())
    print("Contents:", list(matrix_data_dir.glob("*")))
    print("Matrix dict keys (sample):", list(matrix_dict.keys())[:10])

    for split_name in ["train", "validation", "test"]:
        split_dir = matrix_data_dir / split_name
        print(f"\n[INFO] Processing split: {split_name}")
        print(f"Looking in: {split_dir}")
        case_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        print("Found case dirs:", case_dirs)

        for case_dir in case_dirs:
            print(f"Looking in case dir: {case_dir}")
            case_name = case_dir.name
            for file_path in case_dir.glob("*.txt"):
                print(f"Found file: {file_path}")
                key = file_path.stem
                matrix_key = f"{key}"
                print(f"Checking key: {matrix_key}")
                if matrix_key in matrix_dict:
                    print(f"✔ Matched: {matrix_key}")
                    _, matrix, label = matrix_dict[matrix_key]
                    out_path = base_dir / split_name / case_name / f"{key}.txt"
                    os.makedirs(out_path.parent, exist_ok=True)
                    write_matrix_lexer(str(out_path), matrix)
                    label_data[split_name][f"{case_name}/{key}.txt"] = label
                else:
                    print(f"[WARN] Key '{matrix_key}' not found in matrix_dict")

    # Save labels per split
    for split_name, labels in label_data.items():
        label_file = base_dir / f"{split_name}_labels.json"
        with open(label_file, "w", encoding='utf-8') as f:
            json.dump(labels, f, indent=2)

    # Copy original
    for key, (case, matrix) in original_dict.items():
        file_path = base_dir / "original" / case / f"{key}.txt"
        os.makedirs(file_path.parent, exist_ok=True)
        write_matrix_lexer(str(file_path), matrix)
