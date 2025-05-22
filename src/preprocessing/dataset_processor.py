'''
This module processes Java files to extract their
Abstract Syntax Trees (ASTs) and encodes them into a specific format.
'''
import json
import os
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .parser_utils import parse_java_code  # adjust the import if needed
from features.ast_embedding import traverse_ast, encode_features,\
    read_ast_from_file

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

def create_dictionary_from_files(
    input_path: str
) -> dict[str, tuple[str, np.ndarray]]:
    '''
    Create a dictionary from the json files in the input path (ast_data)
    into a dictionary with the following structure:
    {
        'plag-case-01-T1': (case-0{num}, matrix),
        ...
    }
    Arguments:
        input_path: str
            The path to the directory containing the json files.
    Returns:
        matrix_txt_dictionary: dict[str, tuple[str, np.ndarray]]
    '''
    # 1. Read each of the inner directories
    # and make the corresponding dictionary with the output
    # matrix generated
    print(input_path)
    def generate_path_name(
        case_folder: str,
        category: str,
        json_file: str,
        version_folder: str = None,
        level: str = None,
    ) -> str:
        if level is None and version_folder is not None:
            return os.path.join(
                os.getcwd(),
                "..",
                "ast_data",
                str(case_folder),
                str(category),
                str(version_folder),
                str(json_file)
            )
        if version_folder is None:
            return os.path.join(
                os.getcwd(),
                "..",
                "ast_data",
                str(case_folder),
                str(category),
                str(json_file)
            )
        return os.path.join(
            os.getcwd(),
            "..",
            "ast_data",
            str(case_folder),
            str(category),
            str(level),
            str(version_folder),
            str(json_file)
        )
    input_path = Path(input_path)
    matrix_txt_dictionary = {}
    for case_folder in input_path.iterdir():
        if not case_folder.is_dir():
            continue

        for category in CATEGORIES:
            cat_folder = case_folder / category
            if not cat_folder.is_dir():
                continue

            if category == "plagiarized":
                # L leveled name directories
                for folder in cat_folder.iterdir():
                    if not folder.is_dir():
                        continue
                    for version_folder in folder.iterdir():
                        if not version_folder.is_dir():
                            continue
                        for json_file in version_folder.glob("*.json"):
                            json_file_name = json_file.name
                            constructed_path = generate_path_name(
                                case_folder.name,
                                category,
                                json_file_name,
                                version_folder.name,
                                folder,
                            )
                            tree = read_ast_from_file(
                                constructed_path
                            )
                            curr_graph = traverse_ast(tree)
                            # Store in the corresponding dictionary key
                            matrix_txt_dictionary[\
                                "plag-" + case_folder.name + "-" + json_file_name.split(".")[0]
                            ] = (case_folder.name, encode_features(curr_graph))

            elif category == "non-plagiarized":
                for folder in cat_folder.iterdir():
                    if not folder.is_dir():
                        continue
                    for json_file in folder.glob("*.json"):
                        json_file_name = json_file.name
                        generate_path_name(
                            case_folder.name,
                            category,
                            json_file_name,
                            version_folder.name,
                        )
                        tree = read_ast_from_file(
                            constructed_path
                        )
                        curr_graph = traverse_ast(tree)
                        # Store in the corresponding dictionary key
                        matrix_txt_dictionary[\
                            "nonplag-" + case_folder.name + "-" + json_file_name.split(".")[0]
                        ] = (case_folder.name, encode_features(curr_graph))
            else:
                for json_file in cat_folder.glob("*.json"):
                    json_file_name = json_file.name
                    constructed_path = generate_path_name(
                        case_folder.name,
                        category,
                        json_file_name,
                    )
                    tree = read_ast_from_file(
                        constructed_path
                    )
                    curr_graph = traverse_ast(tree)
                    # Store in the corresponding dictionary key
                    matrix_txt_dictionary[\
                        "orig-" + case_folder.name + "-" + json_file_name.split(".")[0]
                    ] = (case_folder.name, encode_features(curr_graph))
    return matrix_txt_dictionary

def write_matrix_to_file(
    base_path: str,
    items: list[tuple[str, np.ndarray]],
    train_data: int,
    val_data: int,
) -> None:
    '''
    Write the matrix to the corresponding files
    Arguments:
        base_path: str
            The base path to the directory where the files will be written.
        items: list[tuple[str, np.ndarray]]
            The list of items to be written to the files.
        train_data: int
            The number of training data points.
        val_data: int
            The number of validation data points.
    Returns:
        None
            '''
    for idx, (key, value) in enumerate(items):
        content = value[1]
        split = ""
        if idx < train_data:
            split = "train"
        elif idx < train_data + val_data:
            split = "validation"
        else:
            split = "test"
        with open(
            os.path.join(base_path, split, key + ".txt"), "w", encoding="utf-8"
        ) as f:
            for row in content:
                f.write(" ".join(map(str, row.tolist())) + "\n")

def create_data_split_from_dict(
    matrix_txt_dictionary: dict[str, np.ndarray],
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
) -> None:
    """
    Split the dictionary into train, validation, and test sets.
    Writing the data into the corresponding directories.
    Arguments:
        matrix_txt_dictionary: dict[str, np.ndarray]
            The dictionary containing the data to be split.
        train_split: float
            The proportion of the data to include in the train split.
        val_split: float
            The proportion of the data to include in the validation split.
        test_split: float
            The proportion of the data to include in the test split.
    Returns:
        None
    """
    if train_split + val_split + test_split != 1.0:
        raise ValueError("Train, validation, and test splits must sum to 1.")

    # Iterate over the dictionary and create check on how many matrices per each case values[0]
    # are there (to split them accordingly)
    case_freqs = {}

    for key, value in matrix_txt_dictionary.items():
        case_name = value[0]
        if case_name not in case_freqs:
            case_freqs[case_name] = 0
        case_freqs[case_name] += 1
    # Split the data based on the frequency of the case (getting the total number per each case)
    # divided by the split values
    processed = {case_name: False for case_name in case_freqs}
    for key, value in matrix_txt_dictionary.items():
        case_name = value[0]
        if processed[case_name]:
            continue
        processed[case_name] = True
        items = [
            (k, v) for k, v in matrix_txt_dictionary.items() if v[0] == case_name
        ]

        random.shuffle(items)
        total = len(items)
        train_data = int(total * train_split)
        val_data = int(total * val_split)
        # Create the directories
        base_path = os.path.join(os.getcwd(), "..", "matrix_data", case_name)

        os.makedirs(os.path.join(base_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "test"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "validation"), exist_ok=True)
        # Create the train, validation and test splits as txt files
        write_matrix_to_file(
            base_path,
            items,
            train_data,
            val_data,
        )
