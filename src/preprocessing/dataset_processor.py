'''
This script processes a dataset of Java files, categorizing them into
'''
import json
from pathlib import Path
from .parser_utils import parse_java_code

CATEGORIES: list[str] = [
    "plagiarized",
    "non-plagiarized",
    "original"
]

def process_plagiarized_folder(
    case_folder: str, src_folder: str, dst_root: str
) -> None:
    '''
    Process a folder of plagiarized Java files, categorizing them into different
    '''
    for ls in src_folder.iterdir():
        if ls.is_dir():
            src_folder = case_folder / "plagiarized" / ls.name
            dst_folder = dst_root / case_folder.name / "plagiarized" / ls.name
            dst_folder.mkdir(parents=True, exist_ok=True)
            for num in src_folder.iterdir():
                if num.is_dir():
                    src_folder = case_folder / "plagiarized" / ls.name / num.name
                    dst_folder = dst_root / case_folder.name / "plagiarized" / ls.name / num.name
                    dst_folder.mkdir(parents=True, exist_ok=True)
                    # Process Java files in the nested folder
                    for java_file in src_folder.glob("*.java"):
                        with open(java_file, "r", encoding="utf-8") as f:
                            code = f.read()
                        try:
                            ast = parse_java_code(code)
                            output_path = dst_folder / f"{java_file.stem}.json"
                            with open(output_path, "w", encoding="utf-8") as out_f:
                                json.dump(ast, out_f, indent=2)
                        except Exception as e:
                            print(f"Error processing {java_file}: {e}")
                            continue
                        print(f"Processed {java_file} -> {output_path}")

def process_non_plagiarized_folder(
    case_folder: str, src_folder: str, dst_root: str
) -> None:
    '''
    Process a folder of non-plagiarized Java files
    '''
    for num in src_folder.iterdir():
        if num.is_dir():
            src_folder = case_folder / "non-plagiarized" / num.name
            dst_folder = dst_root / case_folder.name / "non-plagiarized" / num.name
            dst_folder.mkdir(parents=True, exist_ok=True)
            # Process Java files in the nested folder
            for java_file in src_folder.glob("*.java"):
                with open(java_file, "r", encoding="utf-8") as f:
                    code = f.read()
                try:
                    ast = parse_java_code(code)
                    output_path = dst_folder / f"{java_file.stem}.json"
                    with open(output_path, "w", encoding="utf-8") as out_f:
                        json.dump(ast, out_f, indent=2)
                except Exception as e:
                    print(f"Error processing {java_file}: {e}")
                    continue
                print(f"Processed {java_file} -> {output_path}")

def process_original_folder(
    src_folder: str, dst_folder: str
) -> None:
    '''
    Process a folder of original Directory Java files, converting them into JSON format.
    '''
    for java_file in src_folder.glob("*.java"):
        with open(java_file, "r", encoding="utf-8") as f:
            code = f.read()
        try:
            ast = parse_java_code(code)
            output_path = dst_folder / f"{java_file.stem}.json"
            with open(output_path, "w", encoding="utf-8") as out_f:
                json.dump(ast, out_f, indent=2)
        except Exception as e:
            print(f"Error processing {java_file}: {e}")
            continue
        print(f"Processed {java_file} -> {output_path}")

def process_case_folder(src_root: str, dst_root: str):
    '''
    Process a folder of Java files, categorizing them into different
    categories and converting them into JSON format.
    Args:
        src_root (str): The source root directory containing Java files.
        dst_root (str): The destination root directory for JSON files.
    '''
    src_root: Path = Path(src_root)
    dst_root: Path = Path(dst_root)
    if not src_root.is_dir():
        raise ValueError(f"Source root {src_root} is not a directory.")
    if not dst_root.is_dir():
        raise ValueError(f"Destination root {dst_root} is not a directory.")
    for case_folder in src_root.iterdir():
        print(f"Processing case folder: {case_folder}")
        if not case_folder.is_dir():
            print(f"Skipping {case_folder}, not a directory.")
            continue
        for category in CATEGORIES:
            if not (case_folder / category).is_dir():
                print(f"Skipping {category}, not a directory.")
                continue
            src_folder = case_folder / category
            dst_folder = dst_root / case_folder.name / category
            dst_folder.mkdir(parents=True, exist_ok=True)
            print(f"Processing category: {category}")
            if category == "plagiarized":
                # Go two nested folders deep for plagiarized
                process_plagiarized_folder(case_folder, src_folder, dst_root)
            elif category == "non-plagiarized":
                # Go one nested folder deep for non-plagiarized
                process_non_plagiarized_folder(case_folder, src_folder, dst_root)
            else:
                # Process original files directly
                process_original_folder(src_folder, dst_folder)
