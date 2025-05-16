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
    for case_folder in src_root.iterdir():
        if not case_folder.is_dir():
            continue
        for category in CATEGORIES:
            src_folder = case_folder / category
            dst_folder = dst_root / case_folder.name / category
            dst_folder.mkdir(parents=True, exist_ok=True)

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
