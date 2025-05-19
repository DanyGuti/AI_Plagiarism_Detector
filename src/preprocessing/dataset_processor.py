import json
from pathlib import Path
from tqdm import tqdm
from .parser_utils import parse_java_code  # adjust the import if needed

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
