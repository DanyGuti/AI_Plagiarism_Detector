'''
Module to handle data augmentation code scrapping tasks.
'''
from pathlib import Path
import os
import pandas as pd

def get_max_directory_files_count(
    src_path: str
) -> int:
    '''
    Get the maximum number of directories in the given source path.
    Args:
        src_path (str): The source path to check for directories.
    Returns:
        int: The maximum number of directories found in the source path.
        '''
    return len(
        [name for name in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, name))]
    )

def get_max_problem_column(
    df: pd.DataFrame
) -> int:
    '''
    Get the maximum value in the 'problem' column of a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing the 'problem' column.
    Returns:
        int: The maximum value in the 'problem' column.
    '''
    return df['problem'].max() if 'problem' in df.columns else 0

def read_csv_file(
    csv_file_path: str,
    file_name: str = ''
) -> list[str]:
    '''
    Read a CSV file and return its contents as a list of dictionaries.
    Args:
        csv_file_path (str): The path to the CSV file.
    Returns:
        list: A list of dictionaries representing the rows in the CSV file.
    '''
    return pd.read_csv(csv_file_path + file_name)

def sort_df_by_problem_column(
    df: pd.DataFrame
) -> pd.DataFrame:
    '''
    Sort a DataFrame by the 'problem' column.
    Args:
        df (pd.DataFrame): The DataFrame to sort.
    Returns:
        pd.DataFrame: The sorted DataFrame.
    '''
    return df.sort_values(by='problem').reset_index(drop=True)

def get_unique_problems(
    df: pd.DataFrame
) -> list[str]:
    '''
    Get a list of unique problems from the 'problem' column of a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing the 'problem' column.
    Returns:
        list: A list of unique problems.
    '''
    return df['problem'].unique().tolist()

def create_dir_on_path(
    dir_path: str
) -> None:
    '''
    Create a directory at the specified path if it does not already exist.
    Args:
        dir_path (str): The path where the directory should be created.
    '''
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def write_java_file_on_path(
    file_path: str,
    is_plagiarized: bool,
    comb: str,
    plag_dir: str = '',
    non_plag_dir: str = ''
) -> None:
    '''
    Write Java files to the specified directories based on whether they are plagiarized or not.
    Args:
        file_path (str): The path to the source file containing Java code.
        is_plagiarized (bool): A flag indicating whether the file is plagiarized (1) or not (0).
        comb (str): The combination of sub1 and sub2 used to identify the files.
        plag_dir (str): The directory where plagiarized files should be written.
        non_plag_dir (str): The directory where non-plagiarized files should be written.
    Returns:
        None
    '''
    dst_dir = plag_dir if is_plagiarized else non_plag_dir
    written_files = set(os.listdir(dst_dir))

    for file_name in os.listdir(comb):
        file_path = os.path.join(comb, file_name)
        if file_name in written_files:
            print(f"File {file_name} already exists in {dst_dir}, skipping.")
            continue
        dst_file_path = os.path.join(dst_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as src_file:
                with open(dst_file_path, 'w', encoding='utf-8') as dst_file:
                    dst_file.write(src_file.read())
            written_files.add(file_name)
            print(f"Wrote file: {dst_file_path}")
        except (IOError) as e:
            print(f"Error writing plagiarized file {file_name}: {e}")
            continue

def apply_create_java_files(
    filtered_rows: pd.DataFrame,
    src_csv_path: str,
    plag_dir: str,
    non_plag_dir: str
) -> None:
    '''
    Apply the creation of Java files based on the filtered rows from the DataFrame.
    Args:
        filtered_rows (pd.DataFrame): The DataFrame \
            containing filtered rows with 'sub1', 'sub2', and 'verdict' columns.
        src_csv_path (str): The path to the source CSV file containing data about cases.
        plag_dir (str): The directory where plagiarized files should be written.
        non_plag_dir (str): The directory where non-plagiarized files should be written.
    Returns:
        None
    '''
    # Iterate through the filtered rows access in src_csv_path
    for _, row in filtered_rows.iterrows():
        # Get the sub1 and sub2 values
        sub1 = row['sub1']
        sub2 = row['sub2']
        is_plagiarized = 1 if int(row['verdict']) == 1 else 0
        one_comb = os.path.join(
            src_csv_path, f"{sub1}_{sub2}"
        )
        sec_comb = os.path.join(
            src_csv_path, f"{sub2}_{sub1}"
        )
        print(f"Processing combinations: {one_comb} and {sec_comb}")
        # Check if combinations exists
        if os.path.exists(one_comb):
            # Read files and write them
            write_java_file_on_path(
                src_csv_path,
                is_plagiarized,
                one_comb,
                plag_dir,
                non_plag_dir
            )
        if os.path.exists(sec_comb):
            # Read files and write them
            write_java_file_on_path(
                src_csv_path,
                is_plagiarized,
                sec_comb,
                plag_dir,
                non_plag_dir
            )

def augment_data_directory(
    src_csv_path: str
) -> None:
    '''
    Augment the data directory by creating case
    directories and populating them with plagiarized and non-plagiarized files.
    Args:
        src_csv_path (str): The path to the source CSV file containing data about cases.
    Returns:
        None
    '''
    # Extract data/cases folder
    src_path: str = os.path.join(
        os.getcwd()
        , '..', 'data', 'cases'
    )
    # Get the max number of directories in cases folder
    max_dirs: int = get_max_directory_files_count(
        src_path
    )
    # Create DF
    data_frame_csv: pd.DataFrame = read_csv_file(
        src_csv_path,
        file_name='/labels.csv'
    )
    # Get the MAX number of problems
    unique_problems: int = get_max_problem_column(
        data_frame_csv
    )
    # Sort the DataFrame by 'problem' column
    data_frame_csv = sort_df_by_problem_column(
        data_frame_csv
    )
    start_case = max_dirs + 1
    for i in range(start_case, max_dirs + unique_problems + 1):
        # Create plagiarized and non-plagiarized folders
        case_dir = os.path.join(
            src_path, f"case-0{i}"
        )
        create_dir_on_path(case_dir)
        plag_dir = os.path.join(
            case_dir, 'plagiarized'
        )
        create_dir_on_path(plag_dir)
        non_plag_dir = os.path.join(
            case_dir, 'non-plagiarized'
        )
        create_dir_on_path(non_plag_dir)
        # Access the dataframe and point to the sub1 and sub2 columns
        problem_val = i - max_dirs
        filtered_rows = data_frame_csv[
            data_frame_csv['problem'] == problem_val
        ]
        # Create Java files in the directories
        apply_create_java_files(
            filtered_rows,
            src_csv_path + "/version_2/",
            plag_dir,
            non_plag_dir
        )
