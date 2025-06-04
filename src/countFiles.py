'''
- Count Files Module
Counts the number of .txt files in each split (train, validation, test)
and categorizes them into plagiarized and non-plagiarized based on filename.
'''

import os

def is_not_plagiarized(filename):
    '''
    Determines if a file is non-plagiarized based on its filename.
    Args:
        filename (str): The name of the file to check.
    Returns:
        bool: True if the file is non-plagiarized, False otherwise.
    '''
    # Adjust this logic based on how plagiarism is indicated
    return 'non' in filename.lower()

def count_files(
    base_path: str
)-> tuple[dict, int]:
    '''
    Counts the number of .txt files in each split (train, validation, test)
    based on path structure and categorizes them into
    plagiarized and non-plagiarized.
    Args:
        base_path (str): The base directory containing the dataset splits.
    Returns:
        tuple:
            - dict: A summary of counts for each split.
            - int: The grand total of all .txt files across splits.
    '''
    summary = {}
    grand_total, grand_plagiarized, grand_non_plagiarized = 0, 0, 0

    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(base_path, split)
        total, plagiarized, non_plagiarized = 0, 0, 0

        for case_folder in os.listdir(split_path):
            case_path = os.path.join(split_path, case_folder)
            if os.path.isdir(case_path):
                for file in os.listdir(case_path):
                    if file.endswith('.txt'):
                        total += 1
                        if is_not_plagiarized(file):
                            non_plagiarized += 1
                        else:
                            plagiarized += 1

        summary[split] = {
            'total': total,
            'plagiarized': plagiarized,
            'non_plagiarized': non_plagiarized
        }

        grand_total += total
        grand_plagiarized += plagiarized
        grand_non_plagiarized += non_plagiarized

    return summary, grand_total

# Example usage
base_dir = '/home/diego/AI_Plagiarism_Detector/matrix_data'
results, overall_total = count_files(base_dir)

for split, counts in results.items():
    print(f"{split.capitalize()}: Total={counts['total']}, "
          f"Plagiarized={counts['plagiarized']}, Non-Plagiarized={counts['non_plagiarized']}")

print(f"\nGrand Total of all .txt files: {overall_total}")
